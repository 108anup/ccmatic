import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import z3
from ccac.model import (ModelConfig, cca_paced, cwnd_rate_arrival,
                        epsilon_alpha, initial, loss_detected, loss_oracle,
                        make_solver, multi_flows, relate_tot)
from ccac.variables import VariableNames, Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import (flatten, get_name_for_list, get_renamed_vars,
                            get_val_list)
from cegis import NAME_TEMPLATE, Cegis, get_unsat_core, rename_vars
from cegis.util import get_raw_value, z3_max, z3_min
from pyz3_utils.binary_search import BinarySearch
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver
from pyz3_utils.small_denom import find_small_denom_soln

logger = logging.getLogger('verifier')
GlobalConfig().default_logger_setup(logger)


# * First write concrete $\exists\forall$ formula.
# * Decide classification of variables:
# 1. If variable has $\exists$ quantifier then it is a generator variable. If a
#    variable has $\forall$ quantifier then it can be a verifier variable or a
#    definition variable.
# 2. Technically, we can keep all $\forall$ variables as verifier variable.
#    Though, it is helpful to convert verifier variables to definition variables
#    (reduces CEGIS iterations, provides more information to generator for why
#    or why not a particular candidate works). This can be done iff the variable
#    can be computed as a deterministic function of other generator and verifier
#    variables. Ideally convert as many verifier variables to definition
#    variables as possible (if a variable needs non-deterministic choice, then
#    must keep it as verifier variable).
# * Decide classification of assertions:
# 1. If assertion only contains generator variables, then keep as search
#    constraint.
# 2. If assertion only contains verifier variables, then can keep as
#    specification or definition. Can choose based on convenience, e.g.,
#    depending on if other similar constraints need to be in definition or
#    specification.
# 3. Assertions that only involve definition variables are meant for sanity
#    only, they are not needed / should not be needed.
# 4. If assertion contains a definition variable and at least one other
#    generator or verifier variable, then this should be put in definitions iff
#    this is needed to compute the value of the definition variable in the
#    constraint. If we don't add a required constraint to definition then the
#    candidate solution can repeat, but this is very easy to spot/debug.
# 4. Err on the side of keeping assertions in specification, if we erroneously
#    add assertion into definitions, then a solution maybe erroneously pruned.
#    This is hard to debug.
# * Time per iteration concerns
# 1. For definition variables that solely depend on verifier variables. Can keep
#    those in verifier variables, and corresponding assertions in specification.


@dataclass
class SteadyStateVariable:
    name: str
    initial: z3.ArithRef
    final: z3.ArithRef
    lo: z3.ArithRef
    hi: z3.ArithRef

    def init_outside(self) -> z3.BoolRef:
        ret = z3.Or(self.initial < self.lo, self.initial > self.hi)
        assert isinstance(ret, z3.BoolRef)
        return ret

    def final_outside(self) -> z3.BoolRef:
        ret = z3.Or(self.final < self.lo, self.final > self.hi)
        assert isinstance(ret, z3.BoolRef)
        return ret

    def init_inside(self) -> z3.BoolRef:
        ret = z3.Not(self.init_outside())
        assert isinstance(ret, z3.BoolRef)
        return ret

    def final_inside(self) -> z3.BoolRef:
        ret = z3.Not(self.final_outside())
        assert isinstance(ret, z3.BoolRef)
        return ret

    def does_not_degrage(self) -> z3.BoolRef:
        ret = z3.And(
            z3.Implies(self.initial < self.lo,
                       z3.And(self.final >= self.initial,
                              self.final <= self.hi)),
            z3.Implies(self.initial > self.hi,
                       z3.And(self.final <= self.initial,
                              self.final >= self.lo)),
            z3.Implies(self.init_inside(), self.final_inside())
        )
        assert isinstance(ret, z3.BoolRef)
        return ret

    def strictly_improves(self) -> z3.BoolRef:
        ret = z3.And(
            z3.Implies(self.initial < self.lo,
                       z3.And(self.final > self.initial,
                              self.final <= self.hi)),
            z3.Implies(self.initial > self.hi,
                       z3.And(self.final < self.initial,
                              self.final >= self.lo))
        )
        assert isinstance(ret, z3.BoolRef)
        return ret


@dataclass
class DesiredContainer:
    desired_necessary: Optional[z3.BoolRef] = None
    desired_in_ss: Optional[z3.BoolRef] = None
    desired_invariant: Optional[z3.BoolRef] = None

    fefficient: Optional[z3.BoolRef] = None
    bounded_queue: Optional[z3.BoolRef] = None
    bounded_loss_count: Optional[z3.BoolRef] = None
    bounded_large_loss_count: Optional[z3.BoolRef] = None
    bounded_loss_amount: Optional[z3.BoolRef] = None

    ramp_up_cwnd: Optional[z3.BoolRef] = None
    ramp_down_cwnd: Optional[z3.BoolRef] = None

    ramp_down_queue: Optional[z3.BoolRef] = None
    ramp_up_queue: Optional[z3.BoolRef] = None

    ramp_down_bq: Optional[z3.BoolRef] = None
    ramp_up_bq: Optional[z3.BoolRef] = None

    loss_count: Optional[z3.ArithRef] = None
    large_loss_count: Optional[z3.ArithRef] = None
    loss_amount: Optional[z3.ArithRef] = None

    steady_state_variables: Optional[List[SteadyStateVariable]] = None
    steady_state_exists: Optional[z3.BoolRef] = None

    atleast_one_outside: Optional[z3.BoolRef] = None
    none_degrade: Optional[z3.BoolRef] = None
    atleast_one_moves_inside: Optional[z3.BoolRef] = None

    init_inside: Optional[z3.BoolRef] = None
    final_inside: Optional[z3.BoolRef] = None

    fast_decrease: Optional[z3.BoolRef] = None
    fast_increase: Optional[z3.BoolRef] = None

    def rename_vars(self, var_list: List[z3.ExprRef], template: str):
        conds = {
            "fefficient": self.fefficient,
            "bounded_queue": self.bounded_queue,
            "bounded_loss_count": self.bounded_loss_count,
            "bounded_loss_amount": self.bounded_loss_amount,
            "ramp_up_cwnd": self.ramp_up_cwnd,
            "ramp_down_cwnd": self.ramp_down_cwnd,
            "ramp_up_bq": self.ramp_up_bq,
            "ramp_down_bq": self.ramp_down_bq,
            "ramp_up_queue": self.ramp_up_queue,
            "ramp_down_queue": self.ramp_down_queue,
            "loss_count": self.loss_count,
            "loss_amount": self.loss_amount,
        }
        for attr_name, cond in conds.items():
            if(isinstance(cond, bool)):
                continue
            new_cond = rename_vars(cond, var_list, template)
            setattr(self, attr_name, new_cond)

    def to_string(self, cc: CegisConfig,
                  c: ModelConfig, model: z3.ModelRef) -> str:
        conds = {
            "fefficient": self.fefficient,
            "bounded_queue": self.bounded_queue,
            "bounded_loss_count": self.bounded_loss_count,
            "bounded_large_loss_count": self.bounded_large_loss_count,
            "bounded_loss_amount": self.bounded_loss_amount,
            "ramp_up_cwnd": self.ramp_up_cwnd,
            "ramp_down_cwnd": self.ramp_down_cwnd,
            "ramp_up_bq": self.ramp_up_bq,
            "ramp_down_bq": self.ramp_down_bq,
            "ramp_up_queue": self.ramp_up_queue,
            "ramp_down_queue": self.ramp_down_queue,
            "loss_count": self.loss_count,
            "large_loss_count": self.large_loss_count,
            "loss_amount": self.loss_amount,

            "atleast_one_outside": self.atleast_one_outside,
            "none_degrade": self.none_degrade,
            "atleast_one_moves_inside": self.atleast_one_moves_inside,

            "init_inside": self.init_inside,
            "final_inside": self.final_inside,

            "fast_decrease": self.fast_decrease,
            "fast_increase": self.fast_increase
        }

        def get_val(cond):
            if(isinstance(cond, bool)):
                return cond
            else:
                return model.eval(cond)

        cond_list = []
        term_count = 0
        for cond_name, cond in conds.items():
            if(cond is not None):
                cond_list.append(
                    "{}={}".format(cond_name, get_val(cond)))
                term_count += 1
                if(term_count % 6 == 0):
                    cond_list.append("\n")
        ret = ", ".join(cond_list)
        if(self.steady_state_variables):
            for sv in self.steady_state_variables:
                ret += "\n{}: [{}, {}]".format(
                    sv.name, model.eval(sv.lo), model.eval(sv.hi))
        return ret


def monotone_env(c, s, v):
    for t in range(1, c.T):
        for n in range(c.N):
            # Definition only (assertion really not needed). Keeping in defs.
            # In reality since arrival is max of prev and new value, this
            # constriant should never be broken anyway.
            # s.add(v.A_f[n][t] >= v.A_f[n][t - 1])

            # Verifier only. Keep part of specification.
            s.add(v.S_f[n][t] >= v.S_f[n][t - 1])

            # For loss detected and loss. If non-deterministic then, part of
            # specification, otherwise not needed in defs so keep in
            # specification always.
            s.add(v.Ld_f[n][t] >= v.Ld_f[n][t - 1])
            s.add(v.L_f[n][t] >= v.L_f[n][t - 1])

            # For non-deterministic loss, loss may not be feasible for new
            # arrival. For deterministic loss, not needed in defs.
            s.add(
                v.A_f[n][t] - v.L_f[n][t] >= v.A_f[n][t - 1] - v.L_f[n][t - 1])

        # Verifier only. Keep part of specification.
        if(hasattr(v, 'W')):
            s.add(v.W[t] >= v.W[t - 1])
            if(hasattr(v, 'C0')):
                s.add(v.C0 + c.C * t - v.W[t] >= v.C0 + c.C * (t-1) - v.W[t-1])


def monotone_defs(c, s, v):
    for t in range(1, c.T):
        for n in range(c.N):
            s.add(v.A_f[n][t] >= v.A_f[n][t - 1])
            # s.add(v.Ld_f[n][t] >= v.Ld_f[n][t - 1])
            # s.add(v.S_f[n][t] >= v.S_f[n][t - 1])
            # s.add(v.L_f[n][t] >= v.L_f[n][t - 1])

            # s.add(
            #     v.A_f[n][t] - v.L_f[n][t] >= v.A_f[n][t - 1] - v.L_f[n][t - 1])
        # s.add(v.W[t] >= v.W[t - 1])
        # s.add(v.C0 + c.C * t - v.W[t] >= v.C0 + c.C * (t-1) - v.W[t-1])


def service_waste(c: ModelConfig, s: MySolver, v: Variables):
    for t in range(c.T):
        for n in range(c.N):
            s.add(v.S_f[n][t] <= v.A_f[n][t] - v.L_f[n][t])

        s.add(v.S[t] <= v.C0 + c.C * t - v.W[t])
        if t >= c.D:
            s.add(v.C0 + c.C * (t - c.D) - v.W[t - c.D] <= v.S[t])
        else:
            # The constraint is the most slack when black line is steepest. So
            # we'll say there was no wastage when t < 0
            s.add(v.C0 + c.C * (t - c.D) - v.W[0] <= v.S[t])

        if c.compose:
            if t > 0:
                s.add(
                    z3.Implies(v.W[t] > v.W[t - 1],
                               v.A[t] - v.L[t] <= v.C0 + c.C * t - v.W[t]))
        else:
            if t > 0:
                s.add(
                    z3.Implies(v.W[t] > v.W[t - 1],
                               v.A[t] - v.L[t] <= v.S[t] + v.epsilon))


def service_choice(c: ModelConfig, s: MySolver, v: Variables):
    assert c.N == 1
    for t in range(c.T):
        S_lower = v.C0 + c.C * (t - c.D) - v.W[0]
        if t >= c.D:
            S_lower = v.C0 + c.C * (t - c.D) - v.W[t - c.D]
        feasible_choice = z3.And(
            v.S_choice[t] <= v.A[t] - v.L[t],
            v.S_choice[t] <= v.C0 + c.C * t - v.W[t],
            v.S_choice[t] >= S_lower)
        s.add(z3.Implies(feasible_choice, v.S[t] == v.S_choice[t]))
        s.add(z3.Implies(z3.Not(feasible_choice), v.S[t] == S_lower))


def update_beliefs(c: ModelConfig, s: MySolver, v: Variables):
    for n in range(c.T):

        for et in range(1, c.T):
            # TODO: Since qdelay is enumerated, we'd have to enumerate buffer
            #  beliefs...
            # # buffer beliefs
            # s.add(v.min_buffer)
            # s.add(v.max_buffer)

            utilized = [None for _ in range(et)]
            # bandwidth beliefs (C)
            for st in range(et-1, -1, -1):
                """
                In summary C >= r * n/(n+1) always, if we additionally know that
                sending rate is higher than C then, C <= r * n/(n-1).
                """
                window = et - st
                measured_c = (v.S_f[n][et] - v.S_f[n][st]) / (window)
                s.add(v.min_c[n][et] == z3_max(
                    v.min_c[n][et-1], measured_c * window / (window + 1)))

                # TODO: Check if qdel high for packet recvd means that
                #  utilization was there or not.

                # Encodes that utilization must have been high if loss happened
                # or if queing delay was more than D
                this_utilized = z3.Or(
                    v.Ld_f[n][et] - v.Ld_f[et-1] > 0, v.qbound[et][c.D+1])
                if(st + 1 == et):
                    utilized[st] = this_utilized
                else:
                    utilized[st] = z3.And(utilized[st+1], this_utilized)

                s.add(v.max_c[n][et] == z3.If(
                    utilized[st],
                    z3_min(v.max_c[n][et-1], measured_c * window / (window - 1)),
                    v.max_c[n][et-1]))


def initial_beliefs(c: ModelConfig, s: MySolver, v: Variables):
    # TODO: Use this only if we are requiring CCA to not reset beliefs when it thinks
    #  network changed.
    for n in range(c.N):
        s.add(v.min_c[n][0] >= 0)
        s.add(v.min_c[n][0] <= c.C)
        s.add(v.max_c[n][0] >= c.C)


def loss_deterministic(c: ModelConfig, s: MySolver, v: Variables):
    """
    L[t] - L[t-1] <= Max(0, r - c)
    """
    assert c.deterministic_loss
    assert c.buf_max == c.buf_min

    if c.buf_min is not None:
        s.add(v.A[0] - v.L[0] <= v.C0 + c.C * 0 - v.W[0] + c.buf_min)
    for t in range(1, c.T):
        if c.buf_min is None:  # no loss case
            s.add(v.L[t] == v.L[0])
        else:
            s.add(z3.Implies(
                v.A[t] - v.L[t-1] > v.C0 + c.C * t - v.W[t] + c.buf_min,
                v.A[t] - v.L[t] == v.C0 + c.C * t - v.W[t] + c.buf_min))
            s.add(z3.Implies(
                v.A[t] - v.L[t-1] <= v.C0 + c.C * t - v.W[t] + c.buf_min,
                v.L[t] == v.L[t-1]))
    # L[0] is still chosen non-deterministically in an unconstrained fashion.

    if(c.buf_min is None):
        # For multiflow, L_f is ideally non-deterministic
        # When there is inf buffer however, L_f should be deterministic
        for n in range(c.N):
            for t in range(1, c.T):
                s.add(v.L_f[n][t] == v.L_f[n][0])


def loss_non_deterministic(c: ModelConfig, s: MySolver, v: Variables):
    assert not c.deterministic_loss
    for t in range(c.T):
        if c.buf_min is not None:
            if t > 0:
                s.add(
                    z3.Implies(
                        v.L[t] > v.L[t - 1], v.A[t] - v.L[t] >= v.C0 + c.C *
                        (t - 1) - v.W[t - 1] + c.buf_min
                    ))
            # L[0] is chosen non-deterministically without any constraints.
        else:
            s.add(v.L[t] == v.L[0])

        # Enforce buf_max if given
        if c.buf_max is not None:
            s.add(v.A[t] - v.L[t] <= v.C0 + c.C * t - v.W[t] + c.buf_max)


def calculate_qbound_defs(c: ModelConfig, s: MySolver, v: Variables):
    # qbound[t][dt<=t] is deterministic
    # qbound[t][dt>t] is non deterministic (including qbound[0][dt>0])
    """
           dt
         0 1 2 3
       ----------
      0| d n n n
    t 1| d d n n
      2| d d d n
      3| d d d d
    """

    # Let solver choose non-deterministically what happens for
    # t = 0, dt > 0, i.e., no constraint on qdel[0][dt>0].

    # By definition queuing delay >= 0
    for t in range(c.T):
        s.add(v.qbound[t][0])

    for t in range(1, c.T):
        for dt in range(1, c.T):
            if(dt <= t):
                s.add(
                    z3.Implies(v.S[t] == v.S[t-1],
                               v.qbound[t][dt] == v.qbound[t-1][dt]))
                s.add(
                    z3.Implies(v.S[t] != v.S[t-1],
                               v.qbound[t][dt] ==
                               (v.S[t] <= v.A[t-dt] - v.L[t-dt])))
            else:
                s.add(
                    z3.Implies(v.S[t] == v.S[t-1],
                               v.qbound[t][dt] == v.qbound[t-1][dt]))
                # Let solver choose non-deterministically what happens when
                # S[t] != S[t-1] for t-dt < 0, i.e.,
                # no constraint on qbound[t][dt>t]


def calculate_qbound_env(c: ModelConfig, s: MySolver, v: Variables):
    # These have to be part of environment as vars at t+1 and dt+1 might be
    # deterministic (def var), and those at t, dt might be non determinisitic
    # (verifier var), a new value to def var may make the non-deterministic
    # choice infeasible. This could violate below constraint. Such violation
    # needs to be allowed.

    # Needed only for non-deterministic choices, mostly a sanity constraint for
    # deterministic variables.
    for t in range(c.T):
        for dt in range(c.T-1):
            # If queuing delay at t is greater than dt+1 then
            # it is also greater than dt.
            s.add(z3.Implies(v.qbound[t][dt+1], v.qbound[t][dt]))

    for t in range(c.T-1):
        for dt in range(c.T-1):
            # Queuing delay at time (t+1) cannot be
            # greater than (queuing delay at time t) + 1.

            # Why? -> Say queuing delay at time t1 is qd1 and at time t2>t1 is
            # qd2, so the packet recvd at time t1 was sent at t1-qd1 and at that
            # recvd t2 was sent at t2-qd2. Since packets are sent in FIFO order
            # t2-qd2 >= t1-qd1. Put t2=t1+1, we get qd2 <= qd1+1, i.e., (qd at
            # time t1+1) is less than or equal to qd1+1.

            # We encode if queueing delay at time t is less than dt, then
            # queuing delay at time t+1 is less than dt+1. i.e., qd(t+1) <=
            # qd(t) + 1 < dt + 1. Recall qbound[t][dt] encodes qdelay at time t
            # is greater than or equal to dt. Hence, not(qbound[t][dt]) encodes
            # qdelay is less than dt.
            s.add(z3.Implies(
                z3.Not(v.qbound[t][dt]),
                z3.Not(v.qbound[t+1][dt+1])))


def last_decrease_defs(c: ModelConfig, s: MySolver, v: Variables):

    for n in range(c.N):
        # s.add(v.last_decrease_f[0][0] == v.A_f[0][0] - v.L_f[0][0])
        s.add(v.last_decrease_f[n][0] == v.S_f[n][0])

        for t in range(1, c.T):
            # Const last_decrease in history
            # definition_constrs.append(
            #     last_decrease_f[0][t] == v.S_f[0][t])

            s.add(
                z3.Implies(v.c_f[n][t] < v.c_f[n][t-1],
                           v.last_decrease_f[n][t] == v.A_f[n][t] - v.L_f[n][t]))
            s.add(
                z3.Implies(v.c_f[n][t] >= v.c_f[n][t-1],
                           v.last_decrease_f[n][t] == v.last_decrease_f[n][t-1]))


def exceed_queue_defs(c: ModelConfig, s: MySolver, v: Variables):
    for n in range(c.N):
        for t in range(c.R, c.T):
            for dt in range(c.T):
                s.add(z3.Implies(
                    z3.And(dt == v.qsize_thresh, v.qbound[t-c.R][dt]),
                    v.exceed_queue_f[n][t]))
                s.add(z3.Implies(
                    z3.And(dt == v.qsize_thresh, z3.Not(v.qbound[t-c.R][dt])),
                    z3.Not(v.exceed_queue_f[n][t])))


def calculate_qdel_defs(c: ModelConfig, s: MySolver, v: Variables):
    # qdel[t][dt<t] is deterministic
    # qdel[t][dt>=t] is non-deterministic (including,
    # qdel[0][dt], qdel[t][dt>t-1])
    """
           dt
         0 1 2 3
       ----------
      0| n n n n
    t 1| d n n n
      2| d d n n
      3| d d d n
    """

    # Let solver choose non-deterministically what happens for
    # t = 0, i.e., no constraint on qdel[0][dt].
    for t in range(1, c.T):
        for dt in range(c.T):
            if(dt <= t-1):
                s.add(z3.Implies(v.S[t] != v.S[t - 1],
                                 v.qdel[t][dt] == z3.And(
                    v.A[t - dt - 1] - v.L[t - dt - 1] < v.S[t],
                    v.A[t - dt] - v.L[t - dt] >= v.S[t])))
                s.add(z3.Implies(v.S[t] == v.S[t - 1],
                                 v.qdel[t][dt] == v.qdel[t-1][dt]))
            else:
                s.add(z3.Implies(v.S[t] == v.S[t - 1],
                                 v.qdel[t][dt] == v.qdel[t-1][dt]))
                # We let solver choose non-deterministically what happens when
                # S[t] != S[t-1] for dt > t-1, i.e.,
                # no constraint on qdel[t][dt>t-1]


def calculate_qdel_env(c: ModelConfig, s: MySolver, v: Variables):
    # There can be only one value for queuing delay at a given time.
    # Needed only for non-deterministic choices, mostly a sanity constraint for
    # deterministic variables.
    for t in range(c.T):
        s.add(z3.Sum(*v.qdel[t]) <= 1)

    # qdel[t][dt] is True iff queueing delay is >=dt but <dt+1
    # If queuing delay at time t1 is dt1, then queuing delay at time t2=t1+1,
    # cannot be more than dt1+1. I.e., qdel[t2][dt1+1+1] has to be false.
    # Note qdel[t2][dt1+1] can be true as queueing delay at t1+1 can be dt1+1.
    for t1 in range(c.T-1):
        for dt1 in range(c.T):
            t2 = t1+1
            # dt2 starts from dt1+1+1
            s.add(z3.Implies(
                v.qdel[t1][dt1],
                z3.And(*[z3.Not(v.qdel[t2][dt2])
                         for dt2 in range(dt1+1+1, c.T)])
            ))
            s.add(z3.Implies(
                v.qdel[t1][dt1],
                z3.Or(*[v.qdel[t2][dt2] for dt2 in range(min(c.T, dt1+1+1))])
            ))


def fifo_service(c: ModelConfig, s: MySolver, v: Variables):
    # If a flow sees service of more bytes than in queue at a particular time,
    # then each flow should have seen service of all their bytes in the queue
    # at that particular time.
    for n1 in range(c.N):
        for t1 in range(c.T):
            for t2 in range(t1+1, c.T):
                s.add(
                    z3.Implies(
                        v.S_f[n1][t2] - v.S_f[n1][t1] >
                        v.A_f[n1][t1] - v.L_f[n1][t1] - v.S_f[n1][t1],
                        z3.And(
                            *[v.S_f[n2][t2] - v.S_f[n2][t1] >=
                              v.A_f[n2][t1] - v.L_f[n2][t1] - v.S_f[n2][t1]
                              for n2 in range(c.N)])))


def setup_ccac():
    c = ModelConfig.default()
    c.compose = True
    c.cca = "paced"
    c.simplify = False
    c.calculate_qdel = False
    c.C = 100
    c.T = 9

    s = MySolver()
    v = Variables(c, s)

    # s.add(z3.And(v.S[0] <= 1000, v.S[0] >= -1000))
    return c, s, v


def get_cegis_vars(
    cc: CegisConfig, c: ModelConfig, v: Variables
) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
    history = cc.history

    verifier_vars = flatten(
            [v.A_f[:, :history], v.c_f[:, :history], v.r_f[:, :history], v.W,
             v.alpha, v.C0])
    definition_vars = flatten(
            [v.A_f[:, history:], v.A, v.c_f[:, history:],
             v.r_f[:, history:], v.S, v.L])

    if(not c.loss_oracle):
        verifier_vars.append(v.dupacks)
        definition_vars.extend(flatten(v.timeout_f))

    if(not c.compose):
        verifier_vars.append(v.epsilon)
    if(c.calculate_qdel):
        # qdel[t][dt<t] is deterministic
        # qdel[t][dt>=t] is non-deterministic
        verifier_vars.extend(flatten(
            [v.qdel[t][dt] for t in range(c.T) for dt in range(t, c.T)]))
    if(c.calculate_qbound):
        # qbound[t][dt<=t] is deterministic
        # qbound[t][dt>t] is non deterministic
        verifier_vars.extend(flatten(
            [v.qbound[t][dt] for t in range(c.T) for dt in range(t+1, c.T)]))
        definition_vars.extend(flatten(v.exceed_queue_f))
        definition_vars.extend(flatten(v.last_decrease_f))

    if(c.calculate_qdel):
        definition_vars.extend(flatten(
            [v.qdel[t][dt] for t in range(c.T) for dt in range(t)]))
    if(c.calculate_qbound):
        definition_vars.extend(flatten(
            [v.qbound[t][dt] for t in range(c.T) for dt in range(t+1)]))

    if(c.buf_min is None):
        # No loss
        verifier_vars.extend(flatten([v.L_f, v.Ld_f]))
    elif(c.deterministic_loss):
        # Determinisitic loss
        assert c.loss_oracle
        verifier_vars.extend(flatten([v.L_f[:, :1], v.Ld_f[:, :c.R]]))
        definition_vars.extend(flatten([v.L_f[:, 1:], v.Ld_f[:, c.R:]]))
    else:
        # Non-determinisitic loss
        assert c.loss_oracle
        verifier_vars.extend(flatten([v.L_f, v.Ld_f[:, :c.R]]))
        definition_vars.extend(flatten([v.Ld_f[:, c.R:]]))

    if(isinstance(c.buf_min, z3.ExprRef)):
        verifier_vars.append(c.buf_min)

    if(cc.feasible_response):
        verifier_vars.extend(flatten(v.S_choice))
        definition_vars.extend(flatten(v.S_f))
    else:
        verifier_vars.extend(flatten(v.S_f))

    if(c.mode_switch):
        definition_vars.extend(flatten(v.mode_f[:, 1:]))
        verifier_vars.extend(flatten(v.mode_f[:, :1]))

    return verifier_vars, definition_vars


def setup_ccac_definitions(c: ModelConfig, v: Variables):
    s = MySolver()
    s.warn_undeclared = False

    monotone_defs(c, s, v)  # Def only. Constraint not needed.
    initial(c, s, v)  # Either definition vars or verifier vars.
    # Keep as definitions for convenience.
    relate_tot(c, s, v)  # Definitions required to compute tot variables.
    if(c.deterministic_loss):
        loss_deterministic(c, s, v)  # Def to compute loss.
    if(c.loss_oracle):
        loss_oracle(c, s, v)  # Defs to compute loss detected.
    if(c.calculate_qdel):
        calculate_qdel_defs(c, s, v)  # Defs to compute qdel.
    if(c.calculate_qbound):
        calculate_qbound_defs(c, s, v)  # Defs to compute qbound.
        last_decrease_defs(c, s, v)
        exceed_queue_defs(c, s, v)
    cwnd_rate_arrival(c, s, v)  # Defs to compute arrival.
    if(c.cca == "paced"):
        cca_paced(c, s, v)  # Defs to compute rate.
    if(c.feasible_response):
        service_choice(c, s, v)

    return s


def setup_ccac_for_cegis(cc: CegisConfig):
    c = ModelConfig.default()
    c.compose = True
    c.cca = cc.cca
    c.simplify = False
    c.C = cc.C
    c.T = cc.T
    c.R = cc.R
    c.D = cc.D
    c.compose = cc.compose

    # Add a prefix to all names so we can have multiple Variables instances
    # in one solver
    if cc.name is None:
        pre = ""
    else:
        pre = cc.name + "__"

    # Signals
    c.loss_oracle = cc.template_loss_oracle

    # environment
    c.deterministic_loss = cc.deterministic_loss
    if(cc.infinite_buffer):
        c.buf_max = None
    elif(cc.dynamic_buffer):
        c.buf_max = z3.Real(f'{pre}buf_size')
    else:
        c.buf_max = cc.buffer_size_multiplier * c.C * (c.R + c.D)
    c.buf_min = c.buf_max
    c.N = cc.N

    if(cc.template_queue_bound):
        c.calculate_qbound = True
    if(c.N > 1 or cc.template_qdel):
        c.calculate_qdel = True
    c.mode_switch = cc.template_mode_switching
    c.feasible_response = cc.feasible_response

    return c


def setup_cegis_basic(cc: CegisConfig, name=None):
    if(name is not None):
        cc.name = name
    c = setup_ccac_for_cegis(cc)
    s = MySolver()
    s.warn_undeclared = False
    v = Variables(c, s, cc.name)

    ccac_domain = z3.And(*s.assertion_list)
    sd = setup_ccac_definitions(c, v)
    se = setup_ccac_environment(c, v)
    ccac_definitions = z3.And(*sd.assertion_list)
    # not_too_adversarial_init_cwnd(cc, c, se, v)
    environment = z3.And(*se.assertion_list)

    verifier_vars, definition_vars = get_cegis_vars(cc, c, v)

    return (c, s, v,
            ccac_domain, ccac_definitions, environment,
            verifier_vars, definition_vars)


def not_too_adversarial_init_cwnd(
        cc: CegisConfig, c: ModelConfig, s: MySolver, v: Variables):
    for n in range(c.N):
        for t in range(1, cc.history):
            s.add(v.c_f[n][t] <= v.c_f[n][t-1] * 3/2)


def setup_ccac_environment(c, v):
    s = MySolver()
    s.warn_undeclared = False
    monotone_env(c, s, v)
    service_waste(c, s, v)
    if(not c.deterministic_loss):
        loss_non_deterministic(c, s, v)
    if(not c.loss_oracle):
        loss_detected(c, s, v)
    if(c.calculate_qdel):
        calculate_qdel_env(c, s, v)  # How to pick initial qdels
        # non-deterministically.
    if(c.calculate_qbound):
        calculate_qbound_env(c, s, v)  # How to pick initial qbounds
        # non-deterministically.
    if(c.N > 1):
        multi_flows(c, s, v)  # Flows should be serviced fairly
        fifo_service(c, s, v)
    epsilon_alpha(c, s, v)  # Verifier only

    # Shouldn't be any loss at t0 otherwise cwnd is high and q is still 0.
    # s.add(v.L_f[0][0] == 0)

    # Remove periodicity, as generator overfits and produces monotonic CCAs.
    # make_periodic(c, s, v, c.R + c.D)

    # Avoid weird cases where single packet is larger than BDP.
    s.add(v.alpha < max(0.2, (c.C * c.R)/5))  # Verifier only

    # Buffer should at least have one packet
    if(isinstance(c.buf_min, z3.ExprRef)):
        s.add(c.buf_min > v.alpha)

        # Buffer taken from discrete choices
        # s.add(z3.Or(c.buf_min == c.C * (c.R + c.D),
        #             c.buf_min == 0.1 * c.C * (c.R + c.D)))
    return s


def setup_ccac_full(cca="copa"):
    c = ModelConfig.default()
    c.compose = True
    c.cca = cca
    c.simplify = False
    c.calculate_qdel = True
    # c.S0 = 0
    # c.L0 = 0
    # c.W0 = 1000
    c.C = 100
    c.T = 7
    s, v = make_solver(c)
    # Consider the no loss case for simplicity
    s.add(v.L[0] == v.L[-1])
    # make_periodic(c, s, v, c.R + c.D)
    # Avoid weird cases where single packet is larger than BDP.
    s.add(v.alpha < 1/5)
    return c, s, v


def get_periodic_constraints(cc: CegisConfig, c: ModelConfig, v: Variables):
    periodic = []

    # # In beginning and in end,
    # # For each flow, packet queue and cwnd should be same
    # # Token queue should be same
    # last = cc.T-1 - (cc.history-1)
    # for h in range(cc.history):
    #     for n in range(cc.N):
    #         periodic.append(v.c_f[n][h] == v.c_f[n][last+h])
    #         periodic.append(
    #             v.A_f[n][h] - v.L_f[n][h] - v.S_f[n][h] ==
    #             v.A_f[n][last+h] - v.L_f[n][last+h] - v.S_f[n][last+h])
    #     periodic.append(v.C0 + c.C * h - v.W[h] - v.S[h] ==
    #                     v.C0 + c.C * (last+h) - v.W[last+h] - v.S[last+h])

    # First decided by CCA is equal to last decided by CCA
    for n in range(cc.N):
        periodic.append(v.c_f[n][cc.history] == v.c_f[n][c.T-1])
        periodic.append(
            v.A_f[n][cc.history] - v.L_f[n][cc.history] - v.S_f[n][cc.history] ==
            v.A_f[n][c.T-1] - v.L_f[n][c.T-1] - v.S_f[n][c.T-1])
    periodic.append(v.C0 + c.C * cc.history - v.W[cc.history] - v.S[cc.history] ==
                    v.C0 + c.C * (c.T-1) - v.W[c.T-1] - v.S[c.T-1])

    return z3.And(*periodic)


def ramp_up_when_cwnd_reset_fi(cc: CegisConfig, c: ModelConfig, v: Variables):
    first = cc.history

    total_first_cwnd = 0
    total_second_cwnd = 0
    total_second_last_cwnd = 0
    total_last_cwnd = 0
    assert c.T-2 > first+1
    for n in range(c.N):
        total_first_cwnd += v.c_f[n][first]
        total_second_cwnd += v.c_f[n][first+1]
        total_second_last_cwnd += v.c_f[n][c.T-2]
        total_last_cwnd += v.c_f[n][c.T-1]
    assert isinstance(total_first_cwnd, z3.ArithRef)
    assert isinstance(total_second_cwnd, z3.ArithRef)
    assert isinstance(total_second_last_cwnd, z3.ArithRef)
    assert isinstance(total_last_cwnd, z3.ArithRef)

    min_initial_cwnd = z3_min(total_first_cwnd, total_second_cwnd)
    min_final_cwnd = z3_min(total_second_last_cwnd, total_last_cwnd)
    assert isinstance(min_initial_cwnd, z3.ArithRef)
    assert isinstance(min_final_cwnd, z3.ArithRef)

    ramp_up_cwnd = min_final_cwnd > min_initial_cwnd
    ramp_down_cwnd = min_final_cwnd < min_initial_cwnd
    return ramp_up_cwnd, ramp_down_cwnd


def get_desired_necessary(
        cc: CegisConfig, c: ModelConfig, v: Variables):
    first = cc.history

    d = get_desired_in_ss(cc, c, v)

    # Induction invariants
    total_final_cwnd = 0
    total_initial_cwnd = 0
    for n in range(c.N):
        total_initial_cwnd += v.c_f[n][first]
        total_final_cwnd += v.c_f[n][-1]

    total_final_rate = 0
    total_initial_rate = 0
    for n in range(c.N):
        total_initial_rate += v.r_f[n][first]
        total_final_rate += v.r_f[n][-1]

    # TODO: check if this is the right invariant for rate based CCAs.
    #  For window based CCAs, pacing is const, so this should be fine.
    d.ramp_up_cwnd = z3.And(
        total_final_cwnd > total_initial_cwnd,
        total_final_rate > total_initial_rate)
    if(cc.template_fi_reset):
        ru, rd = ramp_up_when_cwnd_reset_fi(cc, c, v)
        d.ramp_up_cwnd = ru
        d.ramp_down_cwnd = rd
    d.ramp_down_cwnd = z3.And(
        total_final_cwnd < total_initial_cwnd,
        total_final_rate < total_initial_rate)

    d.ramp_down_queue = (v.A[-1] - v.L[-1] - v.S[-1] <
                         v.A[first] - v.L[first] - v.S[first])
    d.ramp_up_queue = (v.A[-1] - v.L[-1] - v.S[-1] >
                       v.A[first] - v.L[first] - v.S[first])

    if(hasattr(v, 'C0') and hasattr(v, 'W')):
        d.ramp_down_bq = (
            (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
            < (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))
        d.ramp_up_bq = (
            (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
            > (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))
    else:
        # For ideal link
        d.ramp_down_bq = False
        d.ramp_up_bq = False

    d.desired_necessary = z3.And(
        z3.Or(d.fefficient, d.ramp_up_cwnd,
              d.ramp_up_queue, d.ramp_up_bq),
        z3.Or(d.bounded_queue, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq),
        z3.Or(d.bounded_loss_count, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq),
        z3.Or(d.bounded_large_loss_count, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq),
        z3.Or(d.bounded_loss_amount, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq))

    return d


def get_desired_in_ss(cc: CegisConfig, c: ModelConfig, v: Variables):
    d = DesiredContainer()

    first = cc.history
    cond_list = []
    for t in range(first, c.T):
        # Queue seen by a new packet should not be
        # more than desired_queue_bound
        cond_list.append(
            v.A[t] - v.L[t] - v.S[t] <=
            cc.desired_queue_bound_multiplier * c.C * (c.R + c.D)
            + cc.desired_queue_bound_alpha * v.alpha)
    d.bounded_queue = z3.And(*cond_list)

    assert first >= 1
    d.fefficient = (
        v.S[-1] - v.S[first-1] >= cc.desired_util_f * c.C * (c.T-1-(first-1)-c.D))

    loss_list: List[z3.BoolRef] = []
    for t in range(first, c.T):
        loss_list.append(v.L[t] > v.L[t-1])
    d.loss_count = z3.Sum(*loss_list)
    d.bounded_loss_count = d.loss_count <= cc.desired_loss_count_bound

    large_loss_list: List[z3.BoolRef] = []
    for t in range(first, c.T):
        large_loss_list.append(v.L[t] > v.L[t-1] + v.alpha * cc.desired_loss_amount_bound_alpha)
    d.large_loss_count = z3.Sum(*large_loss_list)
    d.bounded_large_loss_count = d.large_loss_count <= cc.desired_large_loss_count_bound

    d.loss_amount = v.L[-1] - v.L[first]
    d.bounded_loss_amount = (
        d.loss_amount <=
        cc.desired_loss_amount_bound_multiplier * (c.C * (c.R + c.D))
        + cc.desired_loss_amount_bound_alpha * v.alpha)


    d.desired_in_ss = z3.And(d.fefficient, d.bounded_queue,
                             d.bounded_loss_count, d.bounded_loss_amount)
    return d


def get_desired_ss_invariant(cc: CegisConfig, c: ModelConfig, v: Variables):
    d = get_desired_in_ss(cc, c, v)

    total_final_cwnd = 0
    total_initial_cwnd = 0
    for n in range(c.N):
        total_initial_cwnd += v.c_f[n][cc.history]
        total_final_cwnd += v.c_f[n][-1]

    assert(isinstance(total_final_cwnd, z3.ArithRef))
    assert(isinstance(total_initial_cwnd, z3.ArithRef))
    d.steady_state_variables = [
        SteadyStateVariable(
            'cwnd',
            total_initial_cwnd,
            total_final_cwnd,
            z3.Int('SSThresh_cwnd_lo'),
            z3.Int('SSThresh_cwnd_hi')),

        # Queue
        SteadyStateVariable(
            'queue',
            v.A[cc.history] - v.L[cc.history] - v.S[cc.history],
            v.A[c.T-1] - v.L[c.T-1] - v.S[c.T-1],
            z3.Int('SSThresh_queue_lo'),
            z3.Int('SSThresh_queue_hi'))

        # # Bottleneck Queue
        # SteadyStateVariable(
        #     'queue',
        #     v.A[cc.history] - v.L[cc.history]
        #     - (v.C0 + c.C * (cc.history) - v.W[cc.history]),
        #     v.A[c.T-1] - v.L[c.T-1]
        #     - (v.C0 + c.C * (c.T-1) - v.W[c.T-1]),
        #     z3.Int('SSThresh_queue_lo'),
        #     z3.Int('SSThresh_queue_hi'))
    ]
    d.steady_state_exists = get_steady_state_definitions(
        cc, c, v, d)
    inside_ss = z3.And(*[sv.init_inside()
                         for sv in d.steady_state_variables])

    d.desired_invariant = z3.And(d.steady_state_exists,
                                 z3.Implies(inside_ss, d.desired_in_ss))
    return d


def get_steady_state_definitions(
        cc: CegisConfig, c: ModelConfig, v: Variables,
        d: DesiredContainer):
    assert d.steady_state_variables
    assertions = []

    # # Initial in SS implies Final in SS
    # for sv in steady_state_variables:
    #     assertions.append(z3.Implies(
    #         z3.And(sv.initial >= sv.lo, sv.initial <= sv.hi),
    #         z3.And(sv.final >= sv.lo, sv.final <= sv.hi)))

    # At least one outside
    #     IMPLIES
    #         none should degrade AND
    #         atleast one that is outside must move towards inside
    d.atleast_one_outside = z3.Or(
        *[sv.init_outside() for sv in d.steady_state_variables])
    d.none_degrade = z3.And(*[sv.does_not_degrage()
                              for sv in d.steady_state_variables])
    d.atleast_one_moves_inside = \
        z3.Or(*[z3.And(sv.init_outside(), sv.strictly_improves())
                for sv in d.steady_state_variables])
    assertions.append(z3.Implies(
        d.atleast_one_outside,
        z3.And(
            d.none_degrade,
            d.atleast_one_moves_inside)))

    # All inside
    #    IMPLIES
    #         All remain inside
    d.init_inside = z3.And(*[sv.init_inside() for sv in d.steady_state_variables])
    d.final_inside = z3.And(*[sv.final_inside() for sv in d.steady_state_variables])
    assertions.append(z3.Implies(
        d.init_inside,
        d.final_inside
    ))

    ret = z3.And(*assertions)
    assert isinstance(ret, z3.BoolRef)
    return ret


def maximize_gap(
    c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    """
    Adds constraints as a side effect to the verifier formulation
    that try to maximize the gap between
    service curve and the upper bound of service curve (based on waste choices).
    """
    s = verifier

    orig_sat = s.check()
    if str(orig_sat) != "sat":
        return orig_sat, None

    orig_model = s.model()
    cur_min = np.inf
    for t in range(1, c.T):
        if(orig_model.eval(v.W[t] > v.W[t-1])):
            this_gap = get_raw_value(orig_model.eval(
                v.C0 + c.C * t - v.W[t] - v.S[t]))
            cur_min = min(cur_min, this_gap)

    if(cur_min == np.inf):
        return orig_sat, orig_model

    if(cur_min < 0):
        # This is basically in case of multi-link. Where this link may not be
        # satisfying the environment constraints.
        return orig_sat, orig_model

    eps = max(1, c.C * c.D)/64
    binary_search = BinarySearch(0, c.C * c.D, eps)
    min_gap = z3.Real('min_gap', ctx=ctx)

    for t in range(1, c.T):
        s.add(z3.Implies(
            v.W[t] > v.W[t-1],
            min_gap <= v.C0 + c.C * t - v.W[t] - v.S[t]))

    while True:
        pt = binary_search.next_pt()
        if pt is None:
            break

        s.push()
        s.add(min_gap >= pt)
        try:
            sat = s.check()
        except z3.z3types.Z3Exception as e:
            logger.error(f"During binary search for max gap,"
                         f" verifier threw error: {e}")
            logger.info("Defaulting to using unoptimized model")
            s.pop()
            return orig_sat, orig_model
        s.pop()

        if(str(sat) == 'sat'):
            binary_search.register_pt(pt, 1)
        elif str(sat) == "unknown":
            binary_search.register_pt(pt, 2)
        else:
            assert str(sat) == "unsat", f"Unknown value: {str(sat)}"
            binary_search.register_pt(pt, 3)

    best_gap, _, _ = binary_search.get_bounds()
    s.add(min_gap >= best_gap)

    sat = s.check()
    assert sat == orig_sat
    model = s.model()

    logger.info("Improved gap from {} to {}".format(float(cur_min), best_gap))
    return sat, model


def worst_counter_example(
    first: int, c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    """
    Adds constraints as a side effect to the verifier formulation
    to get the counter example that causes generator to move the most.
    """
    s = verifier

    orig_sat = s.check()
    if str(orig_sat) != "sat":
        return orig_sat, None

    objective_rhs = 0
    for t in range(c.T):
        objective_rhs += v.A[t] - v.L[t] - v.S[t]
        objective_rhs += v.C0 + c.C * t - v.W[t] - v.S[t]
        if(t >= c.D):
            objective_rhs += v.S[t] - (v.C0 + c.C * (t-c.D) - v.W[t-c.D])
    objective_rhs += v.A[-1] - v.L[-1] - v.S[-1] - \
        (v.A[first] - v.L[first] - v.S[first])
    objective_rhs += -v.S[-1] + v.S[first]

    orig_model = s.model()
    cur_obj = get_raw_value(orig_model.eval(objective_rhs))

    binary_search = BinarySearch(0, 1e5, c.C * c.D/64)
    objective = z3.Real('objective', ctx=ctx)

    s.add(objective <= objective_rhs)

    while True:
        pt = binary_search.next_pt()
        if pt is None:
            break

        s.push()
        s.add(objective >= pt)
        sat = s.check()
        s.pop()

        if(str(sat) == 'sat'):
            binary_search.register_pt(pt, 1)
        elif str(sat) == "unknown":
            binary_search.register_pt(pt, 2)
        else:
            assert str(sat) == "unsat", f"Unknown value: {str(sat)}"
            binary_search.register_pt(pt, 3)

    best_obj, _, _ = binary_search.get_bounds()
    s.add(objective >= best_obj)

    sat = s.check()
    assert sat == orig_sat
    model = s.model()

    logger.info("Improved obj from {} to {}".format(float(cur_obj), best_obj))
    return sat, model


def run_verifier_incomplete(
    c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    # This is meant to create a partial function by
    # getting c, v, ctx from closure.
    if(hasattr(v, 'W') and hasattr(v, 'C0')):
        maximize_gap(c, v, ctx, verifier)
    sat, _, model = find_small_denom_soln(verifier, 4096)
    return sat, model


def run_verifier_incomplete_wce(
    first: int, c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    # This is meant to create a partial function by
    # getting c, v, ctx from closure.
    _, _ = maximize_gap(c, v, ctx, verifier)
    _, _ = worst_counter_example(first, c, v, ctx, verifier)
    sat, _, model = find_small_denom_soln(verifier, 4096)
    return sat, model


def get_cex_df(
    counter_example: z3.ModelRef, v: Variables,
    vn: VariableNames, c: ModelConfig
) -> pd.DataFrame:
    def _get_model_value(l):
        ret = []
        for vvar in l:
            val = get_raw_value(counter_example.eval(vvar))
            ret.append(val)
        return ret
    cex_dict = {
        get_name_for_list(vn.A): _get_model_value(v.A),
        get_name_for_list(vn.S): _get_model_value(v.S),
        # get_name_for_list(vn.L): _get_model_value(v.L),
    }
    for n in range(c.N):
        cex_dict.update({
            get_name_for_list(vn.A_f[n]): _get_model_value(v.A_f[n]),
            get_name_for_list(vn.c_f[n]): _get_model_value(v.c_f[n]),
            get_name_for_list(vn.r_f[n]): _get_model_value(v.r_f[n]),
            get_name_for_list(vn.S_f[n]): _get_model_value(v.S_f[n]),
            get_name_for_list(vn.L_f[n]): _get_model_value(v.L_f[n]),
            get_name_for_list(vn.Ld_f[n]): _get_model_value(v.Ld_f[n]),
            # get_name_for_list(vn.timeout_f[n]): _get_model_value(v.timeout_f[n]),
        })
    if(c.feasible_response):
        cex_dict.update({
            get_name_for_list(vn.S_choice): _get_model_value(v.S_choice)})
    if(hasattr(v, 'W')):
        cex_dict.update({
            get_name_for_list(vn.W): _get_model_value(v.W)})
        # cex_dict.update({
        #     'non-wasted-tokens': _get_model_value(
        #         [v.C0 + c.C * t - v.W[t] for t in range(c.T)])})
    df = pd.DataFrame(cex_dict).astype(float)
    # Can remove this by adding queue_t as a definition variable...
    # This would also allow easily quiering this from generator
    queue_t = []
    for t in range(c.T):
        queue_t.append(
            get_raw_value(counter_example.eval(
                v.A[t] - v.L[t] - v.S[t])))
    df["queue_t"] = queue_t

    if(hasattr(v, 'C0') and hasattr(v, 'W')):
        bottle_queue_t = []
        for t in range(c.T):
            bottle_queue_t.append(
                get_raw_value(counter_example.eval(
                    v.A[t] - v.L[t] - (v.C0 + c.C * t - v.W[t])
                )))
        df["bottle_queue_t"] = bottle_queue_t

    if(c.calculate_qbound):
        qdelay = []
        for t in range(c.T):
            assert bool(counter_example.eval(v.qbound[t][0]))
            for dt in range(c.T-1, -1, -1):
                value = counter_example.eval(v.qbound[t][dt])
                try:
                    bool_value = bool(value)
                except z3.z3types.Z3Exception:
                    bool_value = True
                if(bool_value):
                    qdelay.append(dt+1)
                    break
        assert len(qdelay) == c.T
        df["qdelay_t"] = np.array(qdelay).astype(float)
        for n in range(c.N):
            df[f"last_decrease_f_{n},t"] = np.array(
                [counter_example.eval(x).as_fraction()
                 for x in v.last_decrease_f[n]]).astype(float)
            df[f"exceed_queue_f_{n},t"] = [-1] + \
                get_val_list(counter_example, v.exceed_queue_f[n][1:])

        # qbound_vals = []
        # for qbound_list in v.qbound:
        #     qbound_val_list = get_val_list(counter_example, qbound_list)
        #     qbound_vals.append(qbound_val_list)
        # ret += "\n{}".format(np.array(qbound_vals))
        # ret += "\n{}".format(counter_example.eval(qsize_thresh))

    return df.astype(float)


def get_gen_cex_df(
    solution: z3.ModelRef, v: Variables, vn: VariableNames,
    n_cex: int, c: ModelConfig
) -> pd.DataFrame:
    if(n_cex == 0):
        return pd.DataFrame()
    name_template = NAME_TEMPLATE + str(n_cex)
    ctx = solution.ctx

    def _get_model_value(l):
        ret = []
        for vvar in l:
            cex_vvar_name = name_template.format(vvar)
            cex_var = z3.Const(cex_vvar_name, vvar.sort())
            ret.append(get_raw_value(solution.eval(cex_var)))
        return ret
    cex_dict = {
        get_name_for_list(vn.A): _get_model_value(v.A),
        get_name_for_list(vn.S): _get_model_value(v.S),
        # get_name_for_list(vn.L): _get_model_value(v.L),
    }
    for n in range(c.N):
        cex_dict.update({
            get_name_for_list(vn.A_f[n]): _get_model_value(v.A_f[n]),
            get_name_for_list(vn.c_f[n]): _get_model_value(v.c_f[n]),
            get_name_for_list(vn.r_f[n]): _get_model_value(v.r_f[n]),
            get_name_for_list(vn.S_f[n]): _get_model_value(v.S_f[n]),
            get_name_for_list(vn.L_f[n]): _get_model_value(v.L_f[n]),
            # get_name_for_list(vn.Ld_f[n]): _get_model_value(v.Ld_f[n]),
            # get_name_for_list(vn.timeout_f[n]): _get_model_value(v.timeout_f[n]),
        })
    if(hasattr(v, 'W')):
        cex_dict.update({
            get_name_for_list(vn.W): _get_model_value(v.W)})

    df = pd.DataFrame(cex_dict).astype(float)

    if(c.calculate_qbound):
        qdelay = []
        for t in range(c.T):
            g_qbound_t = get_renamed_vars(v.qbound[t], n_cex)
            assert bool(solution.eval(g_qbound_t[0]))
            for dt in range(c.T-1, -1, -1):
                value = solution.eval(g_qbound_t[dt])
                try:
                    bool_value = bool(value)
                except z3.z3types.Z3Exception:
                    bool_value = True
                if(bool_value):
                    qdelay.append(dt+1)
                    break
        assert len(qdelay) == c.T
        df["qdelay"] = np.array(qdelay).astype(float)

        g_last_decrease_f = get_renamed_vars(v.last_decrease_f[0], n_cex)
        df["last_decrease_f"] = np.array(
            [solution.eval(x).as_fraction()
             for x in g_last_decrease_f]).astype(float)

        g_exceed_queue_f = get_renamed_vars(v.exceed_queue_f[0], n_cex)
        df["exceed_queue_f"] = [-1] + \
            get_val_list(solution, g_exceed_queue_f[1:])

        # qbound_vals = []
        # for qbound_list in v.qbound:
        #     qbound_val_list = get_val_list(solution, _get_renamed(qbound_list))
        #     qbound_vals.append(qbound_val_list)
        # ret += "\n{}".format(np.array(qbound_vals))
    return df
