import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import z3
from ccac.model import (ModelConfig, calculate_qbound, calculate_qdel, cca_aimd, cca_bbr,
                        cca_const, cca_copa, cca_paced, cwnd_rate_arrival,
                        epsilon_alpha, initial, loss_detected, loss_oracle,
                        make_solver, multi_flows, relate_tot)
from ccac.variables import VariableNames, Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, get_name_for_list, get_renamed_vars, get_val_list
from cegis import NAME_TEMPLATE
from cegis.util import get_raw_value
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
        s.add(v.W[t] >= v.W[t - 1])
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


def loss_deterministic(c: ModelConfig, s: MySolver, v: Variables):
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

    # s.add(v.last_decrease_f[0][0] == v.A_f[0][0] - v.L_f[0][0])
    s.add(v.last_decrease_f[0][0] == v.S_f[0][0])

    for t in range(1, c.T):
        # Const last_decrease in history
        # definition_constrs.append(
        #     last_decrease_f[0][t] == v.S_f[0][t])

        s.add(
            z3.Implies(v.c_f[0][t] < v.c_f[0][t-1],
                       v.last_decrease_f[0][t] == v.A_f[0][t] - v.L_f[0][t]))
        s.add(
            z3.Implies(v.c_f[0][t] >= v.c_f[0][t-1],
                       v.last_decrease_f[0][t] == v.last_decrease_f[0][t-1]))


def exceed_queue_defs(c: ModelConfig, s: MySolver, v: Variables):
    for t in range(c.R, c.T):
        for dt in range(c.T):
            s.add(z3.Implies(
                z3.And(dt == v.qsize_thresh, v.qbound[t-c.R][dt]),
                v.exceed_queue_f[0][t]))
            s.add(z3.Implies(
                z3.And(dt == v.qsize_thresh, z3.Not(v.qbound[t-c.R][dt])),
                z3.Not(v.exceed_queue_f[0][t])))


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

    conditional_vvars = []
    if(not c.compose):
        conditional_vvars.append(v.epsilon)
    if(c.calculate_qdel):
        # qdel[t][dt<t] is deterministic
        # qdel[t][dt>=t] is non-deterministic
        conditional_vvars.append(
            [v.qdel[t][dt] for t in range(c.T) for dt in range(t, c.T)])
    if(c.calculate_qbound):
        # qbound[t][dt<=t] is deterministic
        # qbound[t][dt>t] is non deterministic
        conditional_vvars.append(
            [v.qbound[t][dt] for t in range(c.T) for dt in range(t+1, c.T)])

    conditional_dvars = []
    if(c.calculate_qdel):
        conditional_dvars.append(
            [v.qdel[t][dt] for t in range(c.T) for dt in range(t)])
    if(c.calculate_qbound):
        conditional_dvars.append(
            [v.qbound[t][dt] for t in range(c.T) for dt in range(t+1)])

    if(c.buf_min is None):
        # No loss
        verifier_vars = flatten(
            [v.A_f[:, :history], v.c_f[:, :history], v.S_f, v.W,
             v.L_f, v.Ld_f, v.dupacks, v.alpha, conditional_vvars, v.C0])
        definition_vars = flatten(
            [v.A_f[:, history:], v.A, v.c_f[:, history:],
             v.r_f, v.S, v.L, v.timeout_f, conditional_dvars])

    elif(c.deterministic_loss):
        # Determinisitic loss
        assert c.loss_oracle
        verifier_vars = flatten(
            [v.A_f[:, :history], v.c_f[:, :history], v.S_f, v.W,
             v.L_f[:, :1], v.Ld_f[:, :c.R],
             v.dupacks, v.alpha, conditional_vvars, v.C0])
        definition_vars = flatten(
            [v.A_f[:, history:], v.A, v.c_f[:, history:],
             v.L_f[:, 1:], v.Ld_f[:, c.R:],
             v.r_f, v.S, v.L, v.timeout_f, conditional_dvars])

    else:
        # Non-determinisitic loss
        assert c.loss_oracle
        verifier_vars = flatten(
            [v.A_f[:, :history], v.c_f[:, :history], v.S_f, v.W, v.L_f,
             v.Ld_f[:, :c.R], v.dupacks, v.alpha, conditional_vvars, v.C0])
        definition_vars = flatten(
            [v.A_f[:, history:], v.A, v.c_f[:, history:], v.Ld_f[:, c.R:],
             v.r_f, v.S, v.L, v.timeout_f, conditional_dvars])

    if(c.calculate_qbound):
        definition_vars.extend(flatten(v.exceed_queue_f))
        definition_vars.extend(flatten(v.last_decrease_f))

    if(isinstance(c.buf_min, z3.ExprRef)):
        verifier_vars.append(c.buf_min)

    if(c.mode_switch):
        definition_vars.extend(flatten(v.mode_f[:, 1:]))
        verifier_vars.extend(flatten(v.mode_f[:, :1]))

    return verifier_vars, definition_vars


def setup_ccac_definitions(c, v):
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
    assert c.cca == "paced"
    cca_paced(c, s, v)  # Defs to compute rate.

    return s


def setup_ccac_for_cegis(cc: CegisConfig):
    c = ModelConfig.default()
    c.compose = True
    c.cca = "paced"
    c.simplify = False
    c.C = 100
    c.T = 9

    # Signals
    c.loss_oracle = cc.template_loss_oracle

    # environment
    c.deterministic_loss = cc.deterministic_loss
    if(cc.infinite_buffer):
        c.buf_max = None
    elif(cc.dynamic_buffer):
        c.buf_max = z3.Real('buf_size')
    else:
        c.buf_max = cc.buffer_size_multiplier * c.C * (c.R + c.D)
    c.buf_min = c.buf_max
    c.N = cc.N

    if(cc.template_queue_bound):
        c.calculate_qbound = True
    if(c.N > 1):
        c.calculate_qdel = True
    c.mode_switch = cc.template_mode_switching

    return c


def setup_cegis_basic(cc: CegisConfig):
    c = setup_ccac_for_cegis(cc)
    s = MySolver()
    v = Variables(c, s)

    ccac_domain = z3.And(*s.assertion_list)
    sd = setup_ccac_definitions(c, v)
    se = setup_ccac_environment(c, v)
    ccac_definitions = z3.And(*sd.assertion_list)
    environment = z3.And(*se.assertion_list)

    verifier_vars, definition_vars = get_cegis_vars(cc, c, v)

    return (c, s, v,
            ccac_domain, ccac_definitions, environment,
            verifier_vars, definition_vars)


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
    s.add(v.alpha < 1/5)  # Verifier only

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


def get_all_desired(
        cc: CegisConfig, c: ModelConfig, v: Variables):
    first = cc.history

    cond_list = []
    for t in range(first, c.T):
        # Queue seen by a new packet should not be
        # more than desired_queue_bound
        cond_list.append(
            v.A[t] - v.L[t] - v.S[t] <=
            cc.desired_queue_bound_multiplier * c.C * (c.R + c.D))
    bounded_queue = z3.And(*cond_list)

    fefficient = (
        v.S[-1] - v.S[first] >= cc.desired_util_f * c.C * (c.T-1-first-c.D))

    loss_list: List[z3.BoolRef] = []
    for t in range(first, c.T):
        loss_list.append(v.L[t] > v.L[t-1])
    total_losses = z3.Sum(*loss_list)
    bounded_loss = total_losses <= cc.desired_loss_bound

    # Induction invariants
    ramp_up_cwnd = v.c_f[0][-1] > v.c_f[0][first]
    ramp_down_cwnd = v.c_f[0][-1] < v.c_f[0][first]
    ramp_down_q = (v.A[-1] - v.L[-1] - v.S[-1] <
                   v.A[first] - v.L[first] - v.S[first])
    ramp_down_bq = (
        (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
        < (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))

    desired = z3.And(
        z3.Or(fefficient, ramp_up_cwnd),
        z3.Or(bounded_queue, ramp_down_bq),
        z3.Or(bounded_loss, ramp_down_bq))
    assert isinstance(desired, z3.ExprRef)

    return (desired, fefficient, bounded_queue, bounded_loss,
            ramp_up_cwnd, ramp_down_cwnd, ramp_down_q, ramp_down_bq,
            total_losses)


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

    binary_search = BinarySearch(0, c.C * c.D, c.C * c.D/64)
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
        sat = s.check()
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
    objective_rhs += v.A[-1] - v.L[-1] - v.S[-1] - (v.A[first] - v.L[first] - v.S[first])
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
    _, _ = maximize_gap(c, v, ctx, verifier)
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
        get_name_for_list(vn.W): _get_model_value(v.W),
        # get_name_for_list(vn.L): _get_model_value(v.L),
    }
    for n in range(c.N):
        cex_dict.update({
            get_name_for_list(vn.A_f[n]): _get_model_value(v.A_f[n]),
            get_name_for_list(vn.c_f[n]): _get_model_value(v.c_f[n]),
            get_name_for_list(vn.S_f[n]): _get_model_value(v.S_f[n]),
            get_name_for_list(vn.L_f[n]): _get_model_value(v.L_f[n]),
            # get_name_for_list(vn.Ld_f[n]): _get_model_value(v.Ld_f[n]),
            # get_name_for_list(vn.timeout_f[n]): _get_model_value(v.timeout_f[n]),
        })
    df = pd.DataFrame(cex_dict).astype(float)
    # Can remove this by adding queue_t as a definition variable...
    # This would also allow easily quiering this from generator
    queue_t = []
    for t in range(c.T):
        queue_t.append(
            get_raw_value(counter_example.eval(
                v.A[t] - v.L[t] - v.S[t])))
    df["queue_t"] = queue_t

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
        df["qdelay"] = np.array(qdelay).astype(float)
        df["last_decrease_f"] = np.array(
            [counter_example.eval(x).as_fraction()
             for x in v.last_decrease_f[0]]).astype(float)
        df["exceed_queue_f"] = [-1] + \
            get_val_list(counter_example, v.exceed_queue_f[0][1:])

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
        get_name_for_list(vn.W): _get_model_value(v.W),
        # get_name_for_list(vn.L): _get_model_value(v.L),
    }
    for n in range(c.N):
        cex_dict.update({
            get_name_for_list(vn.A_f[n]): _get_model_value(v.A_f[n]),
            get_name_for_list(vn.c_f[n]): _get_model_value(v.c_f[n]),
            get_name_for_list(vn.S_f[n]): _get_model_value(v.S_f[n]),
            get_name_for_list(vn.L_f[n]): _get_model_value(v.L_f[n]),
            # get_name_for_list(vn.Ld_f[n]): _get_model_value(v.Ld_f[n]),
            # get_name_for_list(vn.timeout_f[n]): _get_model_value(v.timeout_f[n]),
        })
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


def get_desired_property_string(
        cc: CegisConfig, c: ModelConfig,
        fefficient, bounded_queue, bounded_loss,
        ramp_up_cwnd, ramp_down_bq, ramp_down_q, ramp_down_cwnd,
        total_losses,
        model: z3.ModelRef):
    conds = {
        "fefficient": fefficient,
        "bounded_queue": bounded_queue,
        "bounded_loss": bounded_loss,
        "ramp_up_cwnd": ramp_up_cwnd,
        "ramp_down_bq": ramp_down_bq,
        "ramp_down_q": ramp_down_q,
        "ramp_down_cwnd": ramp_down_cwnd,
        "total_losses": total_losses,
    }
    if(cc.dynamic_buffer):
        conds["buffer"] = c.buf_min
    cond_list = []
    for cond_name, cond in conds.items():
        cond_list.append(
            "{}={}".format(cond_name, model.eval(cond)))
    ret = ", ".join(cond_list)
    return ret
