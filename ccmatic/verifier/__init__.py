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
from ccmatic.common import flatten, get_name_for_list
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
    c: ModelConfig, v: Variables, history: int
) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:

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
    cwnd_rate_arrival(c, s, v)  # Defs to compute arrival.
    assert c.cca == "paced"
    cca_paced(c, s, v)  # Defs to compute rate.

    return s


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
    epsilon_alpha(c, s, v)  # Verifier only

    # Shouldn't be any loss at t0 otherwise cwnd is high and q is still 0.
    # s.add(v.L_f[0][0] == 0)

    # Remove periodicity, as generator overfits and produces monotonic CCAs.
    # make_periodic(c, s, v, c.R + c.D)

    # Avoid weird cases where single packet is larger than BDP.
    s.add(v.alpha < 1/5)  # Verifier only
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


def desired_high_util_low_delay(c, v, first, util_frac, delay_bound):
    cond_list = []
    for t in range(first, c.T):
        cond_list.append(v.A[t] - v.L[t] - v.S[t] <= delay_bound)
    # Queue seen by a new packet should not be more that delay_bound
    low_delay = z3.And(*cond_list)
    # Serviced should be at least util_frac that could have been serviced
    high_util = v.S[-1] - v.S[first] >= util_frac * c.C * (c.T-1-first-c.D)
    # If the cwnd0 is very low then CCA should increase cwnd
    ramp_up = z3.Or(*[v.c_f[n][-1] > v.c_f[n][first] for n in range(c.N)])
    ramp_down = v.c_f[0][-1] < v.c_f[0][first]
    # If the queue is large to begin with then, CCA should cause queue to decrease.
    # ramp_down = v.A[-1] - v.L[-1] - v.S[-1] < v.A[first] - v.L[first] - v.S[first]

    # Bottleneck queue should decrease
    # ramp_down = (
    #     (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
    #     < (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))

    desired = z3.And(
        z3.Or(high_util, ramp_up),
        z3.Or(low_delay, ramp_down))
    return desired, high_util, low_delay, ramp_up, ramp_down


def desired_high_util_low_loss(c, v, first, util_frac, loss_rate):
    high_util = v.S[-1] - v.S[first] >= util_frac * c.C * (c.T-1-first-c.D)

    loss_list: List[z3.BoolRef] = []
    for t in range(first, c.T):
        loss_list.append(v.L[t] > v.L[t-1])
    total_losses = z3.Sum(*loss_list)

    ramp_up = v.c_f[0][-1] > v.c_f[0][first]
    ramp_down = v.c_f[0][-1] < v.c_f[0][first] # Check if we want something on queue.
    # ramp_down = v.A[-1] - v.L[-1] - v.S[-1] < v.A[first] - v.L[first] - v.S[first]
    # Bottleneck queue should decrese
    # ramp_down = (
    #     (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
    #     < (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))
    low_loss = total_losses <= loss_rate * ((c.T-1) - first)
    desired = z3.And(
        z3.Or(high_util, ramp_up),
        z3.Or(low_loss, ramp_down))
    return (desired, high_util, low_loss, ramp_up, ramp_down,
            total_losses)


def desired_high_util_low_loss_low_delay(
        c, v, first, util_frac, loss_rate, delay_bound):
    cond_list = []
    for t in range(first, c.T):
        cond_list.append(v.A[t] - v.L[t] - v.S[t] <= delay_bound)
    # Queue seen by a new packet should not be more that delay_bound
    low_delay = z3.And(*cond_list)

    high_util = v.S[-1] - v.S[first] >= util_frac * c.C * (c.T-1-first-c.D)

    loss_list: List[z3.BoolRef] = []
    for t in range(first, c.T):
        loss_list.append(v.L[t] > v.L[t-1])
    total_losses = z3.Sum(*loss_list)

    ramp_up = v.c_f[0][-1] > v.c_f[0][first]

    ramp_down_cwnd = v.c_f[0][-1] < v.c_f[0][first]

    ramp_down_q = (v.A[-1] - v.L[-1] - v.S[-1] <
                   v.A[first] - v.L[first] - v.S[first])

    # Bottleneck queue should decrese
    ramp_down_bq = (
        (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
        < (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))

    low_loss = total_losses <= loss_rate * ((c.T-1) - first)
    desired = z3.And(
        z3.Or(high_util, ramp_up),
        z3.Or(low_loss, ramp_down_cwnd, ramp_down_bq),
        z3.Or(low_delay, ramp_down_cwnd, ramp_down_bq))
    return (desired, high_util, low_loss, low_delay, ramp_up, ramp_down_cwnd,
            ramp_down_q, ramp_down_bq, total_losses)


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

    logger.info("Improved gap from {} to {}".format(cur_min, best_gap))
    return sat, model


def run_verifier_incomplete(
    c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    # This is meant to create a partial function by
    # getting c, v, ctx from closure.
    _, _ = maximize_gap(c, v, ctx, verifier)
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
    return df
