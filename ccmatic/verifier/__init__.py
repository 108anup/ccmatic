import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import z3
from ccac.model import (ModelConfig, calculate_qdel, cca_aimd, cca_bbr,
                        cca_const, cca_copa, cca_paced, cwnd_rate_arrival,
                        epsilon_alpha, initial, loss_detected, loss_oracle,
                        make_solver, multi_flows, network, relate_tot)
from ccac.variables import VariableNames, Variables
from ccmatic.common import get_name_for_list
from cegis import NAME_TEMPLATE
from cegis.util import get_raw_value
from pyz3_utils.binary_search import BinarySearch
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver
from pyz3_utils.small_denom import find_small_denom_soln

logger = logging.getLogger('verifier')
GlobalConfig().default_logger_setup(logger)


def monotone_env(c, s, v):
    for t in range(1, c.T):
        for n in range(c.N):
            # Keeping in defs so that CCA does not exploit this.
            # In reality since arrival is max of prev and new value,
            # this constriant should never be broken anyway.
            # s.add(v.A_f[n][t] >= v.A_f[n][t - 1])

            # Can be in env or def as only contain verifier vars.
            # Keeping in env for now.
            s.add(v.Ld_f[n][t] >= v.Ld_f[n][t - 1])
            s.add(v.S_f[n][t] >= v.S_f[n][t - 1])
            s.add(v.L_f[n][t] >= v.L_f[n][t - 1])

            # Must be in environment. The loss may not be feasible
            # for new arrival.
            s.add(
                v.A_f[n][t] - v.L_f[n][t] >= v.A_f[n][t - 1] - v.L_f[n][t - 1])

        # Can be in env or def as only contain verifier vars.
        # Keeping in env for now.
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


def setup_ccac():
    c = ModelConfig.default()
    c.compose = True
    c.cca = "paced"
    c.simplify = False
    c.calculate_qdel = False
    c.C = 100
    c.T = 8

    s = MySolver()
    v = Variables(c, s)

    # s.add(z3.And(v.S[0] <= 1000, v.S[0] >= -1000))

    return c, s, v


def setup_ccac_definitions(c, v, use_loss_oracle=False):
    s = MySolver()
    s.warn_undeclared = False

    monotone_defs(c, s, v)
    initial(c, s, v)
    relate_tot(c, s, v)
    if(use_loss_oracle):
        loss_oracle(c, s, v)
        for t in range(c.T):
            s.add(v.A[t] - v.L[t] == v.C0 + c.C * t - v.W[t] + c.buf_max)
    else:
        loss_detected(c, s, v)
    epsilon_alpha(c, s, v)
    if c.calculate_qdel:
        calculate_qdel(c, s, v)
    if c.N > 1:
        assert (c.calculate_qdel)
        multi_flows(c, s, v)
    cwnd_rate_arrival(c, s, v)

    if c.cca == "const":
        cca_const(c, s, v)
    elif c.cca == "aimd":
        cca_aimd(c, s, v)
    elif c.cca == "bbr":
        cca_bbr(c, s, v)
    elif c.cca == "copa":
        cca_copa(c, s, v)
    elif c.cca == "any":
        pass
    elif c.cca == "paced":
        cca_paced(c, s, v)
    else:
        assert False, "CCA {} not found".format(c.cca)

    # Shouldn't be any loss at t0 otherwise cwnd is high and q is still 0.
    s.add(v.L_f[0][0] == 0)

    # Remove periodicity, as generator overfits and produces monotonic CCAs.
    # make_periodic(c, s, v, c.R + c.D)

    # Avoid weird cases where single packet is larger than BDP.
    s.add(v.alpha < 1/5)

    return s


def setup_ccac_environment(c, v):
    s = MySolver()
    s.warn_undeclared = False
    monotone_env(c, s, v)
    network(c, s, v)
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
    ramp_up = v.c_f[0][-1] > v.c_f[0][first]
    # If the queue is large to begin with then, CCA should cause queue to decrease.
    ramp_down = v.A[-1] - v.L[-1] - v.S[-1] < v.A[first] - v.L[first] - v.S[first]

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
    return desired, high_util, low_loss, ramp_up, ramp_down, total_losses/((c.T-1) - first)


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
        get_name_for_list(vn.A_f[0]): _get_model_value(v.A_f[0]),
        get_name_for_list(vn.c_f[0]): _get_model_value(v.c_f[0]),
        get_name_for_list(vn.S_f[0]): _get_model_value(v.S_f[0]),
        get_name_for_list(vn.W): _get_model_value(v.W),
        get_name_for_list(vn.L_f[0]): _get_model_value(v.L_f[0]),
        get_name_for_list(vn.Ld_f[0]): _get_model_value(v.Ld_f[0]),
        # get_name_for_list(vn.timeout_f[0]): _get_model_value(v.timeout_f[0]),
    }
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
    n_cex: int
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
        get_name_for_list(vn.A_f[0]): _get_model_value(v.A_f[0]),
        get_name_for_list(vn.c_f[0]): _get_model_value(v.c_f[0]),
        get_name_for_list(vn.S_f[0]): _get_model_value(v.S_f[0]),
        get_name_for_list(vn.W): _get_model_value(v.W),
        get_name_for_list(vn.L_f[0]): _get_model_value(v.L_f[0]),
        get_name_for_list(vn.Ld_f[0]): _get_model_value(v.Ld_f[0]),
        # get_name_for_list(vn.timeout_f[0]): _get_model_value(v.timeout_f[0]),
    }
    df = pd.DataFrame(cex_dict).astype(float)
    return df
