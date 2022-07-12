import pandas as pd

import logging
from typing import Optional, Tuple

import numpy as np
import z3
from ccac.model import (ModelConfig, calculate_qdel, cca_aimd, cca_bbr,
                        cca_const, cca_copa, cca_paced, cwnd_rate_arrival,
                        epsilon_alpha, initial, loss_detected, make_solver, monotone,
                        multi_flows, network, relate_tot)
from ccac.variables import VariableNames, Variables
from ccmatic.common import get_name_for_list
from pyz3_utils.binary_search import BinarySearch
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver
from pyz3_utils.small_denom import find_small_denom_soln

logger = logging.getLogger('verifier')
GlobalConfig().default_logger_setup(logger)


def setup_ccac():
    c = ModelConfig.default()
    c.compose = True
    c.cca = "paced"
    c.simplify = False
    c.calculate_qdel = False
    c.C = 100
    c.T = 7

    s = MySolver()
    v = Variables(c, s)

    # s.add(z3.And(v.S[0] <= 1000, v.S[0] >= -1000))

    return c, s, v


def setup_ccac_definitions(c, v):
    s = MySolver()
    s.warn_undeclared = False

    monotone(c, s, v)
    initial(c, s, v)
    relate_tot(c, s, v)
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
            this_gap = orig_model.eval(
                v.C0 + c.C * t - v.W[t] - v.S[t]).as_fraction()
            cur_min = min(cur_min, this_gap)

    # if(cur_min == np.inf):
    #     return orig_sat

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
    # Get c, v, ctx from closure
    _, _ = maximize_gap(c, v, ctx, verifier)
    sat, _, model = find_small_denom_soln(verifier, 4096)
    return sat, model


def get_cex_df(
    counter_example: z3.ModelRef, v: Variables, vn: VariableNames
) -> pd.DataFrame:
    def _get_model_value(l):
        ret = []
        for vvar in l:
            ret.append(counter_example.eval(vvar).as_fraction())
        return ret
    cex_dict = {
        get_name_for_list(vn.A_f[0]): _get_model_value(v.A_f[0]),
        get_name_for_list(vn.c_f[0]): _get_model_value(v.c_f[0]),
        get_name_for_list(vn.S_f[0]): _get_model_value(v.S_f[0]),
        get_name_for_list(vn.W): _get_model_value(v.W),
        get_name_for_list(vn.L): _get_model_value(v.L),
    }
    df = pd.DataFrame(cex_dict).astype(float)
    return df
