import z3
from ccac.model import (ModelConfig, calculate_qdel, cca_aimd, cca_bbr,
                        cca_const, cca_copa, cca_paced, cwnd_rate_arrival,
                        epsilon_alpha, initial, loss_detected, monotone,
                        multi_flows, network, relate_tot)
from ccac.variables import Variables
from pyz3_utils.my_solver import MySolver


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

    # Consider the no loss case for simplicity
    s.add(v.L[0] == v.L[-1])

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
