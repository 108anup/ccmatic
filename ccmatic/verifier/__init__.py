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
    c.calculate_qdel = True
    c.C = 100
    c.T = 7

    dummy_s = MySolver()
    v = Variables(c, dummy_s)

    return c, v


def setup_ccac_definitions(c, v):
    s = MySolver()

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

    # Remove periodicity, as generator overfits and produces monotonic CCAs.
    # make_periodic(c, s, v, c.R + c.D)

    # Avoid weird cases where single packet is larger than BDP.
    s.add(v.alpha < 1/5)

    return s


def setup_ccac_environment(c, v):
    s = MySolver()
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
