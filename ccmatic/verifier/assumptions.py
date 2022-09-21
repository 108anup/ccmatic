import z3
from ccac.cca_aimd import cca_aimd
from ccac.cca_bbr import cca_bbr
from ccac.cca_copa import cca_copa
from ccac.config import ModelConfig
from ccac.model import cca_const, cca_paced
from ccac.utils import make_periodic
from ccac.variables import Variables
from ccmatic.cegis import CegisConfig
from ccmatic.verifier import setup_ccac_for_cegis
from pyz3_utils.my_solver import MySolver


def cca_copa_deterministic(c: ModelConfig, s: MySolver, v: Variables):
    for n in range(c.N):
        for t in range(c.T):
            # Basic constraints
            s.add(v.c_f[n][t] > 0)
            s.add(v.r_f[n][t] == v.c_f[n][t] / c.R)

            if t - c.R - c.D < 0:
                continue

            incr_alloweds, decr_alloweds = [], []
            for dt in range(t+1):
                # Whether we are allowd to increase/decrease
                # Warning: Adversary here is too powerful if D > 1. Add
                # a constraint for every point between t-1 and t-1-D
                assert(c.D == 1)
                incr_alloweds.append(
                    z3.And(
                        v.qdel[t-c.R][dt],
                        v.S[t-c.R] > v.S[t-c.R-1],
                        v.c_f[n][t-1] * max(0, dt-1)
                        <= v.alpha*(c.R+max(0, dt-1))))
                decr_alloweds.append(
                    z3.And(
                        v.qdel[t-c.R-c.D][dt],
                        v.S[t-c.R] > v.S[t-c.R-1],
                        v.c_f[n][t-1] * dt >= v.alpha * (c.R + dt)))
            # If inp is high at the beginning, qdel can be arbitrarily
            # large
            decr_alloweds.append(v.S[t-c.R] < v.A[0]-v.L[0])

            incr_allowed = z3.Or(*incr_alloweds)
            decr_allowed = z3.Or(*decr_alloweds)

            # When both incr_allowed and decr_allowed, what to do:
            # # Prefer increase
            # s.add(z3.Implies(incr_allowed,
            #       v.c_f[n][t] == v.c_f[n][t-1]+v.alpha/c.R))
            # sub = v.c_f[n][t-1] - v.alpha / c.R
            # s.add(z3.Implies(z3.Not(incr_allowed), v.c_f[n][t]
            #                  == z3.If(sub < v.alpha, v.alpha, sub)))

            # Prefer decrease
            sub = v.c_f[n][t-1] - v.alpha / c.R
            s.add(z3.Implies(decr_allowed, v.c_f[n][t]
                             == z3.If(sub < v.alpha, v.alpha, sub)))
            s.add(z3.Implies(z3.Not(decr_allowed),
                  v.c_f[n][t] == v.c_f[n][t-1]+v.alpha/c.R))


def get_periodic_constraints_ccac(cc: CegisConfig, c: ModelConfig, v: Variables):
    s = MySolver()
    s.warn_undeclared = False

    make_periodic(c, s, v, cc.history)
    return z3.And(*s.assertion_list)


def get_cca_definition(cc: CegisConfig, c: ModelConfig, v: Variables):
    s = MySolver()
    s.warn_undeclared = False

    if c.cca == "const":
        cca_const(c, s, v)
    elif c.cca == "aimd":
        cca_aimd(c, s, v)
    elif c.cca == "bbr":
        cca_bbr(c, s, v)
    elif c.cca == "copa":
        cca_copa_deterministic(c, s, v)
    elif c.cca == "any":
        pass
    elif c.cca == "paced":
        cca_paced(c, s, v)
    else:
        assert False, "CCA {} not found".format(c.cca)

    return z3.And(*s.assertion_list)