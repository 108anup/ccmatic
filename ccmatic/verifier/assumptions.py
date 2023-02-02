from typing import List, Tuple
import numpy as np
import z3
from ccac.cca_aimd import cca_aimd
from ccac.cca_bbr import cca_bbr
from ccac.cca_copa import cca_copa
from ccac.config import ModelConfig
from ccac.model import cca_const, cca_paced
from ccac.utils import make_periodic
from ccac.variables import Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten
from ccmatic.verifier import setup_ccac_for_cegis
from ccmatic.verifier.ideal import IdealLink
from cegis.util import z3_max
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
                # decr_alloweds.append(
                #     z3.And(
                #         v.qdel[t-c.R-c.D][dt],
                #         v.S[t-c.R] > v.S[t-c.R-1],
                #         v.c_f[n][t-1] * dt >= v.alpha * (c.R + dt)))
                decr_alloweds.append(
                    z3.And(
                        v.qdel[t-c.R-c.D][dt],
                        v.S[t-c.R] > v.S[t-c.R-1],
                        # If the queue is too small, then clearly the min RTT
                        # is also small
                        v.S[t-c.R] < v.A[t-c.R] - v.L[t-c.R] - v.alpha,
                        v.c_f[n][t-1] * (dt+1) >= v.alpha * (c.R + dt)))
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


def cca_bbr_deterministic(c: ModelConfig, s: MySolver, v: Variables, pre=""):
    # The period over which we compute rates
    P = c.R
    # Number of RTTs over which we compute the max_cwnd (=10 in the spec)
    max_R = 4
    # The number of RTTs in the BBR cycle (=8 in the spec)
    cycle = 4
    # The state the flow starts in at t=0
    start_state_f = [z3.Int(f"{pre}bbr_start_state_{n}") for n in range(c.N)]

    for n in range(c.N):
        s.add(start_state_f[n] >= 0)
        s.add(start_state_f[n] < cycle)
        for t in range(c.R + P, c.T):
            # Compute the max RTT over the last max_R RTTs
            max_rate = \
                [0 for dt in range(min(t-c.R-P+1, max_R))]
            max_rate[0] = (v.S_f[n][t-c.R] - v.S_f[n][t-c.R-P]) / P
            for dt in range(1, len(max_rate)):
                rate = (v.S_f[n][t-dt-c.R] - v.S_f[n][t-dt-c.R-P]) / P
                max_rate[dt] = z3.If(rate > max_rate[dt-1],
                                     rate, max_rate[dt-1])

            rate_to_use = z3_max(max_rate[-1], 2 * v.alpha/P)
            s.add(v.c_f[n][t] == 2 * rate_to_use * P)
            s_0 = (start_state_f[n] == (0 - t / c.R) % cycle)
            s_1 = (start_state_f[n] == (1 - t / c.R) % cycle)
            s.add(z3.Implies(s_0,
                             v.r_f[n][t] == 1.25 * rate_to_use))
            s.add(z3.Implies(s_1,
                             v.r_f[n][t] == 0.8 * rate_to_use))
            s.add(z3.Implies(z3.And(z3.Not(s_0), z3.Not(s_1)),
                             v.r_f[n][t] == 1 * rate_to_use))

        # # Fix r so that generator can't change r to break cca_defs
        # for t in range(c.R + P):
        #     s.add(v.r_f[n][t] == v.c_f[n][t] / c.R)


def get_periodic_constraints_ccac(
        cc: CegisConfig, c: ModelConfig, v: Variables):
    s = MySolver()
    s.warn_undeclared = False

    make_periodic(c, s, v, cc.history)
    return z3.And(*s.assertion_list)


def get_cca_definition(c: ModelConfig, v: Variables, pre=""):
    s = MySolver()
    s.warn_undeclared = False

    if c.cca == "const":
        cca_const(c, s, v)
    elif c.cca == "aimd":
        cca_aimd(c, s, v)
    elif c.cca == "bbr":
        cca_bbr_deterministic(c, s, v, pre)
    elif c.cca == "copa":
        cca_copa_deterministic(c, s, v)
    elif c.cca == "any":
        pass
    elif c.cca == "paced":
        cca_paced(c, s, v)
    elif c.cca == "none":
        pass
    else:
        assert False, "CCA {} not found".format(c.cca)

    return z3.And(*s.assertion_list)


def get_cca_vvars(c: ModelConfig, v: Variables, pre=""):
    if(c.cca == "bbr"):
        return np.array([
            z3.Int(f"{pre}bbr_start_state_{n}") for n in range(c.N)])
    return []


class AssumptionVerifier(IdealLink):
    """
    For testing for all initial conditions and arrival curves, the ideal
    response is feasible implies that the synthesized assumption must be
    feasible.

    This verifier will put constraints on what assumptions to pick assumptions
    that are meaningful (i.e., includes behaviors that an ideal link could
    produce.)

    The constraints can simply be picked using IdealLink, we just need to change
    the verifier and definition vars.

    The synthesized assumption cannot in anyway change the trace. All feasible
    traces are always feasible irrespective of the synthesized assumption (this
    is different from the CCA synth case, where the traces may not be feasible
    for a CCA). So all variables are basically verifier vars. (no variables
    except the assumption depends on generator vars.)

    Since all vars are verifier vars, all definitions become environment.
    """

    @staticmethod
    def setup_cegis_basic(cc: CegisConfig):
        (c, s, v, ccac_domain, ccac_definitions, environment,
         verifier_vars, definition_vars) = IdealLink.setup_cegis_basic(cc)

        # Since the assumption depends on waste. Need to specify
        # deterministic choice of waste that the assumption must allow.
        waste_defs_list = []
        for t in range(c.T):
            waste_defs_list.append(v.W[t] == (v.C0 + c.C*t) - v.S[t])
        waste_defs = z3.And(*waste_defs_list)
        environment = z3.And(environment, waste_defs)

        # TODO: see if the process actually needs to say that there exists some
        # way to waste, instead of allowing a prespecified waste.

        new_environment = z3.And(ccac_domain, ccac_definitions, environment)
        assert isinstance(new_environment, z3.ExprRef)

        new_verifier_vars = verifier_vars + definition_vars + flatten([v.W, v.C0])

        return (c, s, v,
                True, True, z3.And(ccac_domain, ccac_definitions, environment),
                new_verifier_vars, [])
