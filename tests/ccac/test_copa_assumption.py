import sys
import pandas as pd
import logging
import z3
from ccac.variables import VariableNames
from ccmatic.cegis import CegisConfig
from cegis.util import Metric, get_raw_value, optimize_multi_var, unroll_assertions
from ccmatic.verifier import get_cex_df, setup_cegis_basic
from ccmatic.verifier.assumptions import (get_cca_definition, get_cca_vvars,
                                          get_periodic_constraints_ccac)
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver


def test_copa_composition():
    cc = CegisConfig()
    cc.T = 10
    cc.history = cc.R + cc.D
    cc.infinite_buffer = True  # No loss for simplicity
    cc.dynamic_buffer = False
    cc.buffer_size_multiplier = 1
    cc.template_queue_bound = False
    cc.template_mode_switching = False
    cc.template_qdel = True

    cc.compose = True
    # cc.cca = "copa"
    cc.cca = "copa"
    if(cc.cca == "copa"):
        cc.history = cc.R + cc.D
    elif(cc.cca == "bbr"):
        cc.history = 2 * cc.R

    cc.feasible_response = True

    cc.desired_util_f = z3.Real('desired_util_f')
    (c, s, v,
     ccac_domain, ccac_definitions, environment,
     verifier_vars, definition_vars) = setup_cegis_basic(cc)
    vn = VariableNames(v)
    periodic_constriants = get_periodic_constraints_ccac(cc, c, v)
    cca_definitions = get_cca_definition(c, v)

    # 10% utilization. Can be made arbitrarily small
    desired = v.S[-1] - v.S[0] >= cc.desired_util_f * c.C * c.T

    def get_counter_example_str(counter_example: z3.ModelRef) -> str:
        df = get_cex_df(counter_example, v, vn, c)
        # for n in range(c.N):
        #     df[f"incr_{n},t"] = [
        #         get_raw_value(counter_example.eval(z3.Bool(f"incr_{n},{t}"))) for t in range(c.T)]
        #     df[f"decr_{n},t"] = [
        #         get_raw_value(counter_example.eval(z3.Bool(f"decr_{n},{t}"))) for t in range(c.T)]
        ret = "\n{}".format(df)
        ret += "\nv.qdel[t][dt]\n"
        ret += "  " + " ".join([str(i) for i in range(c.T)]) + "\n"
        for t in range(c.T):
            ret += f"{t} " + " ".join([
                str(int(bool(counter_example.eval(v.qdel[t][dt]))))
                for dt in range(c.T)]) + "\n"

        return ret

    # CCAC paper
    known_assumption_list = []
    for t in range(1, c.T):
        known_assumption_list.append(
            z3.Implies(v.W[t] > v.W[t-1],
                       v.A[t]-v.L[t]-v.S[t] <= 0)
        )
    known_assumption_ccac = z3.And(known_assumption_list)

    # CCmatic 1
    known_assumption_list = []
    for t in range(1, c.T):
        known_assumption_list.append(
            z3.Implies(v.W[t] > v.W[t-1],
                       v.C0 + c.C * t - v.W[t] - v.S[t] <= 0)
        )
    known_assumption_ccmatic1 = z3.And(known_assumption_list)

    # CCmatic 2
    known_assumption_list = []
    for t in range(1, c.T):
        known_assumption_list.append(
            v.S[t] > v.C0 + c.C * (t-c.D) - v.W[t-c.D])
    known_assumption_ccmatic2 = z3.And(known_assumption_list)

    # # When to waste, Incal
    # known_assumption_list = []
    # for t in range(1, c.T):
    #     known_assumption_list.append(
    #         v.S[t] > v.C0 + c.C * (t-1) - v.W[t-1]
    #     )
    # known_assumption = z3.And(known_assumption_list)

    # delay_f = z3.Real('delay_f')
    # burst_f = 1

    # # Netcal
    # def beta(t):
    #     val = (t - delay_f * c.D)
    #     absval = z3.If(val >= 0, val, 0)
    #     assert(isinstance(absval, z3.ArithRef))
    #     return c.C * absval

    # def alpha(t):
    #     burst = burst_f * c.C * c.D
    #     return burst + c.C * (t)

    # known_assumption_list = []
    # for t in range(1, c.T):
    #     known_assumption_list.append(
    #         z3.And(*[v.S[t] <= v.S[s] + alpha(t-s)
    #                  for s in range(t+1)]))
    #     known_assumption_list.append(
    #         z3.Or(*[v.S[t] >= v.A[s]-v.L[s] + beta(t-s)
    #                 for s in range(t+1)])
    #     )
    # known_assumption = z3.And(known_assumption_list)

    # Monotonic assumption 21 (weakest)
    known_assumption_list = []
    for t in range(c.T):
        # known_assumption_list.append(
        #     z3.Or(
        #         v.A[t] - v.A[t-1] + c.C <= 0,
        #         v.S[t] - v.S[t-1] + v.A[t] - v.A[t-1] >= 0
        #     ))
        known_assumption_list.append(
            z3.Or(
                +v.A[t-0] - v.A[t-1] +
                (v.C0 + c.C*(t-0)) - (v.C0 + c.C*(t-1)) <= 0,
                - v.S[t-0] + v.S[t-1] - v.A[t-0] + v.A[t-1] <= 0
            )
        )
    known_assumption_ccmatic_monotonic21 = z3.And(known_assumption_list)

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(ccac_domain)
    verifier.add(ccac_definitions)
    verifier.add(environment)
    verifier.add(cca_definitions)
    verifier.add(periodic_constriants)
    # verifier.add(z3.Not(known_assumption))
    # verifier.add(known_assumption_ccac)
    # verifier.add(z3.Not(known_assumption_ccmatic2))
    verifier.add(known_assumption_ccmatic_monotonic21)
    verifier.add(z3.Not(desired))
    # verifier.add(desired50)
    # verifier.add(desired)
    # verifier.add(v.A[0]-v.L[0]-v.S[0] == 0)
    # verifier.add(v.c_f[0][4] == 100)
    # verifier.add(v.L[0] == 0)

    optimization_list = [
        # Metric(cc.desired_util_f, 0.1, 1, 0.001, True),
        # Metric(delay_f, 0, 1, 0.001, True)
    ]
    verifier.add(cc.desired_util_f == 0.1)

    verifier.push()
    for metric in optimization_list:
        if(metric.maximize):
            verifier.add(metric.z3ExprRef == metric.lo)
        else:
            verifier.add(metric.z3ExprRef == metric.hi)

    sat = verifier.check()
    print(str(sat))
    if(str(sat) == "sat"):
        model = verifier.model()
        print(get_counter_example_str(model))
        import ipdb; ipdb.set_trace()

    else:
        sys.exit(1)
        # # Unsat core
        # dummy = MySolver()
        # dummy.warn_undeclared = False
        # dummy.set(unsat_core=True)

        # assertion_list = verifier.assertion_list
        # for assertion in assertion_list:
        #     for expr in unroll_assertions(assertion):
        #         dummy.add(expr)
        # assert(str(dummy.check()) == "unsat")
        # unsat_core = dummy.unsat_core()
        # print(len(unsat_core))

        verifier.pop()

        GlobalConfig().logging_levels['cegis'] = logging.DEBUG
        logger = logging.getLogger('cegis')
        GlobalConfig().default_logger_setup(logger)

        ret = optimize_multi_var(verifier, optimization_list)
        df = pd.DataFrame(ret)
        sort_columns = [x.name() for x in optimization_list]
        sort_order = [x.maximize for x in optimization_list]
        df = df.sort_values(by=sort_columns, ascending=sort_order)
        print(df)


if (__name__ == "__main__"):
    test_copa_composition()
