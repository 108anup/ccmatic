import z3
from ccac.variables import VariableNames
from ccmatic.cegis import CegisConfig
from cegis.util import get_raw_value, unroll_assertions
from ccmatic.verifier import get_cex_df, setup_cegis_basic
from ccmatic.verifier.assumptions import (get_cca_definition,
                                          get_periodic_constraints_ccac)
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
    cc.cca = "copa"

    cc.feasible_response = True

    (c, s, v,
     ccac_domain, ccac_definitions, environment,
     verifier_vars, definition_vars) = setup_cegis_basic(cc)
    vn = VariableNames(v)
    periodic_constriants = get_periodic_constraints_ccac(cc, c, v)
    cca_definitions = get_cca_definition(cc, c, v)

    # 10% utilization. Can be made arbitrarily small
    desired10 = v.S[-1] - v.S[0] >= 0.1 * c.C * c.T
    desired50 = v.S[-1] - v.S[0] >= 0.8 * c.C * c.T

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

    known_assumption_list = []
    for t in range(1, c.T):
        known_assumption_list.append(
            z3.Implies(v.W[t] > v.W[t-1],
                       v.A[t]-v.L[t]-v.S[t] <= v.alpha)
        )
    known_assumption = z3.And(known_assumption_list)

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(ccac_domain)
    verifier.add(ccac_definitions)
    verifier.add(environment)
    verifier.add(cca_definitions)
    verifier.add(periodic_constriants)
    verifier.add(z3.Not(known_assumption))
    verifier.add(desired50)
    # verifier.add(desired)
    # verifier.add(v.A[0]-v.L[0]-v.S[0] == 0)
    # verifier.add(v.c_f[0][4] == 100)
    # verifier.add(v.L[0] == 0)

    sat = verifier.check()
    print(str(sat))
    if(str(sat) == "sat"):
        model = verifier.model()
        print(get_counter_example_str(model))

    # else:
    #     # Unsat core
    #     dummy = MySolver()
    #     dummy.warn_undeclared = False
    #     dummy.set(unsat_core=True)

    #     assertion_list = verifier.assertion_list
    #     for assertion in assertion_list:
    #         for expr in unroll_assertions(assertion):
    #             dummy.add(expr)
    #     assert(str(dummy.check()) == "unsat")
    #     unsat_core = dummy.unsat_core()
    #     print(len(unsat_core))
    #     import ipdb; ipdb.set_trace()


if (__name__ == "__main__"):
    test_copa_composition()
