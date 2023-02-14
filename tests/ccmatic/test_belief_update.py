import z3
from typing import List, Literal
from ccac.variables import VariableNames

from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from ccmatic.common import try_except
from ccmatic.verifier import get_cex_df, plot_cex
from cegis import get_unsat_core
from pyz3_utils.my_solver import MySolver


def setup(
        ideal=False,
        buffer: Literal["finite", "infinite", "dynamic"] = "infinite",
        T=6, buf_size=None):
    cc = CegisConfig()
    cc.name = "adv"
    cc.synth_ss = False

    cc.buffer_size_multiplier = 1
    if (buffer == "finite"):
        cc.infinite_buffer = False
        cc.dynamic_buffer = False
    elif (buffer == "infinite"):
        cc.infinite_buffer = True
        cc.dynamic_buffer = False
    elif (buffer == "dynamic"):
        cc.infinite_buffer = False
        cc.dynamic_buffer = True
    else:
        assert False

    if(buf_size is not None):
        assert buffer == "finite"
        cc.buffer_size_multiplier = buf_size

    cc.app_limited = False
    cc.template_qdel = True
    cc.template_queue_bound = False
    cc.template_fi_reset = False
    cc.template_beliefs = True
    cc.N = 1
    cc.T = T
    cc.history = cc.R
    cc.cca = "none"

    cc.use_belief_invariant = True
    # cc.app_limited = True
    # cc.app_fixed_avg_rate = True

    cc.desired_util_f = 0.5
    cc.desired_queue_bound_multiplier = 4
    cc.desired_queue_bound_alpha = 3
    if(cc.infinite_buffer):
        cc.desired_loss_count_bound = 0
        cc.desired_large_loss_count_bound = 0
        cc.desired_loss_amount_bound_multiplier = 0
        cc.desired_loss_amount_bound_alpha = 0
    else:
        cc.desired_loss_count_bound = 3
        cc.desired_large_loss_count_bound = 3
        cc.desired_loss_amount_bound_multiplier = 3
        cc.desired_loss_amount_bound_alpha = 3

    cc.ideal_link = ideal
    cc.feasible_response = False

    link = CCmatic(cc)
    try_except(link.setup_config_vars)

    search_constraints = True
    template_definitions = []
    generator_vars = []

    def get_solution_str(
        solution: z3.ModelRef, generator_vars: List[z3.ExprRef], n_cex: int):
        return ""

    link.setup_cegis_loop(
        search_constraints,
        template_definitions, generator_vars, get_solution_str)
    # link.critical_generator_vars = critical_generator_vars
    print("Adver: " + cc.desire_tag())

    return cc, link


def test_belief_does_not_degrade():
    cc, link = setup()
    c, _, v = link.c, link.s, link.v

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.definitions)
    verifier.add(link.environment)
    beliefs_degrade_list = []
    for n in range(c.N):
        beliefs_degrade_list.append(
            v.max_c[n][-1] > v.max_c[n][0])
        beliefs_degrade_list.append(
            v.min_c[n][-1] < v.min_c[n][0])
        beliefs_degrade_list.append(
            v.max_buffer[n][-1] > v.max_buffer[n][0])
        beliefs_degrade_list.append(
            v.min_buffer[n][-1] < v.min_buffer[n][0])
    verifier.add(z3.Or(*beliefs_degrade_list))

    sat = verifier.check()
    print(sat)
    if(str(sat) == "sat"):
        model = verifier.model()
        print(link.get_counter_example_str(model, link.verifier_vars))
        import ipdb; ipdb.set_trace()


def test_beliefs_remain_consistent(
        ideal=True,
        buffer: Literal["finite", "infinite", "dynamic"] = "infinite"):
    cc, link = setup(ideal, buffer)
    c, _, v = link.c, link.s, link.v

    # Beliefs consistent
    _initial_minc_consistent = z3.And([v.min_c[n][0] <= c.C
                                       for n in range(c.N)])
    _initial_maxc_consistent = z3.And([v.max_c[n][0] >= c.C
                                       for n in range(c.N)])
    initial_c_beliefs_consistent = z3.And(
        _initial_minc_consistent, _initial_maxc_consistent)
    _final_minc_consistent = z3.And([v.min_c[n][-1] <= c.C
                                    for n in range(c.N)])
    _final_maxc_consistent = z3.And([v.max_c[n][-1] >= c.C
                                    for n in range(c.N)])
    final_c_beliefs_consistent = z3.And(
        _final_minc_consistent, _final_maxc_consistent)

    if (c.app_limited and c.app_fixed_avg_rate):
        _initial_min_app_rate_consistent = z3.And(
            [v.min_app_rate[n][0] <= v.app_rate for n in range(c.N)])
        _initial_max_app_rate_consistent = z3.And(
            [v.max_app_rate[n][0] >= v.app_rate for n in range(c.N)])
        initial_app_rate_beliefs_consistent = z3.And(
            _initial_min_app_rate_consistent, _initial_max_app_rate_consistent)
        _final_min_app_rate_consistent = z3.And(
            [v.min_app_rate[n][-1] <= v.app_rate for n in range(c.N)])
        _final_max_app_rate_consistent = z3.And(
            [v.max_app_rate[n][-1] >= v.app_rate for n in range(c.N)])
        final_app_rate_beliefs_consistent = z3.And(
            _final_min_app_rate_consistent, _final_max_app_rate_consistent)

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.definitions)
    verifier.add(link.environment)
    verifier.add(z3.Not(z3.Implies(initial_c_beliefs_consistent,
                                   final_c_beliefs_consistent)))
    if (c.app_limited and c.app_fixed_avg_rate):
        verifier.add(z3.Not(z3.Implies(
            initial_app_rate_beliefs_consistent,
            final_app_rate_beliefs_consistent)))
    # r = z3.Real('r')

    for n in range(c.N):
        for t in range(c.T):
            pass
            # if(t > 0):
            #     verifier.add(v.A_f[n][t] > v.A_f[n][t-1])
            # verifier.add(z3.Or(v.r_f[n][t] == r, v.r_f[n][t] == r/2))
            # verifier.add(v.r_f[n][t] > 0)
            # verifier.add(v.c_f[n][t] == 10 * c.C * (c.R + c.D))

            # verifier.add(v.c_f[n][t] == c.C * (c.R + c.D) / 3)
            # verifier.add(v.r_f[n][t] == v.c_f[n][t]/c.R)

            # # cwnd limited
            # verifier.add(v.c_f[n][t] == c.C * (c.R + c.D) / 3)
            # verifier.add(v.r_f[n][t] == 1000 * c.C)

    sat = verifier.check()
    print(sat)
    if (str(sat) == "sat"):
        model = verifier.model()
        vn = VariableNames(v)
        df = get_cex_df(model, v, vn, c)
        print(link.get_counter_example_str(model, link.verifier_vars))
        print(model.eval(initial_c_beliefs_consistent))
        print(model.eval(final_c_beliefs_consistent))
        if (c.app_limited and c.app_fixed_avg_rate):
            print(model.eval(initial_app_rate_beliefs_consistent))
            print(model.eval(final_app_rate_beliefs_consistent))
        plot_cex(model, df, c, v, 'tmp/cex_df.pdf')

        # verifier2 = MySolver()
        # verifier2.warn_undeclared = False
        # verifier2.add(link.definitions)
        # verifier2.add(link.environment)
        # # # r = z3.Real('r')

        # for n in range(c.N):
        #     for t in range(c.T):
        #         verifier2.add(v.r_f[n][t] == model.eval(v.r_f[n][t]))
        #         if(t >= 1):
        #             verifier2.add(v.A_f[n][t] > v.A_f[n][t-1])
        # #         # verifier2.add(z3.Or(v.r_f[n][t] == r/2, v.r_f[n][t] == r))
        # #         verifier2.add(v.r_f[n][t] > 0)
        # #         verifier2.add(v.c_f[n][t] == 10 * c.C * (c.R + c.D))
        # verifier2.add(v.A[0] == model.eval(v.A[0]))
        # verifier2.add(v.L[0] == model.eval(v.L[0]))

        # for t in range(c.T):
        #     # verifier2.add(v.S[t] == model.eval(v.S[t]))
        #     verifier2.add(v.S[t] >= model.eval(v.S[t]))
        #     verifier2.add(v.S[t] <= 1 + model.eval(v.S[t]))

        # sat = verifier2.check()
        # print(sat)
        # if(str(sat) == 'sat'):
        #     model = verifier2.model()
        #     vn = VariableNames(v)
        #     df = get_cex_df(model, v, vn, c)
        #     print(link.get_counter_example_str(model, link.verifier_vars))
        #     print(model.eval(initial_c_beliefs_consistent))
        #     print(model.eval(final_c_beliefs_consistent))
        #     plot_cex(model, df, c, v, 'tmp/cex_df2.pdf')
        # else:
        #     uc = get_unsat_core(verifier2)

        import ipdb; ipdb.set_trace()


def test_can_learn_beliefs(f: float):
    cc, link = setup(ideal=False, buffer="finite", T=9, buf_size=1.5)
    c, _, v = link.c, link.s, link.v

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.definitions)
    verifier.add(link.environment)
    verifier.add(v.max_c[0][c.T-1] > 2 * v.min_c[0][c.T-1])
    verifier.add(v.min_c[0][0] == v.min_c[0][c.T-1])
    verifier.add(v.max_c[0][0] == v.max_c[0][c.T-1])

    for n in range(c.N):
        for t in range(c.T):
            verifier.add(v.c_f[n][t] == f * c.C * (c.R + c.D))
            verifier.add(v.r_f[n][t] == v.c_f[n][t] / c.R)

            # verifier.add(v.r_f[n][t] == f * c.C)
            # if(t >= 1):
            #     verifier.add(v.c_f[n][t] == v.A_f[n][t-1] - v.S_f[n][t-1] + v.r_f[n][t] * 1000)

    sat = verifier.check()
    print(sat)
    if(str(sat) == "sat"):
        model = verifier.model()
        print(link.get_counter_example_str(model, link.verifier_vars))
        import ipdb; ipdb.set_trace()


if (__name__ == "__main__"):
    # test_belief_does_not_degrade()
    test_beliefs_remain_consistent(ideal=True, buffer="infinite")
    test_beliefs_remain_consistent(ideal=True, buffer="dynamic")
    test_beliefs_remain_consistent(ideal=False, buffer="infinite")
    test_beliefs_remain_consistent(ideal=False, buffer="dynamic")
    # test_can_learn_beliefs(2)