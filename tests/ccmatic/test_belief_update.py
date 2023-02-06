import z3
from typing import List

from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from ccmatic.common import try_except
from pyz3_utils.my_solver import MySolver


def setup():
    cc = CegisConfig()
    cc.name = "adv"
    cc.synth_ss = False
    cc.infinite_buffer = True
    cc.dynamic_buffer = False
    cc.app_limited = False
    cc.buffer_size_multiplier = 1
    cc.template_qdel = True
    cc.template_queue_bound = False
    cc.template_fi_reset = False
    cc.template_beliefs = True
    cc.N = 1
    cc.T = 6
    cc.history = cc.R
    cc.cca = "none"

    cc.use_belief_invariant = True

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

    cc.ideal_link = False
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


def test_beliefs_remain_consistent():
    cc, link = setup()
    c, _, v = link.c, link.s, link.v

    # Beliefs consistent
    _initial_minc_consistent = z3.And([v.min_c[n][0] <= c.C
                                       for n in range(c.N)])
    _initial_maxc_consistent = z3.And([v.max_c[n][0] >= c.C
                                       for n in range(c.N)])
    initial_beliefs_consistent = z3.And(
        _initial_minc_consistent, _initial_maxc_consistent)
    _final_minc_consistent = z3.And([v.min_c[n][-1] <= c.C
                                    for n in range(c.N)])
    _final_maxc_consistent = z3.And([v.max_c[n][-1] >= c.C
                                    for n in range(c.N)])
    final_beliefs_consistent = z3.And(
        _final_minc_consistent, _final_maxc_consistent)

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.definitions)
    verifier.add(link.environment)
    verifier.add(z3.Not(z3.Implies(initial_beliefs_consistent,
                                   final_beliefs_consistent)))

    sat = verifier.check()
    print(sat)
    if(str(sat) == "sat"):
        model = verifier.model()
        print(link.get_counter_example_str(model, link.verifier_vars))
        print(model.eval(initial_beliefs_consistent))
        print(model.eval(final_beliefs_consistent))
        import ipdb; ipdb.set_trace()


if (__name__ == "__main__"):
    # test_belief_does_not_degrade()
    test_beliefs_remain_consistent()