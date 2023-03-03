from typing import List
import z3
from ccmatic import CCmatic

from ccmatic.cegis import CegisConfig, VerifierType
from ccmatic.common import try_except
from ccmatic.verifier.assumptions import get_cca_definition
from cegis import Cegis
from pyz3_utils.my_solver import MySolver


def setup(buffer="infinite", buf_size=1, T=6, cca="none"):
    cc = CegisConfig()
    cc.verifier_type = VerifierType.cbrdelay
    cc.name = "cbrdelay"
    cc = CegisConfig()
    cc.name = "adv"
    cc.synth_ss = False

    cc.buffer_size_multiplier = buf_size
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

    cc.template_qdel = True
    cc.template_queue_bound = False
    cc.template_fi_reset = False
    # cc.template_beliefs = True
    cc.N = 1
    cc.T = T
    cc.history = cc.R
    cc.cca = cca

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

    cc.feasible_response = False
    # cc.send_min_alpha = True

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
    print(f"{cc.name}: {cc.desire_tag()}")

    return cc, link


def test_cbrdelay_basic():
    cc, link = setup(cca="bbr")
    c, _, v = link.c, link.s, link.v

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.definitions)
    verifier.add(link.environment)
    verifier.add(z3.Not(link.desired))
    cca_definitions = get_cca_definition(c, v)
    verifier.add(cca_definitions)

    sat = verifier.check()
    print(sat)
    if(str(sat) == "sat"):
        model = verifier.model()
        print(link.get_counter_example_str(model, link.verifier_vars))
        import ipdb; ipdb.set_trace()


if(__name__ == "__main__"):
    test_cbrdelay_basic()