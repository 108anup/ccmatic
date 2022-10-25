import copy
import logging
from fractions import Fraction
from typing import List

import z3
from ccac.config import ModelConfig
from ccac.variables import Variables
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, get_product_ite

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
domain_clauses = []

coeffs = {
    'c_f[n]_loss': z3.Real('Gen__coeff_c_f[n]_loss'),
    'c_f[n]_noloss': z3.Real('Gen__coeff_c_f[n]_noloss'),
    'ack_f[n]_loss': z3.Real('Gen__coeff_ack_f[n]_loss'),
    'ack_f[n]_noloss': z3.Real('Gen__coeff_ack_f[n]_noloss'),
}

consts = {
    'c_f[n]_loss': z3.Real('Gen__const_c_f[n]_loss'),
    'c_f[n]_noloss': z3.Real('Gen__const_c_f[n]_noloss'),
}

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))
critical_generator_vars = flatten(list(coeffs.values()))
# critical_generator_vars = generator_vars

# Search constr
search_range_coeffs = [Fraction(i, 2) for i in range(5)]
search_range_consts = [-1, 0, 1]
domain_clauses = []
for coeff in flatten(list(coeffs.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeffs]))
for const in flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[const == val for val in search_range_consts]))

# # All expressions should be different. Otherwise that expression is not needed.
# conds = ['delay', 'nodelay']
# for pair in itertools.combinations(conds, 2):
#     is_same = z3.And(
#         coeffs['c_f[n]_{}'.format(pair[0])] ==
#         coeffs['c_f[n]_{}'.format(pair[1])],
#         coeffs['ack_f[n]_{}'.format(pair[0])] ==
#         coeffs['ack_f[n]_{}'.format(pair[1])])
#     domain_clauses.append(z3.Not(is_same))

search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))


def get_template_definitions(
        cc: CegisConfig, c: ModelConfig, v: Variables):
    template_definitions = []
    first = cc_ideal.history
    for n in range(c.N):
        for t in range(first, c.T):
            # paced is already there to ensure r_f is set and c_f > 0

            acked_bytes = v.S_f[n][t-c.R] - v.S_f[n][t-cc_ideal.history]
            loss_detected = v.Ld_f[n][t] > v.Ld_f[n][t-c.R]
            rhs_loss = (
                get_product_ite(
                    coeffs['c_f[n]_loss'], v.c_f[n][t-c.R],
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['ack_f[n]_loss'], acked_bytes,
                    search_range_coeffs)
                + get_product_ite(
                    consts['c_f[n]_loss'], v.alpha,
                    search_range_consts))
            rhs_noloss = (
                get_product_ite(
                    coeffs['c_f[n]_noloss'], v.c_f[n][t-c.R],
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['ack_f[n]_noloss'], acked_bytes,
                    search_range_coeffs)
                + get_product_ite(
                    consts['c_f[n]_noloss'], v.alpha,
                    search_range_consts))
            rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
            assert isinstance(rhs, z3.ArithRef)

            target_cwnd = rhs
            next_cwnd = z3.If(v.c_f[n][t-1] < target_cwnd,
                              v.c_f[n][t-1] + v.alpha,
                              v.c_f[n][t-1] - v.alpha)
            template_definitions.append(
                v.c_f[n][t] == z3.If(next_cwnd >= v.alpha, next_cwnd, v.alpha))

            # target_cwnd = z3.If(rhs >= v.alpha + cc.template_cca_lower_bound,
            #                     rhs, v.alpha + cc.template_cca_lower_bound)
            # template_definitions.append(
            #     v.c_f[n][t] == z3.If(v.c_f[n][t-1] < target_cwnd,
            #                          v.c_f[n][t-1] + v.alpha,
            #                          v.c_f[n][t-1] - v.alpha))
            # template_definitions.append(
            #     v.c_f[n][t] == target_cwnd)
    return template_definitions


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    ret = ""
    rhs_loss = (f"{solution.eval(coeffs['c_f[n]_loss'])}"
                f"c_f[n][t-{c.R}]"
                f" + {solution.eval(coeffs['ack_f[n]_loss'])}"
                f"(S_f[n][t-{c.R}]-S_f[n][t-{cc_ideal.history}])"
                f" + {solution.eval(consts['c_f[n]_loss'])}alpha")
    rhs_noloss = (f"{solution.eval(coeffs['c_f[n]_noloss'])}"
                  f"c_f[n][t-{c.R}]"
                  f" + {solution.eval(coeffs['ack_f[n]_noloss'])}"
                  f"(S_f[n][t-{c.R}]-S_f[n][t-{cc_ideal.history}])"
                  f" + {solution.eval(consts['c_f[n]_noloss'])}alpha")
    ret += (f"if(Ld_f[n][t] > Ld_f[n][t-1]):\n"
            f"\ttarget_cwnd = {rhs_loss}\n"
            f"else:\n"
            f"\ttarget_cwnd = {rhs_noloss}\n")
    ret += (f"\nif(c_f[n][t-1] < target_cwnd):\n"
            f"\tc_f[n][t] = c_f[n][t-1] + alpha\n"
            f"else:\n"
            f"\tc_f[n][t] = max(alpha, c_f[n][t-1] - alpha)\n")

    return ret


known_solution = None
known_solution_list = [
    coeffs['c_f[n]_loss'] == 1/2,
    coeffs['ack_f[n]_loss'] == 0,
    consts['c_f[n]_loss'] == 0,

    coeffs['c_f[n]_noloss'] == 1/2,
    coeffs['ack_f[n]_noloss'] == 1/2,
    consts['c_f[n]_noloss'] == 0,
]
known_solution = z3.And(*known_solution_list)
assert(isinstance(known_solution, z3.ExprRef))

# search_constraints = z3.And(search_constraints, known_solution)
# assert(isinstance(search_constraints, z3.ExprRef))

# ----------------------------------------------------------------
# IDEAL LINK
cc_ideal = CegisConfig()
cc_ideal.name = "ideal"
cc_ideal.synth_ss = False
cc_ideal.infinite_buffer = False
cc_ideal.dynamic_buffer = True
cc_ideal.buffer_size_multiplier = 1
cc_ideal.template_qdel = False
cc_ideal.template_queue_bound = False
cc_ideal.N = 1
cc_ideal.history = 3
cc_ideal.cca = "paced"

cc_ideal.desired_util_f = 1
cc_ideal.desired_queue_bound_multiplier = 1/2
cc_ideal.desired_queue_bound_alpha = 3
cc_ideal.desired_loss_count_bound = 3
cc_ideal.desired_loss_amount_bound_multiplier = 1/2
cc_ideal.desired_loss_amount_bound_alpha = 3

cc_ideal.ideal_link = True
cc_ideal.feasible_response = False

ideal = CCmatic(cc_ideal)
ideal.setup_config_vars()
c, _, v = ideal.c, ideal.s, ideal.v
template_definitions_ideal = get_template_definitions(cc_ideal, c, v)

ideal.setup_cegis_loop(
    search_constraints,
    template_definitions_ideal, generator_vars, get_solution_str)

# ----------------------------------------------------------------
# ADVERSARIAL LINK
cc_adv = copy.copy(cc_ideal)
cc_adv.name = "adv"

cc_adv.desired_util_f = 0.5
cc_adv.desired_queue_bound_multiplier = 1.5
cc_adv.desired_queue_bound_alpha = 3
cc_adv.desired_loss_count_bound = 3
cc_adv.desired_loss_amount_bound_multiplier = 1.5
cc_adv.desired_loss_amount_bound_alpha = 3

cc_adv.ideal_link = False
cc_adv.feasible_response = True

adv = CCmatic(cc_adv)
adv.setup_config_vars()
c, _, v = adv.c, adv.s, adv.v
template_definitions_adv = get_template_definitions(cc_adv, c, v)

adv.setup_cegis_loop(
    search_constraints,
    template_definitions_adv, generator_vars, get_solution_str)

# ----------------------------------------------------------------
# MULTI VERIFIER
joint = ideal
joint.critical_generator_vars = critical_generator_vars
joint.verifier_vars = ideal.verifier_vars + adv.verifier_vars
joint.definition_vars = ideal.definition_vars + adv.definition_vars
joint.definitions = z3.And(ideal.definitions, adv.definitions)
joint.specification = z3.And(ideal.specification, adv.specification)

# Dereference the ptr first, to avoid inf recursion.
ideal_get_counter_example_str = ideal.get_counter_example_str
ideal_get_generator_view = ideal.get_generator_view


def get_counter_example_str(*args, **kwargs):
    ret = "Ideal" + "-"*32 + "\n"
    ret += ideal_get_counter_example_str(*args, **kwargs)

    ret += "\nAdversarial" + "-"*(32-6) + "\n"
    ret += adv.get_counter_example_str(*args, **kwargs)
    return ret


def get_verifier_view(*args, **kwargs):
    return get_counter_example_str(*args, **kwargs)


def get_generator_view(*args, **kwargs):
    ret = "Ideal" + "-"*32 + "\n"
    ret += ideal_get_generator_view(*args, **kwargs)

    ret += "\nAdversarial" + "-"*(32-6) + "\n"
    ret += adv.get_generator_view(*args, **kwargs)
    return ret


joint.get_counter_example_str = get_counter_example_str
joint.get_verifier_view = get_verifier_view
joint.get_generator_view = get_generator_view
# ccmatic_joint.search_constraints = z3.And(search_constraints, known_solution)

logger.info("Ideal: " + cc_ideal.desire_tag())
logger.info("Adver: " + cc_adv.desire_tag())
joint.run_cegis(known_solution)
