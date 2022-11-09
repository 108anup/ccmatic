import copy
import logging
from fractions import Fraction
from typing import Dict, List, Union

import z3
from ccac.config import ModelConfig
from ccac.variables import Variables
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten_dict, get_product_ite, try_except
from cegis.multi_cegis import MultiCegis

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
domain_clauses = []

# For determining lhs, under cond, the coeff of rhs is:
# Gen__coeff_lhs_rhs_cond

rhss_cwnd = ['c_f[n]', 's_f[n]', 'ack_f[n]']
rhss_expr = ['c_f[n]', 'ack_f[n]']
rhss_cond = ['c_f[n]', 's_f[n]', 'ack_f[n]', 'losses']
conds = ['loss', 'noloss']

coeffs: Dict[str, Dict[str, Union[Dict[str, z3.ArithRef], z3.ArithRef]]] = {
    f'{lhs}': {
        f'{cond}': {
            f'{rhs}': z3.Real(f'Gen__coeff_{lhs}_{cond}_{rhs}')
            for rhs in rhss_cwnd
        } for cond in conds
    } for lhs in ['c_f[n]']
}
coeffs.update({
    f'{lhs}': {
        f'{cond}': {
            f'{rhs}': z3.Real(f'Gen__coeff_{lhs}_{cond}_{rhs}')
            for rhs in rhss_expr
        } for cond in conds
    } for lhs in ['s_f[n]']
})
coeffs.update({
    f'{lhs}': {
        f'{rhs}': z3.Real(f'Gen__coeff_{lhs}_{rhs}')
        for rhs in rhss_cond
    } for lhs in ['cond']
})

# coeffs = {}
# coeffs['c_f[n]'] = {
#     's_f[n]_loss': z3.Real('Gen__coeff_c_f[n]_s_f[n]_loss'),
#     's_f[n]_noloss': z3.Real('Gen__coeff_c_f[n]_s_f[n]_noloss'),
#     'c_f[n]_loss': z3.Real('Gen__coeff_c_f[n]_c_f[n]_loss'),
#     'c_f[n]_noloss': z3.Real('Gen__coeff_c_f[n]_c_f[n]_noloss'),
#     'ack_f[n]_loss': z3.Real('Gen__coeff_c_f[n]_ack_f[n]_loss'),
#     'ack_f[n]_noloss': z3.Real('Gen__coeff_c_f[n]_ack_f[n]_noloss'),
# }
# coeffs['s_f[n]'] = {
#     's_f[n]_loss': z3.Real('Gen__coeff_s_f[n]_s_f[n]_loss'),
#     's_f[n]_noloss': z3.Real('Gen__coeff_s_f[n]_s_f[n]_noloss'),
#     'c_f[n]_loss': z3.Real('Gen__coeff_s_f[n]_c_f[n]_loss'),
#     'c_f[n]_noloss': z3.Real('Gen__coeff_s_f[n]_c_f[n]_noloss'),
#     'ack_f[n]_loss': z3.Real('Gen__coeff_s_f[n]_ack_f[n]_loss'),
#     'ack_f[n]_noloss': z3.Real('Gen__coeff_s_f[n]_ack_f[n]_noloss'),
# }

consts = {
    f'{lhs}': {
        f'{cond}': z3.Real(f'Gen__const_{lhs}_{cond}')
        for cond in conds
    } for lhs in ['c_f[n]', 's_f[n]']
}

# consts['c_f[n]'] = {
#     'c_f[n]_loss': z3.Real('Gen__const_c_f[n]_c_f[n]_loss'),
#     'c_f[n]_noloss': z3.Real('Gen__const_c_f[n]_c_f[n]_noloss'),
# }
# consts['s_f[n]'] = {
#     'c_f[n]_loss': z3.Real('Gen__const_s_f[n]_c_f[n]_loss'),
#     'c_f[n]_noloss': z3.Real('Gen__const_s_f[n]_c_f[n]_noloss'),
# }

generator_vars = (flatten_dict(coeffs) +
                  flatten_dict(consts))
critical_generator_vars = flatten_dict(coeffs)

# Search constr
search_range_conds = [-1, 0, 1]
# search_range_coeffs = [Fraction(i, 2) for i in range(-4, 5)]
search_range_coeffs = [Fraction(i, 2) for i in range(5)]
search_range_consts = [Fraction(i, 2) for i in range(-4, 5)]
# search_range_consts = [-1, 0, 1]
domain_clauses = []
for coeff in flatten_dict([coeffs[x] for x in ['c_f[n]', 's_f[n]']]):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeffs]))
for coeff in flatten_dict([coeffs[x] for x in ['cond']]):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_conds]))
for const in flatten_dict(consts):
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

            # RHS for expr
            rhs_loss = (
                get_product_ite(
                    coeffs['s_f[n]']['loss']['c_f[n]'], v.c_f[n][t-c.R],
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['s_f[n]']['loss']['ack_f[n]'], acked_bytes,
                    search_range_coeffs)
                + get_product_ite(
                    consts['s_f[n]']['loss'], v.alpha,
                    search_range_consts))
            rhs_noloss = (
                get_product_ite(
                    coeffs['s_f[n]']['noloss']['c_f[n]'], v.c_f[n][t-c.R],
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['s_f[n]']['noloss']['ack_f[n]'], acked_bytes,
                    search_range_coeffs)
                + get_product_ite(
                    consts['s_f[n]']['noloss'], v.alpha,
                    search_range_consts))
            rhs_expr = z3.If(loss_detected, rhs_loss, rhs_noloss)
            assert isinstance(rhs_expr, z3.ArithRef)

            # cond
            cond = (
                get_product_ite(
                    coeffs['cond']['c_f[n]'], v.c_f[n][t-c.R],
                    search_range_conds)
                + get_product_ite(
                    coeffs['cond']['ack_f[n]'], acked_bytes,
                    search_range_conds)
                + get_product_ite(
                    coeffs['cond']['s_f[n]'], rhs_expr,
                    search_range_conds)
                + coeffs['cond']['losses'] * loss_detected > 0)

            # RHS for cwnd
            rhs_cond = (
                get_product_ite(
                    coeffs['c_f[n]']['loss']['c_f[n]'], v.c_f[n][t-c.R],
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['c_f[n]']['loss']['s_f[n]'], rhs_expr,
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['c_f[n]']['loss']['ack_f[n]'], acked_bytes,
                    search_range_coeffs)
                + get_product_ite(
                    consts['c_f[n]']['loss'], v.alpha,
                    search_range_consts))
            rhs_nocond = (
                get_product_ite(
                    coeffs['c_f[n]']['noloss']['c_f[n]'], v.c_f[n][t-c.R],
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['c_f[n]']['noloss']['s_f[n]'], rhs_expr,
                    search_range_coeffs)
                + get_product_ite(
                    coeffs['c_f[n]']['noloss']['ack_f[n]'], acked_bytes,
                    search_range_coeffs)
                + get_product_ite(
                    consts['c_f[n]']['noloss'], v.alpha,
                    search_range_consts))
            next_cwnd = z3.If(cond, rhs_cond, rhs_nocond)
            assert isinstance(next_cwnd, z3.ArithRef)

            # next_cwnd = z3.If(v.c_f[n][t-1] < target_cwnd,
            #                   v.c_f[n][t-1] + v.alpha,
            #                   v.c_f[n][t-1] - v.alpha)
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
    # RHS for expr
    rhs_loss = (f"{solution.eval(coeffs['s_f[n]']['loss']['c_f[n]'])}"
                f"c_f[n][t-{c.R}]"
                f" + {solution.eval(coeffs['s_f[n]']['loss']['ack_f[n]'])}"
                f"(S_f[n][t-{c.R}]-S_f[n][t-{cc_ideal.history}])"
                f" + {solution.eval(consts['s_f[n]']['loss'])}alpha")
    rhs_noloss = (f"{solution.eval(coeffs['s_f[n]']['noloss']['c_f[n]'])}"
                  f"c_f[n][t-{c.R}]"
                  f" + {solution.eval(coeffs['s_f[n]']['noloss']['ack_f[n]'])}"
                  f"(S_f[n][t-{c.R}]-S_f[n][t-{cc_ideal.history}])"
                  f" + {solution.eval(consts['s_f[n]']['noloss'])}alpha")

    # cond
    cond = (f"{solution.eval(coeffs['cond']['c_f[n]'])}"
            f"c_f[n][t-{c.R}]"
            f" + {solution.eval(coeffs['cond']['ack_f[n]'])}"
            f"(S_f[n][t-{c.R}]-S_f[n][t-{cc_ideal.history}])"
            f" + {solution.eval(coeffs['cond']['s_f[n]'])}"
            f"expr"
            f" + {solution.eval(coeffs['cond']['losses'])}"
            f"Indicator(Ld_f[n][t] > Ld_f[n][t-1]) > 0")

    # RHS for cwnd
    rhs_cond = (f"{solution.eval(coeffs['c_f[n]']['loss']['c_f[n]'])}"
                f"c_f[n][t-{c.R}]"
                f" + {solution.eval(coeffs['c_f[n]']['loss']['s_f[n]'])}"
                f"expr"
                f" + {solution.eval(coeffs['c_f[n]']['loss']['ack_f[n]'])}"
                f"(S_f[n][t-{c.R}]-S_f[n][t-{cc_ideal.history}])"
                f" + {solution.eval(consts['c_f[n]']['loss'])}alpha")
    rhs_nocond = (f"{solution.eval(coeffs['c_f[n]']['noloss']['c_f[n]'])}"
                  f"c_f[n][t-{c.R}]"
                  f" + {solution.eval(coeffs['c_f[n]']['noloss']['s_f[n]'])}"
                  f"expr"
                  f" + {solution.eval(coeffs['c_f[n]']['noloss']['ack_f[n]'])}"
                  f"(S_f[n][t-{c.R}]-S_f[n][t-{cc_ideal.history}])"
                  f" + {solution.eval(consts['c_f[n]']['noloss'])}alpha")

    ret += (f"if(Ld_f[n][t] > Ld_f[n][t-1]):\n"
            f"\texpr = {rhs_loss}\n"
            f"else:\n"
            f"\texpr = {rhs_noloss}\n")

    ret += (f"\nif({cond}):\n"
            f"\tc_f[n][t] = max(alpha, {rhs_cond})\n"
            f"else:\n"
            f"\tc_f[n][t] = max(alpha, {rhs_nocond})\n")

    return ret


known_solution = None

# AIAD
known_solution_list = [
    coeffs['cond']['c_f[n]'] == 1,
    coeffs['cond']['s_f[n]'] == -1,
    coeffs['cond']['ack_f[n]'] == 0,
    coeffs['cond']['losses'] == 0,

    coeffs['c_f[n]']['loss']['c_f[n]'] == 1,
    coeffs['c_f[n]']['loss']['s_f[n]'] == 0,
    coeffs['c_f[n]']['loss']['ack_f[n]'] == 0,
    consts['c_f[n]']['loss'] == -1,

    coeffs['c_f[n]']['noloss']['c_f[n]'] == 1,
    coeffs['c_f[n]']['noloss']['s_f[n]'] == 0,
    coeffs['c_f[n]']['noloss']['ack_f[n]'] == 0,
    consts['c_f[n]']['noloss'] == 1
]

# Fixed target
known_solution_list = [
    coeffs['cond']['c_f[n]'] == 1,
    coeffs['cond']['s_f[n]'] == -1,
    coeffs['cond']['ack_f[n]'] == 0,
    coeffs['cond']['losses'] == 0,

    # coeffs['c_f[n]']['loss']['c_f[n]'] == 1,
    coeffs['c_f[n]']['loss']['s_f[n]'] == 0,
    # coeffs['c_f[n]']['loss']['ack_f[n]'] == 0,
    # consts['c_f[n]']['loss'] == -1,

    # coeffs['c_f[n]']['noloss']['c_f[n]'] == 1,
    coeffs['c_f[n]']['noloss']['s_f[n]'] == 0,
    # coeffs['c_f[n]']['noloss']['ack_f[n]'] == 0,
    # consts['c_f[n]']['noloss'] == 1,

    coeffs['s_f[n]']['loss']['c_f[n]'] == 1/2,
    coeffs['s_f[n]']['loss']['ack_f[n]'] == 0,
    consts['s_f[n]']['loss'] == 0,

    coeffs['s_f[n]']['noloss']['c_f[n]'] == 1/2,
    coeffs['s_f[n]']['noloss']['ack_f[n]'] == 1/2,
    consts['s_f[n]']['noloss'] == 1
]

# # Full known solution
# known_solution_list = [
#     coeffs['cond']['c_f[n]'] == 1,
#     coeffs['cond']['s_f[n]'] == -1,
#     coeffs['cond']['ack_f[n]'] == 0,
#     coeffs['cond']['losses'] == 0,

#     coeffs['c_f[n]']['loss']['c_f[n]'] == 1,
#     coeffs['c_f[n]']['loss']['s_f[n]'] == 0,
#     coeffs['c_f[n]']['loss']['ack_f[n]'] == 0,
#     consts['c_f[n]']['loss'] == -1,

#     coeffs['c_f[n]']['noloss']['c_f[n]'] == 1,
#     coeffs['c_f[n]']['noloss']['s_f[n]'] == 0,
#     coeffs['c_f[n]']['noloss']['ack_f[n]'] == 0,
#     consts['c_f[n]']['noloss'] == 1,

#     coeffs['s_f[n]']['loss']['c_f[n]'] == 1,
#     coeffs['s_f[n]']['loss']['ack_f[n]'] == -1/2,
#     consts['s_f[n]']['loss'] == 3/2,

#     coeffs['s_f[n]']['noloss']['c_f[n]'] == 0,
#     coeffs['s_f[n]']['noloss']['ack_f[n]'] == 1,
#     consts['s_f[n]']['noloss'] == 3/2
# ]

# known_solution_list = [
#     coeffs['c_f[n]_loss'] == 1/2,
#     coeffs['ack_f[n]_loss'] == 0,
#     consts['c_f[n]_loss'] == 0,

#     coeffs['c_f[n]_noloss'] == 1/2,
#     coeffs['ack_f[n]_noloss'] == 1/2,
#     consts['c_f[n]_noloss'] == 0,
# ]
known_solution = z3.And(*known_solution_list)
assert(isinstance(known_solution, z3.ExprRef))

search_constraints = z3.And(search_constraints, known_solution)
assert(isinstance(search_constraints, z3.ExprRef))

# ----------------------------------------------------------------
# IDEAL LINK
cc_ideal = CegisConfig()
# cc_ideal.DEBUG = True
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

logger.info("Ideal: " + cc_ideal.desire_tag())
logger.info("Adver: " + cc_adv.desire_tag())

# # ----------------------------------------------------------------
# # MULTI VERIFIER
# joint = ideal  # Joint must be adv, so that we can have max gap optimization for adv.
# joint.critical_generator_vars = critical_generator_vars
# joint.verifier_vars = ideal.verifier_vars + adv.verifier_vars
# joint.definition_vars = ideal.definition_vars + adv.definition_vars
# joint.definitions = z3.And(ideal.definitions, adv.definitions)
# joint.specification = z3.And(ideal.specification, adv.specification)

# # Dereference the ptr first, to avoid inf recursion.
# ideal_get_counter_example_str = ideal.get_counter_example_str
# ideal_get_generator_view = ideal.get_generator_view
# adv_get_counter_example_str = adv.get_counter_example_str
# adv_get_generator_view = adv.get_generator_view


# def get_counter_example_str(*args, **kwargs):
#     ret = "Ideal" + "-"*32 + "\n"
#     ret += ideal_get_counter_example_str(*args, **kwargs)

#     ret += "\nAdversarial" + "-"*(32-6) + "\n"
#     ret += adv_get_counter_example_str(*args, **kwargs)
#     return ret


# def get_verifier_view(
#         counter_example: z3.ModelRef,
#         verifier_vars: List[z3.ExprRef],
#         definition_vars: List[z3.ExprRef]) -> str:
#     return get_counter_example_str(counter_example, verifier_vars)


# def get_generator_view(*args, **kwargs):
#     ret = "Ideal" + "-"*32 + "\n"
#     ret += ideal_get_generator_view(*args, **kwargs)

#     ret += "\nAdversarial" + "-"*(32-6) + "\n"
#     ret += adv_get_generator_view(*args, **kwargs)
#     return ret


# joint.get_counter_example_str = get_counter_example_str
# joint.get_verifier_view = get_verifier_view
# joint.get_generator_view = get_generator_view
# # ccmatic_joint.search_constraints = z3.And(search_constraints, known_solution)
# # joint.run_cegis(known_solution)

# ----------------------------------------------------------------
# MULTI CEGIS
verifier_structs = [
    ideal.get_verifier_struct('ideal'), adv.get_verifier_struct('adv')]

multicegis = MultiCegis(
    generator_vars, search_constraints, critical_generator_vars,
    verifier_structs, ideal.ctx, None, None)
multicegis.get_solution_str = get_solution_str
try_except(multicegis.run)
