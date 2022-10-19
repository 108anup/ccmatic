import itertools
import logging
from fractions import Fraction
from typing import List

import z3
from ccac.model import loss_detected
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, get_product_ite

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)

cc = CegisConfig()
# cc.DEBUG = True
cc.synth_ss = False
cc.infinite_buffer = False
cc.dynamic_buffer = True
cc.buffer_size_multiplier = 1
cc.template_qdel = False
cc.template_queue_bound = False
cc.N = 1
cc.history = 4
cc.cca = "none"
# import ipdb; ipdb.set_trace()
# cc.template_loss_oracle = True

cc.desired_util_f = 0.55
cc.desired_queue_bound_multiplier = 2
cc.desired_loss_count_bound = 3
cc.desired_loss_amount_bound_multiplier = 2

cc.feasible_response = True

ccmatic = CCmatic(cc)
ccmatic.setup_config_vars()
c, s, v = ccmatic.c, ccmatic.s, ccmatic.v

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
domain_clauses = []

coeffs = {
    'c_f[n]_loss': z3.Real('Gen__coeff_c_f[n]_loss'),
    'c_f[n]_noloss': z3.Real('Gen__coeff_c_f[n]_noloss'),
    'ack_f[n]_loss': z3.Real('Gen__coeff_ack_f[n]_loss'),
    'ack_f[n]_noloss': z3.Real('Gen__coeff_ack_f[n]_noloss'),
    'r_f[n]_noloss': z3.Real('Gen__coeff_r_f[n]_noloss'),
    'r_f[n]_loss': z3.Real('Gen__coeff_r_f[n]_loss')
}

consts = {
    'c_f[n]_loss': z3.Real('Gen__const_c_f[n]_loss'),
    'c_f[n]_noloss': z3.Real('Gen__const_c_f[n]_noloss'),
    'r_f[n]_noloss': z3.Real('Gen__const_r_f[n]_noloss'),
    'r_f[n]_loss': z3.Real('Gen__const_r_f[n]_loss')
}

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))
critical_generator_vars = flatten(list(coeffs.values()))

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

template_definitions = []
first = cc.history
for n in range(c.N):
    for t in range(first, c.T):
        # paced is already there to ensure r_f is set and c_f > 0

        acked_bytes = v.S_f[n][t-c.R] - v.S_f[n][t-cc.history]
        loss_detected = v.Ld_f[n][t] > v.Ld_f[n][t-c.R]
        rhs_loss = (
            get_product_ite(
                coeffs['c_f[n]_loss'], v.c_f[n][t-c.R], search_range_coeffs)
            + get_product_ite(
                coeffs['ack_f[n]_loss'], acked_bytes, search_range_coeffs)
            + consts['c_f[n]_loss'])
        rhs_noloss = (
            get_product_ite(
                coeffs['c_f[n]_noloss'], v.c_f[n][t-c.R], search_range_coeffs)
            + get_product_ite(
                coeffs['ack_f[n]_noloss'], acked_bytes, search_range_coeffs)
            + consts['c_f[n]_noloss'])
        rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
        assert isinstance(rhs, z3.ArithRef)
        template_definitions.append(
            v.c_f[n][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                                 rhs, cc.template_cca_lower_bound)
        )

# Pacing
for n in range(c.N):
    for t in range(first):
        # Basic constraints when adversary decides cwnd, loss
        template_definitions.append(v.c_f[n][t] > 0)
        template_definitions.append(v.r_f[n][t] == v.c_f[n][t] / c.R)

    for t in range(first, c.T):
        loss_detected = v.Ld_f[n][t] > v.Ld_f[n][t-c.R]
        template_definitions.append(v.c_f[n][t] > 0)
        rhs_loss = get_product_ite(
            coeffs['r_f[n]_loss'], v.c_f[n][t] / c.R, search_range_coeffs) \
            + consts['r_f[n]_loss']
        rhs_noloss = get_product_ite(
            coeffs['r_f[n]_noloss'], v.c_f[n][t] / c.R, search_range_coeffs) \
            + consts['r_f[n]_noloss']
        rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
        assert isinstance(rhs, z3.ArithRef)
        template_definitions.append(
            v.r_f[n][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                                 rhs, cc.template_cca_lower_bound)
        )


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    ret = "CWND:\n"
    rhs_loss = (f"{solution.eval(coeffs['c_f[n]_loss'])}"
                f"c_f[n][t-{c.R}]"
                f" + {solution.eval(coeffs['ack_f[n]_loss'])}"
                f"(S_f[n][t-{c.R}]-S_f[n][t-{cc.history}])"
                f" + {solution.eval(consts['c_f[n]_loss'])}")
    rhs_noloss = (f"{solution.eval(coeffs['c_f[n]_noloss'])}"
                  f"c_f[n][t-{c.R}]"
                  f" + {solution.eval(coeffs['ack_f[n]_noloss'])}"
                  f"(S_f[n][t-{c.R}]-S_f[n][t-{cc.history}])"
                  f" + {solution.eval(consts['c_f[n]_noloss'])}")
    ret += (f"if(Ld_f[n][t] > Ld_f[n][t-1]):\n"
            f"\tc_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_loss})\n"
            f"else:\n"
            f"\tc_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_noloss})\n")

    # Pacing
    ret += "PACING:\n"
    rhs_loss = (f"{solution.eval(coeffs['r_f[n]_loss'])}"
                f"c_f[n][t]/{c.R}"
                f" + {solution.eval(consts['r_f[n]_loss'])}")
    rhs_noloss = (f"{solution.eval(coeffs['r_f[n]_noloss'])}"
                  f"c_f[n][t]/{c.R}"
                  f" + {solution.eval(consts['r_f[n]_noloss'])}")
    ret += (f"if(Ld_f[n][t] > Ld_f[n][t-1]):\n"
            f"\tr_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_loss})\n"
            f"else:\n"
            f"\tr_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_noloss})\n")

    return ret


known_solution = None
known_solution_list = [
    coeffs['c_f[n]_loss'] == 1/2,
    coeffs['ack_f[n]_loss'] == 0,
    consts['c_f[n]_loss'] == 0,

    coeffs['c_f[n]_noloss'] == 1/2,
    coeffs['ack_f[n]_noloss'] == 1/2,
    consts['c_f[n]_noloss'] == 1,
]
known_solution = z3.And(*known_solution_list)
assert(isinstance(known_solution, z3.ExprRef))

# search_constraints = z3.And(search_constraints, coeffs['r_f[n]_loss'] == 1/2)
# search_constraints = z3.And(
#     search_constraints,
#     z3.And(
#         coeffs['c_f[n]_loss'] == coeffs['c_f[n]_noloss'],
#         coeffs['ack_f[n]_loss'] == coeffs['ack_f[n]_noloss'],
#         consts['c_f[n]_loss'] == consts['c_f[n]_noloss'],
#     ))

ccmatic.setup_cegis_loop(
    search_constraints,
    template_definitions, generator_vars, get_solution_str)
ccmatic.critical_generator_vars = critical_generator_vars
# ccmatic.search_constraints = z3.And(search_constraints, known_solution)
ccmatic.run_cegis(known_solution)
