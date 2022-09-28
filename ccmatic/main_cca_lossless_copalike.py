import itertools
import logging
from fractions import Fraction
from typing import List

import z3
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, get_product_ite

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)

cc = CegisConfig()
cc.compose = True
cc.infinite_buffer = True
cc.template_qdel = True

cc.desired_util_f = 0.66
cc.desired_queue_bound_multiplier = 2
cc.desired_loss_amount_bound_multiplier = 0
cc.desired_loss_count_bound = 0

ccmatic = CCmatic(cc)
ccmatic.setup_config_vars()
c, s, v = ccmatic.c, ccmatic.s, ccmatic.v

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
domain_clauses = []

coeffs = {
    'c_f[n]_delay': z3.Real('Gen__coeff_c_f[n]_delay'),
    'c_f[n]_nodelay': z3.Real('Gen__coeff_c_f[n]_nodelay'),
    'ack_f[n]_delay': z3.Real('Gen__coeff_ack_f[n]_delay'),
    'ack_f[n]_nodelay': z3.Real('Gen__coeff_ack_f[n]_nodelay')
}

consts = {
    'c_f[n]_delay': z3.Real('Gen__const_c_f[n]_delay'),
    'c_f[n]_nodelay': z3.Real('Gen__const_c_f[n]_nodelay'),
}

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))

# Search constr
search_range_coeffs = [Fraction(i, 2) for i in range(5)]
search_range_consts = [-1, 0, 1]
domain_clauses = []
for coeff in flatten(list(coeffs.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeffs]))
for const in flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[const == val for val in search_range_consts]))

# All expressions should be different. Otherwise that expression is not needed.
conds = ['delay', 'nodelay']
for pair in itertools.combinations(conds, 2):
    is_same = z3.And(
        coeffs['c_f[n]_{}'.format(pair[0])] ==
        coeffs['c_f[n]_{}'.format(pair[1])],
        coeffs['ack_f[n]_{}'.format(pair[0])] ==
        coeffs['ack_f[n]_{}'.format(pair[1])])
    domain_clauses.append(z3.Not(is_same))

template_definitions = []
first = cc.history
for n in range(c.N):
    for t in range(first, c.T):
        # paced is already there to ensure r_f is set and c_f > 0

        acked_bytes = v.S_f[n][t] - v.S_f[n][t-cc.history]

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
        # Prefer decrease
        acked_bytes = v.S_f[n][t-c.R] - v.S_f[n][t-cc.history]
        rhs_delay = (
            get_product_ite(
                coeffs['c_f[n]_delay'], v.c_f[n][t-c.R], search_range_coeffs)
            + get_product_ite(
                coeffs['ack_f[n]_delay'], acked_bytes, search_range_coeffs)
            + consts['c_f[n]_delay'])
        rhs_nodelay = (
            get_product_ite(
                coeffs['c_f[n]_nodelay'], v.c_f[n][t-c.R], search_range_coeffs)
            + get_product_ite(
                coeffs['ack_f[n]_nodelay'], acked_bytes, search_range_coeffs)
            + consts['c_f[n]_nodelay'])
        rhs = z3.If(decr_allowed, rhs_delay, rhs_nodelay)
        assert isinstance(rhs, z3.ArithRef)
        template_definitions.append(
            v.c_f[n][t] == z3.If(rhs >= v.alpha,
                                 rhs, v.alpha)
        )


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    rhs_delay = (f"{solution.eval(coeffs['c_f[n]_delay'])}"
                 f"c_f[n][t-{c.R}]"
                 f" + {solution.eval(coeffs['ack_f[n]_delay'])}"
                 f"(S_f[n][t-{c.R}]-S_f[n][t-{cc.history}])"
                 f" + {solution.eval(consts['c_f[n]_delay'])}")
    rhs_nodelay = (f"{solution.eval(coeffs['c_f[n]_nodelay'])}"
                   f"c_f[n][t-{c.R}]"
                   f" + {solution.eval(coeffs['ack_f[n]_nodelay'])}"
                   f"(S_f[n][t-{c.R}]-S_f[n][t-{cc.history}])"
                   f" + {solution.eval(consts['c_f[n]_nodelay'])}")
    ret = (f"if(decr_allowed):\n"
           f"\tc_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_delay})\n"
           f"else:\n"
           f"\tc_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_nodelay})")
    return ret


ccmatic.setup_cegis_loop(
    template_definitions, generator_vars, get_solution_str)
ccmatic.run_cegis(None)
