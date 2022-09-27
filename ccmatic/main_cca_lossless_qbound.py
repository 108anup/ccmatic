import functools
import logging
from fractions import Fraction
from typing import List

import z3
from ccac.variables import VariableNames
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

import ccmatic.common  # Used for side effects
from ccmatic.cegis import CegisCCAGen, CegisConfig, CegisMetaData
from ccmatic.common import (flatten, get_product_ite, get_renamed_vars,
                            get_val_list)

from .verifier import (get_cex_df, get_desired_necessary, get_desired_ss_invariant, get_gen_cex_df,
                       run_verifier_incomplete, setup_cegis_basic)

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


DEBUG = False
cc = CegisConfig()
cc.N = 1
cc.T = 9
cc.compose = False
cc.synth_ss = False

cc.infinite_buffer = True
cc.template_queue_bound = True

cc.desired_util_f = 0.66
cc.desired_queue_bound_multiplier = 2
cc.desired_loss_amount_bound_multiplier = 0
cc.desired_loss_count_bound = 0

(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)

d = get_desired_necessary(cc, c, v)
desired = d.desired_necessary

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
vn = VariableNames(v)
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

# qsize_thresh_choices = [Fraction(i, 8) for i in range(2 * 8 + 1)]
qsize_thresh_choices = [x for x in range(1, c.T)]

# Search constr
search_range_coeffs = [Fraction(i, 2) for i in range(5)]
search_range_consts = [-1, 0, 1]
domain_clauses = []
for coeff in flatten(list(coeffs.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeffs]))
for const in flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[const == val for val in search_range_consts]))
domain_clauses.append(z3.Or(
    *[v.qsize_thresh == val for val in qsize_thresh_choices]))
search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))

# Generator definitions
template_definitions = []
first = cc.history
for n in range(c.N):
    for t in range(first, c.T):
        # loss_detected = v.Ld_f[n][t] > v.Ld_f[n][t-1]

        # This is meaningless as c.C * (c.R + c.D) is unknown...
        # delay_detected = (v.A_f[n][t-c.R] - v.Ld_f[n][t] - v.S_f[n][t-c.R]
        #                   >= v.qsize_thresh * c.C * (c.R + c.D))

        delay_detected = v.exceed_queue_f[n][t]

        # Decrease this time iff queue exceeded AND new qdelay measurement
        # (i.e., new packets received) AND in the previous cycle we had received
        # all the packets sent since last decrease (S_f[n][t-c.R-1] >=
        # last_decrease[n][t-1])
        # TODO: see if we want to replace the last
        #  statement with (S_f[n][t-c.R] > last_decrease[n][t-1])
        this_decrease = z3.And(delay_detected,
                               v.S_f[n][t-c.R] > v.S_f[n][t-c.R-1],
                               v.S_f[n][t-1-c.R] >= v.last_decrease_f[n][t-1])

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
        rhs = z3.If(this_decrease, rhs_delay, rhs_nodelay)
        assert isinstance(rhs, z3.ArithRef)
        template_definitions.append(
            v.c_f[n][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                                 rhs, cc.template_cca_lower_bound)
        )

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_domain, ccac_definitions, *template_definitions)
assert isinstance(definitions, z3.ExprRef)

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())) + [v.qsize_thresh])
critical_generator_vars = flatten(list(coeffs.values()))


# Method overrides
# These use function closures, hence have to be defined here.
# Can use partial functions to use these elsewhere.


def get_counter_example_str(counter_example: z3.ModelRef,
                            verifier_vars: List[z3.ExprRef]) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    for n in range(c.N):
        df[f"this_decrease_f_{n}"] = [-1] + get_val_list(counter_example, [
            counter_example.eval(z3.And(
                v.exceed_queue_f[n][t],
                v.S_f[n][t-c.R] > v.S_f[n][t-c.R-1],
                v.S_f[n][t-1-c.R] >= v.last_decrease_f[n][t-1]
            ))
            for t in range(1, c.T)])
    desired_string = d.to_string(cc, c, counter_example)
    ret = "{}\n{}.".format(df, desired_string)
    return ret


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
    ret = (f"if(qbound[t-1][{solution.eval(v.qsize_thresh)}]):\n"
           f"\tc_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_delay})\n"
           f"else:\n"
           f"\tc_f[n][t] = max({cc.template_cca_lower_bound}, {rhs_nodelay})")
    return ret


def get_verifier_view(
            counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
            definition_vars: List[z3.ExprRef]) -> str:
    return get_counter_example_str(counter_example, verifier_vars)


def get_generator_view(solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                       definition_vars: List[z3.ExprRef], n_cex: int) -> str:
    df = get_gen_cex_df(solution, v, vn, n_cex, c)

    for n in range(c.N):
        g_last_decrease_f = get_renamed_vars(v.last_decrease_f[n], n_cex)
        g_exceed_queue_f = get_renamed_vars(v.exceed_queue_f[n], n_cex)
        g_S_f = get_renamed_vars(v.S_f[0], n_cex)
        df[f"this_decrease_f_{n}"] = [-1] + get_val_list(solution, [
            solution.eval(z3.And(
                g_exceed_queue_f[t],
                g_S_f[t-c.R] > g_S_f[t-c.R-1],
                g_S_f[t-1-c.R] >= g_last_decrease_f[t-1]
            ))
            for t in range(1, c.T)])

    ret = "{}".format(df)
    return ret


# Known solution
known_solution = None

# known_solution_list = []
# known_solution_list.append(coeffs['c_f[0]_delay'] == 1/2)
# known_solution_list.append(coeffs['ack_f[0]_delay'] == 0)
# known_solution_list.append(consts['c_f[0]_delay'] == 0)

# known_solution_list.append(coeffs['c_f[0]_nodelay'] == 1)
# known_solution_list.append(coeffs['ack_f[0]_nodelay'] == 0)
# known_solution_list.append(consts['c_f[0]_nodelay'] == 1)
# known_solution = z3.And(*known_solution_list)
# assert(isinstance(known_solution, z3.ExprRef))

# Debugging:
debug_known_solution = None
if DEBUG:
    if(known_solution is not None):
        known_solver = MySolver()
        known_solver.warn_undeclared = False
        known_solver.add(known_solution)
        print(known_solver.check())
        print(known_solver.model())

    # Search constraints
    debug_known_solution = known_solution
    search_constraints = z3.And(search_constraints, known_solution)
    assert(isinstance(search_constraints, z3.ExprRef))
    with open('tmp/search.txt', 'w') as f:
        f.write(search_constraints.sexpr())

    # Definitions (including template)
    with open('tmp/definitions.txt', 'w') as f:
        f.write(definitions.sexpr())

try:
    md = CegisMetaData(critical_generator_vars)
    cg = CegisCCAGen(generator_vars, verifier_vars, definition_vars,
                     search_constraints, definitions, specification, ctx,
                     debug_known_solution, md)
    cg.get_solution_str = get_solution_str
    cg.get_counter_example_str = get_counter_example_str
    cg.get_generator_view = get_generator_view
    cg.get_verifier_view = get_verifier_view
    run_verifier = functools.partial(
        run_verifier_incomplete, c=c, v=v, ctx=ctx)
    cg.run_verifier = run_verifier
    cg.run()

except Exception:
    import sys
    import traceback

    import ipdb
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
