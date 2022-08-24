import functools
import logging
from fractions import Fraction
from typing import List

import z3
from ccac.variables import VariableNames
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

import ccmatic.common  # Used for side effects
from ccmatic.cegis import CegisCCAGen, CegisConfig
from ccmatic.common import (flatten, get_product_ite, get_renamed_vars,
                            get_val_list)

from .verifier import (get_all_desired, get_cex_df,
                       get_desired_property_string, get_gen_cex_df,
                       run_verifier_incomplete, setup_cegis_basic)

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


DEBUG = False
cc = CegisConfig()
cc.infinite_buffer = False
cc.dynamic_buffer = True
cc.buffer_size_multiplier = 1
cc.template_queue_bound = True
cc.template_mode_switching = True

cc.desired_util_f = 0.33
cc.desired_queue_bound_multiplier = 2
cc.desired_loss_bound = 3
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)

(desired, fefficient, bounded_queue, bounded_loss,
 ramp_up_cwnd, ramp_down_cwnd, ramp_down_q, ramp_down_bq,
 total_losses) = get_all_desired(cc, c, v)

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
vn = VariableNames(v)
lower_bound = 0.01
coeffs = {
    'c_f[0]_mode0_if': z3.Real('Gen__coeff_c_f[0]_mode0_if'),
    'c_f[0]_mode0_else': z3.Real('Gen__coeff_c_f[0]_mode0_else'),
    'ack_f[0]_mode0_if': z3.Real('Gen__coeff_ack_f[0]_mode0_if'),
    'ack_f[0]_mode0_else': z3.Real('Gen__coeff_ack_f[0]_mode0_else'),

    'c_f[0]_mode1_if': z3.Real('Gen__coeff_c_f[0]_mode1_if'),
    # 'c_f[0]_mode1_else': z3.Real('Gen__coeff_c_f[0]_mode1_else'),
    'ack_f[0]_mode1_if': z3.Real('Gen__coeff_ack_f[0]_mode1_if'),
    # 'ack_f[0]_mode1_else': z3.Real('Gen__coeff_ack_f[0]_mode1_else')
}

consts = {
    'c_f[0]_mode0_if': z3.Real('Gen__const_c_f[0]_mode0_if'),
    'c_f[0]_mode0_else': z3.Real('Gen__const_c_f[0]_mode0_else'),

    'c_f[0]_mode1_if': z3.Real('Gen__const_c_f[0]_mode1_if'),
    # 'c_f[0]_mode1_else': z3.Real('Gen__const_c_f[0]_mode1_else')
}

# qsize_thresh_choices = [Fraction(i, 8) for i in range(2 * 8 + 1)]
qsize_thresh_choices = [x for x in range(1, c.T)]

# Search constr
search_range_coeff = [Fraction(i, 2) for i in range(5)]
# search_range_const = [Fraction(i, 2) for i in range(-4, 5)]
search_range_const = [-1, 0, 1]
search_range_const = [0]
# search_range = [-1, 0, 1]
domain_clauses = []
for coeff in flatten(list(coeffs.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeff]))
for const in flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[const == val for val in search_range_const]))
domain_clauses.append(z3.Or(
    *[v.qsize_thresh == val for val in qsize_thresh_choices]))
search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))

# Generator definitions
template_definitions = []
first = cc.history
for t in range(1, c.T):
    # mode selection
    loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    template_definitions.append(
        z3.If(loss_detected, v.mode_f[0][t],  # else
              z3.If(v.exceed_queue_f[0][t], z3.Not(v.mode_f[0][t]),
              v.mode_f[0][t] == v.mode_f[0][t-1])))
    # True means mode0 otherwise mode1
    # Check if we want this_decrease instead of v.exceed_queue_f

for t in range(first, c.T):
    loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    delay_detected = v.exceed_queue_f[0][t]
    acked_bytes = v.S_f[0][t-c.R] - v.S_f[0][t-cc.history]

    # When did decrease happen
    # TODO: see if we want to replace the last statement
    #  with (S_f[n][t-c.R] > last_decrease[n][t-1])
    this_decrease = z3.And(delay_detected,
                           v.S_f[0][t-c.R] > v.S_f[0][t-c.R-1],
                           v.S_f[0][t-1-c.R] >= v.last_decrease_f[0][t-1])
    # mode 0
    rhs_mode0_if = (
        get_product_ite(coeffs['c_f[0]_mode0_if'], v.c_f[0][t-c.R],
                        search_range_coeff)
        + get_product_ite(coeffs['ack_f[0]_mode0_if'], acked_bytes,
                          search_range_coeff)
        + consts['c_f[0]_mode0_if'])
    rhs_mode0_else = (
        get_product_ite(coeffs['c_f[0]_mode0_else'], v.c_f[0][t-c.R],
                        search_range_coeff)
        + get_product_ite(coeffs['ack_f[0]_mode0_else'], acked_bytes,
                          search_range_coeff)
        + consts['c_f[0]_mode0_else'])

    # mode 1
    rhs_mode1_if = (
        get_product_ite(coeffs['c_f[0]_mode1_if'], v.c_f[0][t-c.R],
                        search_range_coeff)
        + get_product_ite(coeffs['ack_f[0]_mode1_if'], acked_bytes,
                          search_range_coeff)
        + consts['c_f[0]_mode1_if'])

    rhs = z3.If(v.mode_f[0][t], z3.If(
        loss_detected, rhs_mode0_if, rhs_mode0_else), rhs_mode1_if)
    assert isinstance(rhs, z3.ArithRef)
    template_definitions.append(
        v.c_f[0][t] == z3.If(rhs >= lower_bound, rhs, lower_bound))

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_domain, ccac_definitions, *template_definitions)
assert isinstance(definitions, z3.ExprRef)

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())) +
                  [v.qsize_thresh])


# Method overrides
# These use function closures, hence have to be defined here.
# Can use partial functions to use these elsewhere.

def get_counter_example_str(counter_example: z3.ModelRef,
                            verifier_vars: List[z3.ExprRef]) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    df["this_decrease"] = [-1] + get_val_list(counter_example, [
        counter_example.eval(z3.And(
            v.exceed_queue_f[0][t],
            v.S_f[0][t-c.R] > v.S_f[0][t-c.R-1],
            v.S_f[0][t-1-c.R] >= v.last_decrease_f[0][t-1]
        ))
        for t in range(1, c.T)])
    df["v.mode_f"] = get_val_list(counter_example, v.mode_f[0])

    desired_string = get_desired_property_string(
        cc, c, fefficient, bounded_queue, bounded_loss,
        ramp_up_cwnd, ramp_down_bq, ramp_down_q, ramp_down_cwnd,
        total_losses, counter_example)
    ret = "{}\n{}.".format(df, desired_string)
    return ret


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    rhs_mode0_if = (f"{solution.eval(coeffs['c_f[0]_mode0_if'])}"
                    f"v.c_f[0][t-{c.R}]"
                    f" + {solution.eval(coeffs['ack_f[0]_mode0_if'])}"
                    f"(S_f[0][t-{c.R}]-S_f[0][t-{cc.history}])"
                    f" + {solution.eval(consts['c_f[0]_mode0_if'])}")
    rhs_mode0_else = (f"{solution.eval(coeffs['c_f[0]_mode0_else'])}"
                      f"v.c_f[0][t-{c.R}]"
                      f" + {solution.eval(coeffs['ack_f[0]_mode0_else'])}"
                      f"(S_f[0][t-{c.R}]-S_f[0][t-{cc.history}])"
                      f" + {solution.eval(consts['c_f[0]_mode0_else'])}")
    rhs_mode1_if = (f"{solution.eval(coeffs['c_f[0]_mode1_if'])}"
                    f"v.c_f[0][t-{c.R}]"
                    f" + {solution.eval(coeffs['ack_f[0]_mode1_if'])}"
                    f"(S_f[0][t-{c.R}]-S_f[0][t-{cc.history}])"
                    f" + {solution.eval(consts['c_f[0]_mode1_if'])}")
    # rhs_mode1_else = (f"{solution.eval(coeffs['c_f[0]_mode1_else'])}"
    #                   f"v.c_f[0][t-{c.R}]"
    #                   f" + {solution.eval(coeffs['ack_f[0]_mode1_else'])}"
    #                   f"(S_f[0][t-{c.R}]-S_f[0][t-{cc.history}])"
    #                   f" + {solution.eval(consts['c_f[0]_mode1_else'])}")

    ret = (f"if(v.mode_f[0][t]):\n"
           f"\tif(Ld_f[0][t] > Ld_f[0][t-1]):\n"
           f"\t\tc_f[0][t] = max({lower_bound}, {rhs_mode0_if})\n"
           f"\telse:\n"
           f"\t\tc_f[0][t] = max({lower_bound}, {rhs_mode0_else})\n"
           f"else:\n"
           f"\tc_f[0][t] = max({lower_bound}, {rhs_mode1_if})\n\n"

           f"if(Ld_f[0][t] > Ld_f[0][t-1]):\n"
           f"\tmode[0][t] = True\n"
           f"elif(qbound[t-1][{solution.eval(v.qsize_thresh)}]):\n"
           f"\tmode[0][t] = False\n"
           f"else:\n"
           f"\tmode[0][t] = mode[0][t-1]")
    return ret


def get_verifier_view(
            counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
            definition_vars: List[z3.ExprRef]) -> str:
    return get_counter_example_str(counter_example, verifier_vars)


def get_generator_view(solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                       definition_vars: List[z3.ExprRef], n_cex: int) -> str:
    df = get_gen_cex_df(solution, v, vn, n_cex, c)

    g_last_decrease_f = get_renamed_vars(v.last_decrease_f[0], n_cex)
    g_exceed_queue_f = get_renamed_vars(v.exceed_queue_f[0], n_cex)
    g_S_f = get_renamed_vars(v.S_f[0], n_cex)
    df["this_decrease"] = [-1] + get_val_list(solution, [
        solution.eval(z3.And(
            g_exceed_queue_f[t],
            g_S_f[t-c.R] > g_S_f[t-c.R-1],
            g_S_f[t-1-c.R] >= g_last_decrease_f[t-1]
        ))
        for t in range(1, c.T)])
    df["v.mode_f"] = get_val_list(
        solution, get_renamed_vars(v.mode_f[0], n_cex))

    ret = "{}".format(df)
    return ret


# Known solution
known_solution = None

# Debugging:
if DEBUG:
    if(known_solution is not None):
        known_solver = MySolver()
        known_solver.warn_undeclared = False
        known_solver.add(known_solution)
        print(known_solver.check())
        print(known_solver.model())

    # Search constraints
    search_constraints = z3.And(search_constraints, known_solution)
    assert(isinstance(search_constraints, z3.ExprRef))
    with open('tmp/search.txt', 'w') as f:
        f.write(search_constraints.sexpr())

    # Definitions (including template)
    with open('tmp/definitions.txt', 'w') as f:
        f.write(definitions.sexpr())

try:
    cg = CegisCCAGen(generator_vars, verifier_vars, definition_vars,
                     search_constraints, definitions, specification, ctx,
                     known_solution)
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
