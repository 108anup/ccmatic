import functools
import logging
from fractions import Fraction
from typing import List

import z3
from ccac.variables import VariableNames
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic.cegis import CegisCCAGen, CegisConfig
from ccmatic.common import flatten, get_product_ite
from cegis.util import get_raw_value

from .verifier import (get_cex_df, get_desired_necessary,
                       get_desired_ss_invariant, get_gen_cex_df,
                       get_periodic_constraints, run_verifier_incomplete,
                       setup_cegis_basic)

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


DEBUG = False
cc = CegisConfig()
cc.compose = False
# cc.synth_ss = True
# cc.T = 5 + cc.history * 2
# cc.T = 15
# cc.N = 2

cc.infinite_buffer = True

cc.desired_util_f = 0.66
cc.desired_queue_bound_multiplier = 2
cc.desired_loss_amount_bound_multiplier = 0
cc.desired_loss_count_bound = 0
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)

# Periodic cex
# environment = z3.And(environment, get_periodic_constraints(cc, c, v))

# # To get rid of monotonic solutions
# cwnd_has_upper_bound = []
# for t in range(cc.history, c.T):
#     cwnd_has_upper_bound.append(v.c_f[0][t] <= 10 * c.C * (c.R + c.D))
# desired = z3.And(d.desired_in_ss, z3.And(*cwnd_has_upper_bound))

if(cc.synth_ss):
    d = get_desired_ss_invariant(cc, c, v)
    desired = d.desired_invariant
else:
    d = get_desired_necessary(cc, c, v)
    desired = d.desired_necessary

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
domain_clauses = []

if(cc.synth_ss):
    # Steady state search
    assert d.steady_state_variables
    for sv in d.steady_state_variables:
        domain_clauses.append(sv.lo >= 0)
        domain_clauses.append(sv.hi >= 0)
        domain_clauses.append(sv.hi >= sv.lo)

    sv_dict = {sv.name: sv for sv in d.steady_state_variables}
    # domain_clauses.extend([
    #     sv_dict['cwnd'].lo == c.C * (c.R),
    #     sv_dict['cwnd'].hi == (cc.history-1) * c.C * (c.R + c.D),
    #     sv_dict['queue'].lo == 0,
    #     sv_dict['queue'].hi == 2 * c.C * (c.R + c.D),
    # ])

    domain_clauses.extend([
        sv_dict['cwnd'].lo == c.C * (c.R),
        sv_dict['cwnd'].hi == 3 * c.C * (c.R + c.D),
        sv_dict['queue'].lo == 0,
        sv_dict['queue'].hi == 3 * c.C * (c.R + c.D),
    ])

    domain_clauses.extend([
        sv_dict['cwnd'].lo >= 0.5 * c.C * (c.R),
        sv_dict['cwnd'].hi <= c.T * c.C * (c.R + c.D),
        sv_dict['queue'].lo == 0,
        sv_dict['queue'].hi == 3 * c.C * (c.R + c.D),
    ])

vn = VariableNames(v)
rhs_var_symbols = ['S_f']
# rhs_var_symbols = ['c_f', 'S_f']
lhs_var_symbols = ['c_f']
lvar_lower_bounds = {
    'c_f': 0.01
}
n_coeffs = len(rhs_var_symbols) * cc.history
n_const = 1

# Coeff for determining rhs var, of lhs var, at shift t
coeffs = {
    lvar: [
        [z3.Real('Gen__coeff_{}_{}_{}'.format(lvar, rvar, h))
         for h in range(cc.history)]
        for rvar in rhs_var_symbols
    ] for lvar in lhs_var_symbols
}
consts = {
    lvar: z3.Real('Gen__const_{}'.format(lvar))
    for lvar in lhs_var_symbols
}

# Search constr
search_range = [Fraction(i, 2) for i in range(-4, 5)]
search_range = [-1, 0, 1]
for coeff in flatten(list(coeffs.values())) + flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range]))
search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))

# Generator definitions
template_definitions = []


# TODO(108anup): For multi-flow, need to add constraints for both CCAs.
def get_expr(lvar_symbol, t, n) -> z3.ArithRef:
    term_list = []
    for rvar_idx in range(len(rhs_var_symbols)):
        rvar_symbol = rhs_var_symbols[rvar_idx]
        for h in range(cc.history):
            this_coeff = coeffs[lvar_symbol][rvar_idx][h]
            time_idx = t - c.R - h
            rvar = eval(f'v.{rvar_symbol}[{n}]')
            this_term = get_product_ite(
                this_coeff, rvar[time_idx], search_range)
            term_list.append(this_term)
    expr = z3.Sum(*term_list) + consts[lvar_symbol]
    assert isinstance(expr, z3.ArithRef)
    return expr


first = cc.history
for n in range(c.N):
    for lvar_symbol in lhs_var_symbols:
        lower_bound = lvar_lower_bounds[lvar_symbol]
        for t in range(first, c.T):
            lvar = eval(f'v.{lvar_symbol}[{n}]')
            rhs = get_expr(lvar_symbol, t, n)
            template_definitions.append(
                lvar[t] == z3.If(rhs >= lower_bound, rhs, lower_bound))

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_domain, ccac_definitions, *template_definitions)
assert(isinstance(definitions, z3.ExprRef))

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))
if(d.steady_state_variables):
    for sv in d.steady_state_variables:
        generator_vars.append(sv.lo)
        generator_vars.append(sv.hi)


# Method overrides
# These use function closures, hence have to be defined here.
# Can use partial functions to use these elsewhere.


def get_counter_example_str(counter_example: z3.ModelRef,
                            verifier_vars: List[z3.ExprRef]) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    desired_string = d.to_string(cc, c, counter_example)
    ret = "{}\n{}.".format(df, desired_string)
    return ret


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    assert(len(lhs_var_symbols) == 1)
    lvar_symbol = "c_f"
    rhs_expr = ""
    for rvar_idx in range(len(rhs_var_symbols)):
        rvar_symbol = rhs_var_symbols[rvar_idx]
        for h in range(cc.history):
            this_coeff = coeffs[lvar_symbol][rvar_idx][h]
            this_coeff_val = get_raw_value(solution.eval(this_coeff))
            if(this_coeff_val != 0):
                rhs_expr += "+ {}{}[t-{}] ".format(
                    this_coeff_val, rvar_symbol, h+1)
    this_const = consts[lvar_symbol]
    this_const_val = solution.eval(this_const)
    rhs_expr += "+ {}".format(this_const_val)
    ret = "{}[t] = max({}, {})".format(
        lvar_symbol, lvar_lower_bounds[lvar_symbol], rhs_expr)

    if d.steady_state_variables:
        for sv in d.steady_state_variables:
            ret += "\n{}: [{}, {}]".format(
                sv.name, solution.eval(sv.lo), solution.eval(sv.hi))
    return ret


def get_verifier_view(
            counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
            definition_vars: List[z3.ExprRef]) -> str:
    return get_counter_example_str(counter_example, verifier_vars)


def get_generator_view(solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                       definition_vars: List[z3.ExprRef], n_cex: int) -> str:
    gen_view_str = "{}".format(get_gen_cex_df(solution, v, vn, n_cex, c))
    return gen_view_str


# Known solution
lvar_symbol = "c_f"
rvar_idx = 0
known_solution = z3.And(coeffs[lvar_symbol][rvar_idx][0] == 1,
                        coeffs[lvar_symbol][rvar_idx][1] == 0,
                        coeffs[lvar_symbol][rvar_idx][2] == 0,
                        coeffs[lvar_symbol][rvar_idx][3] == -1,
                        consts[lvar_symbol] == 0)
assert isinstance(known_solution, z3.ExprRef)

if(cc.synth_ss):
    search_constraints = z3.And(search_constraints, known_solution)
    assert(isinstance(search_constraints, z3.ExprRef))

# Debugging:
debug_known_solution = None
if DEBUG:
    debug_known_solution = known_solution
    search_constraints = z3.And(search_constraints, known_solution)
    assert(isinstance(search_constraints, z3.ExprRef))

    # Definitions (including template)
    with open('tmp/definitions.txt', 'w') as f:
        assert(isinstance(definitions, z3.ExprRef))
        f.write(definitions.sexpr())

try:

    cg = CegisCCAGen(generator_vars, verifier_vars, definition_vars,
                     search_constraints, definitions, specification, ctx,
                     debug_known_solution)
    cg.get_solution_str = get_solution_str
    cg.get_counter_example_str = get_counter_example_str
    cg.get_generator_view = get_generator_view
    cg.get_verifier_view = get_verifier_view
    run_verifier = functools.partial(
        run_verifier_incomplete, c=c, v=v, ctx=ctx)
    # run_verifier = functools.partial(
    #     run_verifier_incomplete_wce, first=first, c=c, v=v, ctx=ctx)
    cg.run_verifier = run_verifier
    cg.run()

except Exception:
    import sys
    import traceback

    import ipdb
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
