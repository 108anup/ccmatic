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
from ccmatic.common import flatten, get_product_ite

from .verifier import (get_cex_df, get_desired_necessary, get_desired_ss_invariant, get_gen_cex_df,
                       run_verifier_incomplete, setup_cegis_basic)

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


DEBUG = False
cc = CegisConfig()
cc.synth_ss = False
cc.feasible_response = True
cc.R = 1
cc.D = 1
cc.history = 4
cc.T = 9

cc.infinite_buffer = False
cc.dynamic_buffer = True
cc.buffer_size_multiplier = 1
# cc.template_queue_bound = True

cc.desired_util_f = 0.33
cc.desired_queue_bound_multiplier = 2
cc.desired_loss_count_bound = 3
cc.desired_loss_amount_bound_multiplier = 2
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)

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
    #     sv_dict['cwnd'].lo == 0.75 * c.C * (c.R + c.D),
    #     sv_dict['cwnd'].hi == (cc.history-1) * c.C * (c.R + c.D),
    #     sv_dict['queue'].lo == 0,
    #     sv_dict['queue'].hi == 1.5 * c.C * (c.R + c.D),
    # ])

    domain_clauses.extend([
        sv_dict['cwnd'].lo >= 0.5 * c.C * (c.R),
        sv_dict['cwnd'].hi <= c.T * c.C * (c.R + c.D),
        sv_dict['queue'].lo >= 0,
        sv_dict['queue'].hi <= 2 * c.C * (c.R + c.D),
    ])

vn = VariableNames(v)
coeffs = {
    'c_f[0]_loss': z3.Real('Gen__coeff_c_f[0]_loss'),
    'c_f[0]_noloss': z3.Real('Gen__coeff_c_f[0]_noloss'),
    'ack_f[0]_loss': z3.Real('Gen__coeff_ack_f[0]_loss'),
    'ack_f[0]_noloss': z3.Real('Gen__coeff_ack_f[0]_noloss')
}

consts = {
    'c_f[0]_loss': z3.Real('Gen__const_c_f[0]_loss'),
    'c_f[0]_noloss': z3.Real('Gen__const_c_f[0]_noloss')
}

# Search constr
search_range_coeff = [Fraction(i, 2) for i in range(5)]
# search_range_const = [Fraction(i, 2) for i in range(-4, 5)]
search_range_const = [-1, 0, 1]
# search_range = [-1, 0, 1]
for coeff in flatten(list(coeffs.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeff]))
for const in flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[const == val for val in search_range_const]))
search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))

# Generator definitions
template_definitions = []
first = cc.history
for t in range(first, c.T):
    loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    acked_bytes = v.S_f[0][t-c.R] - v.S_f[0][t-cc.history]
    rhs_loss = (get_product_ite(coeffs['c_f[0]_loss'], v.c_f[0][t-c.R],
                                search_range_coeff)
                + get_product_ite(coeffs['ack_f[0]_loss'], acked_bytes,
                                  search_range_coeff)
                + consts['c_f[0]_loss'])
    rhs_noloss = (get_product_ite(coeffs['c_f[0]_noloss'], v.c_f[0][t-c.R],
                                  search_range_coeff)
                  + get_product_ite(coeffs['ack_f[0]_noloss'], acked_bytes,
                                    search_range_coeff)
                  + consts['c_f[0]_noloss'])
    rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
    assert isinstance(rhs, z3.ArithRef)
    template_definitions.append(
        v.c_f[0][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                             rhs, cc.template_cca_lower_bound)
    )

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_domain, ccac_definitions, *template_definitions)
assert isinstance(definitions, z3.ExprRef)

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))
if(d.steady_state_variables):
    for sv in d.steady_state_variables:
        generator_vars.append(sv.lo)
        generator_vars.append(sv.hi)
critical_generator_vars = flatten(list(coeffs.values()))


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
    rhs_loss = (f"{solution.eval(coeffs['c_f[0]_loss'])}"
                f"v.c_f[0][t-{c.R}]"
                f" + {solution.eval(coeffs['ack_f[0]_loss'])}"
                f"(S_f[0][t-{c.R}]-S_f[0][t-{cc.history}])"
                f" + {solution.eval(consts['c_f[0]_loss'])}")
    rhs_noloss = (f"{solution.eval(coeffs['c_f[0]_noloss'])}"
                  f"v.c_f[0][t-{c.R}]"
                  f" + {solution.eval(coeffs['ack_f[0]_noloss'])}"
                  f"(S_f[0][t-{c.R}]-S_f[0][t-{cc.history}])"
                  f" + {solution.eval(consts['c_f[0]_noloss'])}")
    ret = (f"if(Ld_f[0][t] > Ld_f[0][t-1]):\n"
           f"\tc_f[0][t] = max({cc.template_cca_lower_bound}, {rhs_loss})\n"
           f"else:\n"
           f"\tc_f[0][t] = max({cc.template_cca_lower_bound}, {rhs_noloss})")

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
known_solution = None

# MD + combination
known_solution_list = [
    coeffs['c_f[0]_loss'] == 1/2,
    coeffs['ack_f[0]_loss'] == 0,
    consts['c_f[0]_loss'] == 0,

    coeffs['c_f[0]_noloss'] == 1/2,
    coeffs['ack_f[0]_noloss'] == 1/2,
    consts['c_f[0]_noloss'] == 1,
]
known_solution = z3.And(*known_solution_list)
assert(isinstance(known_solution, z3.ExprRef))

if(cc.synth_ss):
    search_constraints = z3.And(search_constraints, known_solution)
    assert(isinstance(search_constraints, z3.ExprRef))

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
