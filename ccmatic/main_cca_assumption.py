import functools
import logging
from fractions import Fraction
from os import environ
from typing import List

import z3
from ccac.variables import VariableNames
from cegis import rename_vars
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic.cegis import CegisCCAGen, CegisConfig, CegisMetaData
from ccmatic.common import flatten, get_product_ite
from ccmatic.verifier import (get_cex_df, get_desired_necessary,
                              get_desired_ss_invariant, get_gen_cex_df,
                              run_verifier_incomplete, setup_cegis_basic)
from ccmatic.verifier.assumptions import get_cca_definition, get_periodic_constraints_ccac
from cegis.util import get_raw_value

logger = logging.getLogger('assumption_gen')
GlobalConfig().default_logger_setup(logger)


DEBUG = False
cc = CegisConfig()
cc.T = 10
cc.history = cc.R + cc.D
cc.infinite_buffer = True  # No loss for simplicity
cc.dynamic_buffer = False
cc.buffer_size_multiplier = 1
cc.template_queue_bound = False
cc.template_mode_switching = False
cc.template_qdel = True

cc.compose = True
cc.cca = "copa"

cc.feasible_response = True

# CCA under test
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)
vn = VariableNames(v)

periodic_constriants = get_periodic_constraints_ccac(cc, c, v)
cca_definitions = get_cca_definition(c, v)
environment = z3.And(environment, periodic_constriants, cca_definitions)
poor_utilization = v.S[-1] - v.S[0] < 0.1 * c.C * c.T

# Referenece CCA
prefix_alt = "alt"
(c_alt, s_alt, v_alt,
 ccac_domain_alt, ccac_definitions_alt, environment_alt,
 verifier_vars_alt, definition_vars_alt) = setup_cegis_basic(cc, prefix_alt)
c_alt.cca = "paced"
vn_alt = VariableNames(v_alt)

periodic_constriants_alt = get_periodic_constraints_ccac(cc, c_alt, v_alt)
cca_definitions_alt = get_cca_definition(c_alt, v_alt)
cca_definitions_alt = z3.And(cca_definitions_alt, z3.And(
    *[v_alt.c_f[n][t] == cc.template_cca_lower_bound
      for t in range(c.T) for n in range(c.N)]))
environment_alt = z3.And(
    environment_alt, periodic_constriants_alt, cca_definitions_alt)
poor_utilization_alt = v_alt.S[-1] - v_alt.S[0] < 0.1 * c_alt.C * c_alt.T

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
domain_clauses = []

vn = VariableNames(v)
ineq_var_symbols = ['S', 'A', 'W', 'C', 'mmBDP']
# Coeff for determining rhs var, of lhs var, at shift t
coeffs: List[z3.ExprRef] = [
    z3.Real('Gen__coeff_{}'.format(var)) for var in ineq_var_symbols]
consts: List[z3.ExprRef] = [
    z3.Real('Gen__const')]

# Search constr
search_range_coeff = [-1, 0, 1]
search_range_const = [x/2 for x in range(-4, 5)]
for coeff in coeffs:
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeff]))
for const in consts:
    domain_clauses.append(z3.Or(*[const == val for val in search_range_const]))
search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))

# Generator definitions
assumption_constraints = []


def get_pdt_from_sym(var_sym, t, i):
    if(var_sym == 'mmBDP'):
        return consts[0] * c.C * (c.R + c.D)
    elif(var_sym == 'C'):
        return coeffs[i] * (c.C * t + v.C0)
    else:
        return coeffs[i] * eval(f'v.{var_sym}[t]')


for t in range(c.T):
    lhs = z3.Sum(*[get_pdt_from_sym(var, t, i)
                   for i, var in enumerate(ineq_var_symbols)])
    # Basically assumption encodes when waste is allowed
    assumption_constraints.append(z3.Implies(v.W[t] > v.W[t-1], lhs <= 0))

# CCmatic inputs
ctx = z3.main_ctx()
assumption = z3.And(assumption_constraints)
assert isinstance(assumption, z3.ExprRef)
assumption_alt = rename_vars(
    assumption, verifier_vars + definition_vars, v_alt.pre + "{}")
specification = z3.Implies(
    z3.And(environment, assumption), z3.Not(poor_utilization))
specification_alt = z3.And(environment_alt, assumption_alt, poor_utilization_alt)
definitions = z3.And(ccac_domain, ccac_definitions, cca_definitions)
definitions_alt = z3.And(
    ccac_domain_alt, ccac_definitions_alt, cca_definitions_alt)
assert isinstance(definitions, z3.ExprRef)

generator_vars: List[z3.ExprRef] = coeffs + consts
critical_generator_vars: List[z3.ExprRef] = coeffs + consts

# Method overrides
# These use function closures, hence have to be defined here.
# Can use partial functions to use these elsewhere.


def get_counter_example_str(counter_example: z3.ModelRef,
                            verifier_vars: List[z3.ExprRef]) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    # for n in range(c.N):
    #     df[f"incr_{n},t"] = [
    #         get_raw_value(counter_example.eval(z3.Bool(f"incr_{n},{t}"))) for t in range(c.T)]
    #     df[f"decr_{n},t"] = [
    #         get_raw_value(counter_example.eval(z3.Bool(f"decr_{n},{t}"))) for t in range(c.T)]
    ret = "{}".format(df)
    # ret += "\nv.qdel[t][dt]\n"
    # ret += "  " + " ".join([str(i) for i in range(c.T)]) + "\n"
    # for t in range(c.T):
    #     ret += f"{t} " + " ".join([
    #         str(int(bool(counter_example.eval(v.qdel[t][dt]))))
    #         for dt in range(c.T)]) + "\n"

    return ret


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    def get_term_from_sym(sym, i):
        if(sym == 'C'):
            return f'{solution.eval(coeffs[i])}(C_0 + Ct)'
        elif(sym == 'mmBDP'):
            return f'{solution.eval(consts[0])}C(R + D)'
        else:
            return f'{solution.eval(coeffs[i])}{sym}[t]'
    ret = "W[t] > W[t-1] implies " + " + ".join([
        get_term_from_sym(sym, i) for i, sym in enumerate(ineq_var_symbols)
    ]) + " <= 0"
    df_alt = get_cex_df(solution, v_alt, vn_alt, c_alt)
    df_alt['alt_tokens_t'] = [float(solution.eval(v_alt.C0 + c.C * t).as_fraction())
                              for t in range(c.T)]
    ret += "\n{}".format(df_alt)

    return ret


def get_verifier_view(
            counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
            definition_vars: List[z3.ExprRef]) -> str:
    return get_counter_example_str(counter_example, verifier_vars)


def get_generator_view(solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                       definition_vars: List[z3.ExprRef], n_cex: int) -> str:
    df = get_gen_cex_df(solution, v, vn, n_cex, c)
    gen_view_str = "{}".format(df)
    return gen_view_str


# Known solution
known_solution = None
# assert isinstance(known_solution, z3.ExprRef)

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
    md = CegisMetaData(critical_generator_vars)
    all_generator_vars = (generator_vars + verifier_vars_alt
                          + definition_vars_alt)
    search_constraints = z3.And(search_constraints, definitions_alt,
                                specification_alt)
    assert isinstance(search_constraints, z3.ExprRef)
    cg = CegisCCAGen(generator_vars, verifier_vars,
                     definition_vars, search_constraints,
                     definitions, specification, ctx,
                     debug_known_solution, md)
    # verifier_vars_combined = verifier_vars + verifier_vars_alt
    # definition_vars_combined = definition_vars + definition_vars_alt
    # specification_combined = z3.And(specification, specification_alt)
    # definitions_combined = z3.And(definitions, definitions_alt)
    # assert isinstance(specification_combined, z3.ExprRef)
    # assert isinstance(definitions_combined, z3.ExprRef)
    # cg = CegisCCAGen(generator_vars, verifier_vars_combined,
    #                  definition_vars_combined, search_constraints,
    #                  definitions_combined, specification_combined, ctx,
    #                  debug_known_solution, md)
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
