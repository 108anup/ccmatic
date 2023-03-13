import functools
import logging
from typing import List

import z3
from ccac.variables import VariableNames
from cegis.util import unroll_assertions
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic.cegis import CegisCCAGen, CegisConfig
from ccmatic.common import flatten
from ccmatic.verifier import (get_cex_df, get_gen_cex_df,
                              run_verifier_incomplete, setup_cegis_basic)
from ccmatic import get_desired_necessary
from ccmatic.verifier.assumptions import (get_cca_definition, get_cca_vvars,
                                          get_periodic_constraints_ccac)
from cegis import remove_solution, rename_vars
from pyz3_utils.my_solver import extract_vars

logger = logging.getLogger('assumption_gen')
GlobalConfig().default_logger_setup(logger)

"""
Ideally cwnd, r, etc, should be verifier variables.
There aren't really any definition vars...

Can keep cca_definitions in environment as they only involve def var or env var.
Since all def vars only depend on env vars. They are fixed...
Though, need to ensure that each def var takes exactly one value.
"""

DEBUG = False
cc = CegisConfig()
cc.T = 10
cc.infinite_buffer = True  # No loss for simplicity
cc.dynamic_buffer = False
cc.buffer_size_multiplier = 1
cc.template_queue_bound = True
cc.template_mode_switching = False
cc.template_qdel = False

cc.desired_util_f = 0.33
cc.desired_queue_bound_multiplier = 3
cc.desired_loss_count_bound = 0
cc.desired_loss_amount_bound_multiplier = 0

cc.compose = True
cc.cca = "paced"
cc.history = cc.R
if(cc.cca == "copa"):
    cc.history = cc.R + cc.D
elif(cc.cca == "bbr"):
    cc.history = 2 * cc.R

cc.feasible_response = True

# CCA under test
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)
vn = VariableNames(v)

# AIMD on delay
cca_definitions_list = []
cca_definitions_list.append(v.qsize_thresh == 3)
first = cc.history
for t in range(first, c.T):
    delay_detected = v.exceed_queue_f[0][t]
    if(t-c.R-1 >= 0):
        this_decrease = z3.And(delay_detected,
                               v.S_f[0][t-c.R] > v.S_f[0][t-c.R-1],
                               v.S_f[0][t-c.R] > v.last_decrease_f[0][t-1])
    else:
        this_decrease = z3.And(delay_detected,
                               v.S_f[0][t-c.R] > v.last_decrease_f[0][t-1])

    rhs_delay = v.c_f[0][t-c.R]/2
    rhs_nodelay = v.c_f[0][t-c.R] + 1

    rhs = z3.If(this_decrease, rhs_delay, rhs_nodelay)
    assert isinstance(rhs, z3.ArithRef)
    cca_definitions_list.append(
        v.c_f[0][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                             rhs, cc.template_cca_lower_bound))

cca_definitions = z3.And(*cca_definitions_list)

# cca_vvars = get_cca_vvars(c, v)
# verifier_vars.extend(flatten(cca_vvars))
# periodic_constriants = get_periodic_constraints_ccac(cc, c, v)
periodic_constriants = True
environment = z3.And(environment, periodic_constriants, cca_definitions)
d = get_desired_necessary(cc, c, v)
desired = d.desired_necessary

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
ineq_terms = flatten([
    v.S[0], v.A[0], v.c_f[0][0],
    v.qbound[0, :5], v.last_decrease_f[0][0],
    c.C, c.C * (c.R + c.D)])
nineq = 3
nclause = 1

# coeffs[ineqnum][vnum]: Coeff of vnum in ineqnum.
coeffs: List[List[z3.ExprRef]] = [[
    z3.Real(f"Gen__coeff_{ineqnum}_{vnum}")
    for vnum in range(len(ineq_terms))]
    for ineqnum in range(nineq)]

# consts[ineqnum]: Const term in ineqnum
consts: List[z3.ExprRef] = [
    z3.Real(f'Gen__const_{ineqnum}')
    for ineqnum in range(nineq)
]

# True iff ineqnum appears in clausenum
clauses: List[List[z3.ExprRef]] = [[
    z3.Bool(f"Gen__clause_{clausenum}_{ineqnum}")
    for ineqnum in range(nineq)]
    for clausenum in range(nclause)]

# True iff negation of ineqnum appears in clausenum
clausenegs: List[List[z3.ExprRef]] = [[
    z3.Bool(f"Gen__clauseneg_{clausenum}_{ineqnum}")
    for ineqnum in range(nineq)]
    for clausenum in range(nclause)]

# Search constr
search_range_coeff = [-1, 0, 1]
search_range_const = [x for x in range(-3, 4)]
for coeff in flatten(coeffs):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range_coeff]))
for const in flatten(consts):
    domain_clauses.append(z3.Or(*[const == val for val in search_range_const]))

# Symmetry breaking
for clausenum in range(nclause):
    for ineqnum in range(1, nineq):

        # Ineq i+1 can appear only if ineq i appears
        # Sort literals that appear
        # Keep only (0) instead of (1), (2), ...
        domain_clauses.append(
            z3.Implies(
                z3.Or(clauses[clausenum][ineqnum],
                      clausenegs[clausenum][ineqnum]),
                z3.Or(clauses[clausenum][ineqnum-1],
                      clausenegs[clausenum][ineqnum-1])))

        # Ineq i+1 can only appear as positive literal
        # if ineq i was positive literal
        # Sort positive literals first
        # Keep only (~0 or 1) instead of (0 or ~1), ...
        domain_clauses.append(
            z3.Implies(
                clauses[clausenum][ineqnum],
                clauses[clausenum][ineqnum-1]))

        # Both variable and its negation can only happen for ineq 0.
        domain_clauses.append(z3.Not(
            z3.And(clauses[clausenum][ineqnum],
                   clausenegs[clausenum][ineqnum])))

    # If both +/- appear for ineq 0, then all other ineqs must not appear.
    domain_clauses.append(
        z3.Implies(
            z3.And(clauses[clausenum][0],
                   clausenegs[clausenum][0]),
            z3.And(*[
                z3.And(z3.Not(clauses[clausenum][ineqnum]),
                       z3.Not(clausenegs[clausenum][ineqnum]))
                for ineqnum in range(1, 0)])))

search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))

# Generator definitions
assumption_constraints = []

# Truth value of ineq
evaluation_ineq: List[z3.ExprRef] = []
for ineqnum in range(nineq):
    lhs_list = [coeffs[ineqnum][vnum] * ineq_term
                for vnum, ineq_term in enumerate(ineq_terms)] + \
        [consts[ineqnum]]
    lhs = z3.Sum(*lhs_list)
    evaluation = lhs <= 0
    assert isinstance(evaluation, z3.ExprRef)
    evaluation_ineq.append(evaluation)

# Truth value of clause
evaluation_clause: List[z3.ExprRef] = []
for clausenum in range(nclause):
    evaluation = z3.Or(
        z3.Or(*[
            z3.And(clauses[clausenum][ineqnum],
                   evaluation_ineq[ineqnum])
            for ineqnum in range(nineq)]),
        z3.Or(*[
            z3.And(clausenegs[clausenum][ineqnum],
                   z3.Not(evaluation_ineq[ineqnum]))
            for ineqnum in range(nineq)]))
    assert isinstance(evaluation, z3.ExprRef)
    evaluation_clause.append(evaluation)

assumption_constraints.append(z3.And(*evaluation_clause))

# CCmatic inputs
ctx = z3.main_ctx()
assumption = z3.And(assumption_constraints)
assert isinstance(assumption, z3.ExprRef)
assumption_alt = rename_vars(
    assumption, verifier_vars + definition_vars, v_alt.pre + "{}")
specification = z3.Implies(
    z3.And(environment, assumption), z3.Not(desired))
specification_alt = z3.And(
    environment_alt, assumption_alt, poor_utilization_alt)
definitions = z3.And(ccac_domain, ccac_definitions, cca_definitions)
definitions_alt = z3.And(
    ccac_domain_alt, ccac_definitions_alt, cca_definitions_alt)
assert isinstance(definitions, z3.ExprRef)

generator_vars: List[z3.ExprRef] = \
    flatten(coeffs) + flatten(consts) + flatten(clauses) + flatten(clausenegs)
critical_generator_vars: List[z3.ExprRef] = \
    flatten(coeffs) + flatten(consts) + flatten(clauses) + flatten(clausenegs)

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
    desired_string = d.to_string(cc, c, counter_example)
    ret = "{}\n{}".format(df, desired_string)
    # ret += "\nv.qdel[t][dt]\n"
    # ret += "  " + " ".join([str(i) for i in range(c.T)]) + "\n"
    # for t in range(c.T):
    #     ret += f"{t} " + " ".join([
    #         str(int(bool(counter_example.eval(v.qdel[t][dt]))))
    #         for dt in range(c.T)]) + "\n"

    return ret


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    def get_term_from_sym(ineqnum: int, vnum: int, term):
        val = int(solution.eval(coeffs[ineqnum][vnum]).as_fraction())
        suffix = f'{term}'

        if(val == 0):
            return None
        if(val == 1):
            return f'+ {suffix}'
        if(val == -1):
            return f'- {suffix}'
        if(val > 0):
            return f'+ {val}{suffix}'
        if(val < 0):
            return f'- {-val}{suffix}'
        else:
            return f'{val}{suffix}'

    ret = ""
    for ineqnum in range(nineq):
        ret += f"Ineq {ineqnum}: "
        terms = [
            get_term_from_sym(ineqnum, vnum, term)
            for vnum, term in enumerate(ineq_terms)] + \
            [f"+ {solution.eval(consts[ineqnum])}"]
        terms = list(filter(None, terms))
        ret += " ".join(terms) + " <= 0\n"

    def bool_or_default(z3var: z3.ExprRef):
        try:
            return bool(solution.eval(z3var))
        except z3.z3types.Z3Exception:
            return False

    for clausenum in range(nclause):
        this_clause = [
            f"~{ineqnum}" for ineqnum in range(nineq)
            if bool_or_default(clausenegs[clausenum][ineqnum])
        ] + [
            f"{ineqnum}" for ineqnum in range(nineq)
            if bool_or_default(clauses[clausenum][ineqnum])
        ]
        if(len(this_clause) == 0):
            this_clause.append("False")
        ret += f"Clause {clausenum}: " + " or ".join(this_clause) + "\n"

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
    ret = "{}".format(df)
    return ret


def override_remove_solution(self: CegisCCAGen, solution: z3.ModelRef):
    this_critical_generator_vars: List[z3.ExprRef] = \
        flatten(clauses) + flatten(clausenegs)

    for ineqnum in range(nineq):
        ineq_appears = False
        for clausenum in range(nclause):
            if(bool(solution.eval(clauses[clausenum][ineqnum])) or
               bool(solution.eval(clausenegs[clausenum][ineqnum]))):
                ineq_appears = True
                break
        if(ineq_appears):
            # We don't want new solutions that only differ in consts.
            this_critical_generator_vars += flatten(coeffs[ineqnum])

    remove_solution(self.generator, solution,
                    this_critical_generator_vars, self.ctx,
                    self._n_proved_solutions)


# Known solution
known_solution = None

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
    # md = CegisMetaData(critical_generator_vars)
    all_generator_vars = (generator_vars + verifier_vars_alt
                          + definition_vars_alt)
    search_constraints = z3.And(search_constraints, definitions_alt,
                                specification_alt)
    assert isinstance(search_constraints, z3.ExprRef)
    cg = CegisCCAGen(generator_vars, verifier_vars,
                     definition_vars, search_constraints,
                     definitions, specification, ctx,
                     debug_known_solution)
    cg.get_solution_str = get_solution_str
    cg.get_counter_example_str = get_counter_example_str
    cg.get_generator_view = get_generator_view
    cg.get_verifier_view = get_verifier_view
    # https://stackoverflow.com/a/46757134/5039326.
    # Since remove_solution is a bound method of the class,
    # need to bind the method to instance!
    cg.remove_solution = override_remove_solution.__get__(cg, CegisCCAGen)
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
