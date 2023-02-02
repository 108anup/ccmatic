import numpy as np
import copy
from cegis.multi_cegis import MultiCegis, VerifierStruct
from cegis.util import get_raw_value
import sys
import argparse
import functools
import logging
from typing import List

import pandas as pd
import z3
from ccac.variables import VariableNames
from pyz3_utils.common import GlobalConfig

import ccmatic.common  # Used for side effects
from ccmatic import CCmatic
from ccmatic.cegis import CegisCCAGen, CegisConfig
from ccmatic.common import flatten, substitute_values_df, try_except
from ccmatic.generator.analyse_assumptions import filter_print_assumptions, get_solution_df_from_known_solution, read_assumption_records, sort_print_assumptions
from ccmatic.verifier import (get_cex_df, get_gen_cex_df,
                              run_verifier_incomplete, setup_cegis_basic)
from ccmatic.verifier.assumptions import (get_cca_definition, get_cca_vvars,
                                          get_periodic_constraints_ccac)
from cegis import Cegis, remove_solution, rename_vars, substitute_values
from pyz3_utils.my_solver import MySolver

logger = logging.getLogger('assumption_gen')
GlobalConfig().default_logger_setup(logger)

"""
Ideally cwnd, r, etc, should be verifier variables.
There aren't really any definition vars...

Can keep cca_definitions in environment as they only involve def var or env var.
Since all def vars only depend on env vars. They are fixed...
Though, need to ensure that each def var takes exactly one value.
"""


def get_args():

    parser = argparse.ArgumentParser(description='Synthesize assumptions')

    parser.add_argument(
        '--dut', required=True,
        type=str, choices=["copa", "bbr"],
        action='store')
    parser.add_argument(
        '--ref', type=str, action='store',
        choices=["paced", "bbr", "copa"])
    parser.add_argument(
        '--util', required=True,
        type=float, action='store')

    parser.add_argument('--solution-log-path', type=str,
                        action='store', default=None)
    parser.add_argument('--solution-seed-path', type=str,
                        action='store', default=None)
    parser.add_argument('--sort-assumptions', action='store_true', default=False)
    parser.add_argument('--filter-assumptions', action='store_true', default=False)
    parser.add_argument('-o', '--outdir', type=str,
                        action='store', default="tmp")
    parser.add_argument('--suffix', type=str,
                        action='store', default="")
    # parser.add_argument('--simplify-assumptions', action='store_true', default=False)
    parser.add_argument('--use-assumption-verifier',
                        action='store_true', default=False)

    args = parser.parse_args()
    return args


args = get_args()
logger.info(args)
assert not (args.ref is not None and args.use_assumption_verifier)

DEBUG = False
cc = CegisConfig()
cc.name = "sufficient"
cc.T = 10
cc.infinite_buffer = True  # No loss for simplicity
cc.dynamic_buffer = False
cc.buffer_size_multiplier = 1
cc.template_queue_bound = False
cc.template_mode_switching = False
cc.template_qdel = True

cc.use_ref_cca = True if args.ref is not None else False
cc.monotonic_inc_assumption = False

cc.compose = True
cc.cca = args.dut
# cc.cca = "copa"
# cc.cca = "bbr"
if(cc.cca == "copa"):
    cc.history = cc.R + cc.D
elif(cc.cca == "bbr"):
    cc.history = 2 * cc.R

cc.feasible_response = False
util_frac = args.util

logger.info(f"Testing {cc.cca} comparing with {args.ref}, for utilization {util_frac}.")

# CCA under test
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)
vn = VariableNames(v)

# periodic_constriants = get_periodic_constraints_ccac(cc, c, v)
cca_definitions = get_cca_definition(c, v)
cca_vvars = get_cca_vvars(c, v)
verifier_vars.extend(flatten(cca_vvars))
# environment = z3.And(environment, periodic_constriants)
assert c.N == 1
desired = z3.Or(v.c_f[0][-1] > v.c_f[0][cc.history],
                v.S[-1] - v.S[0] >= util_frac * c.C * c.T)
poor_utilization = z3.Not(desired)

# Referenece CCA
if(cc.use_ref_cca):
    prefix_alt = "alt"
    (c_alt, s_alt, v_alt,
     ccac_domain_alt, ccac_definitions_alt, environment_alt,
     verifier_vars_alt, definition_vars_alt) = setup_cegis_basic(
        cc, prefix_alt)
    c_alt.cca = args.ref
    vn_alt = VariableNames(v_alt)

    # periodic_constriants_alt = get_periodic_constraints_ccac(cc, c_alt, v_alt)
    cca_definitions_alt = get_cca_definition(c_alt, v_alt)
    if(c_alt.cca == "paced"):
        cca_definitions_alt = z3.And(cca_definitions_alt, z3.And(
            *[v_alt.c_f[n][t] == cc.template_cca_lower_bound
              for t in range(c_alt.T) for n in range(c_alt.N)]))
    # environment_alt = z3.And(
    #     environment_alt, periodic_constriants_alt, cca_definitions_alt)
    environment_alt = z3.And(
        environment_alt, cca_definitions_alt)
    desired_alt = z3.Or(v_alt.c_f[0][-1] > v_alt.c_f[0][cc.history],
                        v_alt.S[-1] - v_alt.S[0] >=
                        util_frac * c_alt.C * c_alt.T)
    poor_utilization_alt = z3.Not(desired_alt)

# Novel trace (for monotonically increasing CCAs)
if(cc.monotonic_inc_assumption):
    prefix_novel = "novel"
    (c_novel, s_novel, v_novel,
     ccac_domain_novel, ccac_definitions_novel, environment_novel,
     verifier_vars_novel, definition_vars_novel) = setup_cegis_basic(
        cc, prefix_novel)
    vn_novel = VariableNames(v_novel)

    # periodic_constriants_novel = get_periodic_constraints_ccac(
    #     cc, c_novel, v_novel)
    cca_definitions_novel = get_cca_definition(c_novel, v_novel)
    # environment_novel = z3.And(
    #     environment_novel, periodic_constriants_novel, cca_definitions_novel)
    environment_novel = z3.And(
        environment_novel, cca_definitions_novel)
    desired_novel = z3.Or(v_novel.c_f[0][-1] > v_novel.c_f[0][cc.history],
                          v_novel.S[-1] - v_novel.S[0] >=
                          util_frac * c_novel.C * c_novel.T)
    poor_utilization_novel = z3.Not(desired_novel)

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
domain_clauses = []

vn = VariableNames(v)
ineq_var_symbols = ['S', 'A', 'L', 'W', 'C', 'mmBDP']
vname2vnum = {}
for vnum, vname in enumerate(ineq_var_symbols):
    vname2vnum[vname] = vnum
nineq = 2
nclause = 1
nshift = 2

# coeffs[ineqnum][vnum][shift]: Coeff of "vnum shifted by shift" in ineqnum.
coeffs: List[List[List[z3.ExprRef]]] = [[[
    z3.Real(f"Gen__coeff_{ineqnum}_{vnum}_{shift}")
    for shift in range(nshift)]
    for vnum in range(len(ineq_var_symbols)-1)]  # don't need coeff for mmBDP
    for ineqnum in range(nineq)]

# consts[ineqnum]: Const term in ineqnum
consts: List[z3.ExprRef] = [
    z3.Real(f'Gen__const_{ineqnum}')
    for ineqnum in range(nineq)
]

# clauses[clausenum][ineqnum] = True iff
# ineqnum appears in clausenum
clauses: List[List[z3.ExprRef]] = [[
    z3.Bool(f"Gen__clause_{clausenum}_{ineqnum}")
    for ineqnum in range(nineq)]
    for clausenum in range(nclause)]

# clausenegs[clausenum][ineqnum] = True iff
# negation of ineqnum appears in clausenum
clausenegs: List[List[z3.ExprRef]] = [[
    z3.Bool(f"Gen__clauseneg_{clausenum}_{ineqnum}")
    for ineqnum in range(nineq)]
    for clausenum in range(nclause)]

# Search constr
search_range_coeff = [-1, 0, 1]
search_range_const = [x/2 for x in range(-4, 5)]
search_range_const = [0]
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

# # Don't use A[t]
# domain_clauses.extend([coeffs[ineqnum][vname2vnum['A']][0] == 0
#                        for ineqnum in range(nineq)])

search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))


def get_assumption(c, v):
    # Generator definitions
    assumption_constraints = []

    def get_pdt_from_sym(ineqnum: int, vnum: int, shift: int, vname: str, t: int):
        if(vname == 'mmBDP'):
            if(shift == 0):
                return consts[ineqnum] * c.C * (c.R + c.D)
            else:
                return 0
        elif(vname == 'C'):
            return coeffs[ineqnum][vnum][shift] * (c.C * (t-shift) + v.C0)
        else:
            return coeffs[ineqnum][vnum][shift] * eval(f'v.{vname}[t-shift]')

    for t in range(nshift-1, c.T):
        # Truth value of ineq at time t
        evaluation_ineq: List[z3.ExprRef] = []
        for ineqnum in range(nineq):
            lhs = z3.Sum(*[
                get_pdt_from_sym(ineqnum, vnum, shift, vname, t)
                for vnum, vname in enumerate(ineq_var_symbols)
                for shift in range(nshift)])
            evaluation = lhs <= 0
            assert isinstance(evaluation, z3.ExprRef)
            evaluation_ineq.append(evaluation)

        # Truth value of clause at time t
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

        # CNF at time t
        assumption_constraints.append(z3.And(*evaluation_clause))
    assumption = z3.And(assumption_constraints)
    assert isinstance(assumption, z3.ExprRef)
    return assumption


# CCmatic inputs
ctx = z3.main_ctx()
assumption = get_assumption(c, v)
definitions = z3.And(ccac_domain, ccac_definitions, cca_definitions)
assert isinstance(definitions, z3.ExprRef)
# cca_definitions are verifier only. does not matter if they are here...
specification = z3.Implies(
    z3.And(environment, assumption), z3.Not(poor_utilization))
# # Try synth assumption that breaks CCA.
# specification = z3.Implies(
#     z3.And(environment, assumption), poor_utilization)

NO_VE = False
if(NO_VE):
    environment = z3.And(environment, definitions)
    specification = z3.Implies(
        z3.And(environment, assumption), z3.Not(poor_utilization))
    verifier_vars = verifier_vars + definition_vars
    definitions = True
    definition_vars = []

if(cc.use_ref_cca):
    assumption_alt = get_assumption(c_alt, v_alt)
    # assumption_alt = rename_vars(
    #     assumption, verifier_vars + definition_vars, v_alt.pre + "{}")
    specification_alt = z3.And(
        environment_alt, assumption_alt, poor_utilization_alt)
    definitions_alt = z3.And(
        ccac_domain_alt, ccac_definitions_alt, cca_definitions_alt)

if(cc.monotonic_inc_assumption):
    assumption_novel = get_assumption(c_novel, v_novel)
    # assumption_novel = rename_vars(
    #     assumption, verifier_vars + definition_vars, v_novel.pre + "{}")
    specification_novel = z3.And(
        environment_novel, assumption_novel, z3.Not(poor_utilization_novel))
    # # Try synth assumption that breaks CCA.
    # specification_novel = z3.And(
    #     environment_novel, assumption_novel, poor_utilization_novel)
    definitions_novel = z3.And(
        ccac_domain_novel, ccac_definitions_novel, cca_definitions_novel)


critical_generator_vars: List[z3.ExprRef] = \
    flatten(coeffs) + flatten(clauses) + flatten(clausenegs)
generator_vars: List[z3.ExprRef] = \
    critical_generator_vars + flatten(consts)

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
    ret += f"\npoor_utilization: {counter_example.eval(poor_utilization)}"
    # ret += "\nv.qdel[t][dt]\n"
    # ret += "  " + " ".join([str(i) for i in range(c.T)]) + "\n"
    # for t in range(c.T):
    #     ret += f"{t} " + " ".join([
    #         str(int(bool(counter_example.eval(v.qdel[t][dt]))))
    #         for dt in range(c.T)]) + "\n"

    return ret


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    use_model = False
    if(isinstance(solution, z3.ModelRef)):
        use_model = True

    def get_solution_val(var: z3.ExprRef):
        if(use_model):
            return solution.eval(var, model_completion=True)
        else:
            return solution[var.decl().name()]

    def get_term_from_sym(ineqnum: int, vnum: int, shift: int, vname: str):
        if(vname == 'mmBDP'):
            suffix = 'C(R + D)'
            if(shift == 0):
                val = get_solution_val(consts[ineqnum])
            else:
                val = 0
        elif(vname == 'C'):
            val = get_solution_val(coeffs[ineqnum][vnum][shift])
            suffix = f'(C_0 + C(t-{shift}))'
        else:
            val = get_solution_val(coeffs[ineqnum][vnum][shift])
            suffix = f'{vname}[t-{shift}]'

        return CCmatic.get_pretty_term(val, suffix)

    ret = ""
    for ineqnum in range(nineq):
        ret += f"Ineq {ineqnum}: "
        terms = [
            get_term_from_sym(ineqnum, vnum, shift, vname)
            for vnum, vname in enumerate(ineq_var_symbols)
            for shift in range(nshift)]
        terms = list(filter(None, terms))
        ret += " ".join(terms) + " <= 0\n"

    def bool_or_default(z3var: z3.ExprRef):
        try:
            return bool(get_solution_val(z3var))
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

    if(use_model):
        if(cc.use_ref_cca):
            df_alt = get_cex_df(solution, v_alt, vn_alt, c_alt)
            df_alt['alt_tokens_t'] = [
                float(get_solution_val(v_alt.C0 + c.C * t).as_fraction())
                for t in range(c.T)]
            ret += "\n{}".format(df_alt)

        if(cc.monotonic_inc_assumption):
            df_novel = get_cex_df(solution, v_novel, vn_novel, c_novel)
            df_novel['novel_tokens_t'] = [
                float(get_solution_val(v_novel.C0 + c.C * t).as_fraction())
                for t in range(c.T)]
            ret += "\n\n{}".format(df_novel)

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


def override_remove_solution(self: Cegis, solution: z3.ModelRef):
    # this_critical_generator_vars: List[z3.ExprRef] = \
    #     flatten(clauses) + flatten(clausenegs)

    # for ineqnum in range(nineq):
    #     ineq_appears = False
    #     for clausenum in range(nclause):
    #         pos_appears = get_raw_value(solution.eval(clauses[clausenum][ineqnum]))
    #         neg_appears = get_raw_value(solution.eval(clausenegs[clausenum][ineqnum]))
    #         # Assume don't cares are false.
    #         pos_appears = pos_appears if isinstance(pos_appears, bool) else False
    #         neg_appears = neg_appears if isinstance(neg_appears, bool) else False
    #         if(pos_appears or neg_appears):
    #             ineq_appears = True
    #             break
    #     if(ineq_appears):
    #         # We don't want new solutions that only differ in consts.
    #         this_critical_generator_vars += flatten(coeffs[ineqnum])

    # Above trick gets rid of proved solutions.
    remove_solution(self.generator, solution,
                    critical_generator_vars, self.ctx,
                    self._n_proved_solutions, model_completion=True)

    # import ipdb; ipdb.set_trace()
    # Monotonically increasing assumption set.
    if(cc.monotonic_inc_assumption):
        name_template = f"Assumption{self._n_proved_solutions}___"+"{}"
        assumption_assign = substitute_values(
            self.generator_vars, solution, name_template, ctx, model_completion=True)
        assumption_expr = rename_vars(
            assumption_novel, self.generator_vars, name_template)
        self.generator.add(assumption_assign)
        self.generator.add(z3.Not(assumption_expr))


genvar_dict = {x.decl().name(): x
               for x in generator_vars}


def process_seed_assumptions(seed_assumptions_path: str):
    # import ipdb; ipdb.set_trace()
    f = open(seed_assumptions_path, 'r')
    df = pd.read_csv(f)
    seed_assumptions = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    f.close()

    global search_constraints
    for i, seed in seed_assumptions.iterrows():
        name_template = f"SeedAssumption{i}___"+"{}"
        assumption_assign = substitute_values_df(
            seed, name_template, genvar_dict)
        assumption_expr = rename_vars(
            assumption_novel, generator_vars, name_template)
        search_constraints = z3.And(
            search_constraints,
            assumption_assign, z3.Not(assumption_expr))


# Known solution
known_solution = None
known_solution_list = []

# Waste does not happen or Q <= 0
known_solution_list = []
# Ineq 0: W[t] - W[t-1] <= 0
known_solution_list.append(coeffs[0][vname2vnum['W']][0] == 1)
known_solution_list.append(coeffs[0][vname2vnum['W']][1] == -1)
for vname in ineq_var_symbols[:-1]:
    if(vname != 'W'):
        known_solution_list.append(coeffs[0][vname2vnum[vname]][0] == 0)
        known_solution_list.append(coeffs[0][vname2vnum[vname]][1] == 0)
known_solution_list.append(clauses[0][0])
known_solution_list.append(z3.Not(clausenegs[0][0]))

# Ineq 1: Q >= 0 <=> A-L-S <= 0. Note lossless, so no mention of L
known_solution_list.append(coeffs[1][vname2vnum['A']][0] == 1)
known_solution_list.append(coeffs[1][vname2vnum['L']][0] == -1)
known_solution_list.append(coeffs[1][vname2vnum['S']][0] == -1)
for vname in ineq_var_symbols[:-1]:
    known_solution_list.append(coeffs[1][vname2vnum[vname]][1] == 0)
    if(vname not in ['A', 'S', 'L']):
        known_solution_list.append(coeffs[1][vname2vnum[vname]][0] == 0)
known_solution_list.append(clauses[0][1])
known_solution_list.append(z3.Not(clausenegs[0][1]))
ccac_paper_assumption = z3.And(*known_solution_list)
assert isinstance(ccac_paper_assumption, z3.BoolRef)
ccac_paper_assumption_record = get_solution_df_from_known_solution(
    z3.And(ccac_paper_assumption, search_constraints), critical_generator_vars)

# Never retain tokens (ideal link)
known_solution_list = []
# Ineq 0: C0 + Ct - W[t] - S[t] <= 0
known_solution_list.append(coeffs[0][vname2vnum['C']][0] == 1)
known_solution_list.append(coeffs[0][vname2vnum['W']][0] == -1)
known_solution_list.append(coeffs[0][vname2vnum['S']][0] == -1)
for vname in ineq_var_symbols[:-1]:
    if(vname not in ['C', 'W', 'S']):
        known_solution_list.append(coeffs[0][vname2vnum[vname]][0] == 0)
    known_solution_list.append(coeffs[0][vname2vnum[vname]][1] == 0)
known_solution_list.append(clauses[0][0])
known_solution_list.append(z3.Not(clausenegs[0][0]))
# Ineq 1: 0 <= 0
for vname in ineq_var_symbols[:-1]:
    known_solution_list.append(coeffs[1][vname2vnum[vname]][0] == 0)
    known_solution_list.append(coeffs[1][vname2vnum[vname]][1] == 0)
known_solution_list.append(z3.Not(clauses[0][1]))
known_solution_list.append(z3.Not(clausenegs[0][1]))
ideal_link_assumption = z3.And(*known_solution_list)

# # Waste never happens
# known_solution_list = []
# known_solution_list.append(coeffs[0][vname2vnum['W']][0] == 1)
# known_solution_list.append(coeffs[0][vname2vnum['W']][1] == -1)
# for vname in ineq_var_symbols[:-1]:
#     if(vname != 'W'):
#         known_solution_list.append(coeffs[0][vname2vnum[vname]][0] == 0)
#         known_solution_list.append(coeffs[0][vname2vnum[vname]][1] == 0)
# known_solution_list.append(clauses[0][0])
# known_solution_list.append(z3.Not(clausenegs[0][0]))
# known_solution_list.append(z3.Not(clauses[0][1]))
# known_solution_list.append(z3.Not(clausenegs[0][1]))

# Don't use A[t] in template
# known_solution_list = []
# # Ineq 0
# known_solution_list.append(coeffs[0][vname2vnum['A']][0] == 1)
# known_solution_list.append(coeffs[0][vname2vnum['']][1] == -1)
# for vname in ineq_var_symbols[:-1]:
#     if(vname != 'W'):
#         known_solution_list.append(coeffs[0][vname2vnum[vname]][0] == 0)
#         known_solution_list.append(coeffs[0][vname2vnum[vname]][1] == 0)
# known_solution_list.append(clauses[0][0])
# known_solution_list.append(z3.Not(clausenegs[0][0]))

known_solution = z3.And(known_solution_list)
# Check the False assumption
# known_solution = z3.And([z3.Not(x)
#                          for x in (flatten(clauses) + flatten(clausenegs))])
assert isinstance(known_solution, z3.ExprRef)
# search_constraints = z3.And(search_constraints, known_solution)
# assert(isinstance(search_constraints, z3.ExprRef))

lemmas = z3.And(
    # search_constraints,

    ccac_domain,
    ccac_definitions,

    environment,  # includes CCA definition and periodic if present

    # z3.Not(poor_utilization)
)
assert isinstance(lemmas, z3.ExprRef)

if (__name__ == "__main__"):

    if(args.sort_assumptions or args.filter_assumptions):
        assumption_records = read_assumption_records(args.solution_log_path)
        if(args.sort_assumptions):
            sort_print_assumptions(assumption_records, assumption, lemmas,
                                   get_solution_str, args.outdir, args.suffix)
        elif(args.filter_assumptions):
            known_assumption_record = None
            if(args.dut == "copa"):
                known_assumption_record = ccac_paper_assumption_record

            def wrap():
                filter_print_assumptions(assumption_records, assumption, lemmas,
                                         get_solution_str, args.outdir,
                                         args.suffix, known_assumption_record)
            try_except(wrap)
        else:
            assert False

        import sys
        sys.exit(0)

    if(args.solution_seed_path and cc.monotonic_inc_assumption):
        process_seed_assumptions(args.solution_seed_path)

    # Debugging:
    debug_known_solution = None
    if DEBUG:
        debug_known_solution = known_solution
        # search_constraints = z3.And(search_constraints, known_solution)
        # assert(isinstance(search_constraints, z3.ExprRef))

        # Definitions (including template)
        with open('tmp/definitions.txt', 'w') as f:
            assert(isinstance(definitions, z3.ExprRef))
            f.write(definitions.sexpr())

    # md = CegisMetaData(critical_generator_vars)

    # Ideally all these are generator vars. But since generator vars are
    # never substituted, the CEGIS loop really does not need to know all the
    # generator vars. Generator vars are really only used for identifying
    # repeat solutions. We are not going to use alt, novel vars for this
    # anyway.

    # Unused because ^^^
    # all_generator_vars = (generator_vars +
    #                       verifier_vars_alt + definition_vars_alt +
    #                       verifier_vars_novel + definition_vars_novel)
    if(cc.use_ref_cca):
        search_constraints = z3.And(search_constraints,
                                    definitions_alt, specification_alt)
    if(cc.monotonic_inc_assumption):
        search_constraints = z3.And(search_constraints,
                                    definitions_novel, specification_novel)
    assert isinstance(search_constraints, z3.ExprRef)

    # import ipbd; ipdb.set_trace()
    cg = CegisCCAGen(generator_vars, verifier_vars,
                     definition_vars, search_constraints,
                     definitions, specification, ctx,
                     debug_known_solution,
                     solution_log_path=args.solution_log_path)
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
    # https://stackoverflow.com/a/46757134/5039326.
    # Since remove_solution is a bound method of the class,
    # need to bind the method to instance!
    cg.remove_solution = override_remove_solution.__get__(cg, CegisCCAGen)
    run_verifier = functools.partial(
        run_verifier_incomplete, c=c, v=v, ctx=ctx)
    # run_verifier = functools.partial(
    #     run_verifier_incomplete_wce, first=first, c=c, v=v, ctx=ctx)
    cg.run_verifier = run_verifier

    if(not args.use_assumption_verifier):
        try_except(cg.run)
    else:
        # Verifier struct for assumption is sufficient
        vs = VerifierStruct(
            cc.name, verifier_vars, definition_vars, definitions,
            specification)
        vs.run_verifier = run_verifier
        vs.get_counter_example_str = get_counter_example_str
        vs.get_verifier_view = get_verifier_view
        vs.get_generator_view = get_generator_view

        # Assumption verifier (av)
        cc_av: CegisConfig = copy.copy(cc)
        cc_av.name = "av"
        cc_av.cca = "paced"
        cc_av.history = 3  # this should ideally not be used by anything.
        cc_av.assumption_verifier = True

        ## For backwards compatibility. This is unused. Ideally should not be needed.
        ## TODO: refactor legacy code.
        cc_av.desired_util_f = 1
        cc_av.desired_queue_bound_multiplier = 4
        cc_av.desired_loss_count_bound = 3
        cc_av.desired_loss_amount_bound_multiplier = 3

        av_link = CCmatic(cc_av)
        try_except(av_link.setup_config_vars)

        av_link.setup_cegis_loop(
            search_constraints,
            [], generator_vars, get_solution_str)
        av_link.critical_generator_vars = critical_generator_vars
        assumption = get_assumption(av_link.c, av_link.v)
        av_link.specification = z3.Implies(av_link.environment, assumption)

        # Multi-cegis
        verifier_structs = [vs, av_link.get_verifier_struct()]
        multicegis = MultiCegis(
            generator_vars, search_constraints, critical_generator_vars,
            verifier_structs, ctx, None, args.solution_log_path)
        multicegis.get_solution_str = get_solution_str
        multicegis.remove_solution = override_remove_solution.__get__(
            multicegis, MultiCegis)

        # import ipdb; ipdb.set_trace()
        try_except(multicegis.run)
