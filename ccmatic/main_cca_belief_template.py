import argparse
import copy
import logging
from typing import Dict, List

import z3

import ccmatic.common  # Used for side effects
from ccac.config import ModelConfig
from ccac.variables import Variables
from ccmatic import (BeliefProofs, CCmatic, OptimizationStruct,
                     find_optimum_bounds)
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, flatten_dict, get_product_ite, try_except
from cegis.multi_cegis import MultiCegis
from cegis.util import Metric, get_raw_value, z3_max
from pyz3_utils.common import GlobalConfig

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


def get_args():

    parser = argparse.ArgumentParser(description='Belief template')

    parser.add_argument('--infinite-buffer', action='store_true', default=False)
    parser.add_argument('--finite-buffer', action='store_true', default=False)
    parser.add_argument('--dynamic-buffer', action='store_true', default=False)
    parser.add_argument('-T', action='store', type=int, default=6)
    parser.add_argument('--ideal', action='store_true', default=False)
    parser.add_argument('--app-limited', action='store_true', default=False)
    parser.add_argument('--fix-minc', action='store_true', default=False)
    parser.add_argument('--fix-maxc', action='store_true', default=False)
    parser.add_argument('--optimize', action='store_true', default=False)
    parser.add_argument('--proofs', action='store_true', default=False)
    parser.add_argument('--solution', action='store', type=int, default=None)
    args = parser.parse_args()
    return args


args = get_args()
assert args.infinite_buffer + args.finite_buffer + args.dynamic_buffer == 1
logger.info(args)

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
# R = 1
# D = 1
# HISTORY = R
USE_T_LAST_LOSS = False
USE_MAX_QDEL = False
USE_BUFFER = False
USE_BUFFER_BYTES = False
ADD_IDEAL_LINK = args.ideal

"""
if (cond):
    rate = choose [minc + eps, 2*minc, minc - eps, minc/2]
elif (cond):
    rate = ...
...
"""

n_expr = 3
n_cond = n_expr - 1
rhs_vars = ['min_c', 'max_c']
expr_coeffs: Dict[str, List[z3.ExprRef]] = {
    rv: [z3.Real(f"Gen__coeff_expr__{rv}{i}")
         for i in range(n_expr)]
    for rv in rhs_vars
}
expr_consts = [z3.Real(f"Gen__const_expr{i}") for i in range(n_expr)]

# Cond vars with units
bytes_cvs = ['min_c', 'max_c']
if(args.app_limited):
    bytes_cvs.extend(['A_f', 'app_limits'])
time_cvs = ['min_qdel']
if(USE_MAX_QDEL):
    time_cvs.append('max_qdel')
if(USE_T_LAST_LOSS):
    time_cvs.append('t_last_loss')

if(not args.infinite_buffer and USE_BUFFER):
    time_cvs += ['min_buffer', 'max_buffer']
if(not args.infinite_buffer and USE_BUFFER_BYTES):
    bytes_cvs += ['min_buffer_bytes', 'max_buffer_bytes']

cond_vars = bytes_cvs + time_cvs
cv_to_cvi = {x: i for i, x in enumerate(cond_vars)}
logger.info(f"Using cond_vars: {cond_vars}")

cond_coeffs = [[z3.Real(f"Gen__coeff_{cv}_cond{i}")
                for cv in cond_vars] for i in range(n_cond)]
cond_consts = [z3.Real(f"Gen__const_cond{i}")
               for i in range(n_cond)]

critical_generator_vars = flatten(cond_coeffs) \
    + flatten_dict(expr_coeffs) + flatten(expr_consts)
generator_vars: List[z3.ExprRef] = critical_generator_vars \
    + flatten(cond_consts)

search_range_expr_coeffs = [1/2, 1, 3/2, 2]
search_range_expr_consts = [-1, 0, 1]
search_range_cond_coeffs_time = [-1, 0, 1]
search_range_cond_coeffs_bytes = [x/2 for x in range(-4, 5)]
search_range_cond_coeffs = list(
    set(search_range_cond_coeffs_bytes + search_range_cond_coeffs_time))
search_range_cond_consts = list(range(-10, 11))

domain_clauses = []
for expr_coeff in flatten_dict(expr_coeffs):
    domain_clauses.append(
        z3.Or(*[expr_coeff == x for x in search_range_expr_coeffs]))
for expr_const in expr_consts:
    domain_clauses.append(
        z3.Or(*[expr_const == x for x in search_range_expr_consts]))
for cond_coeff in flatten(cond_coeffs):
    if(any(x in cond_coeff.decl().name() for x in bytes_cvs)):
        domain_clauses.append(
            z3.Or(*[cond_coeff == x for x in search_range_cond_coeffs_bytes]))
    elif(any(x in cond_coeff.decl().name() for x in time_cvs)):
        domain_clauses.append(
            z3.Or(*[cond_coeff == x for x in search_range_cond_coeffs_time]))
    else:
        assert False
for cond_const in cond_consts:
    domain_clauses.append(
        z3.Or(*[cond_const == x for x in search_range_cond_consts]))

# # Limit the search space
# # Only 4 instead of 9 expressions.
# for ei in range(n_expr):
#     domain_clauses.append(
#         z3.Or(*[
#             z3.And(*[expr_coeffs['min_c'][ei] == 2, expr_consts[ei] == 0]),
#             z3.And(*[expr_coeffs['min_c'][ei] == 1/2, expr_consts[ei] == 0]),
#             expr_coeffs['min_c'][ei] == 1,
#             # z3.And(*[expr_coeffs[ei] == 1, expr_consts[ei] == 1]),
#             # z3.And(*[expr_coeffs[ei] == 1, expr_consts[ei] == -1]),
#         ]))

# Only compare qtys with the same units.
for ci in range(n_cond):
    bytes_non_zero = z3.Or(
        *[cond_coeffs[ci][cv_to_cvi[cv]] != 0 for cv in bytes_cvs])
    time_non_zero = z3.Or(
        *[cond_coeffs[ci][cv_to_cvi[cv]] != 0 for cv in time_cvs])
    domain_clauses.extend(
        [z3.Implies(bytes_non_zero, z3.Not(time_non_zero)),
         z3.Implies(time_non_zero, z3.Not(bytes_non_zero))]
    )

search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))


def get_template_definitions(
        cc: CegisConfig, c: ModelConfig, v: Variables):
    template_definitions = []

    first = cc.history
    for n in range(c.N):
        for t in range(first, c.T):

            conds = []
            exprs = []
            for ci in range(n_cond):
                cond_lhs = 0
                for cvi, cond_var_str in enumerate(cond_vars):
                    if(cond_var_str.endswith('_c')):
                        cond_var = v.__getattribute__(cond_var_str)[n][t-1] * c.R
                    elif(cond_var_str == 'app_limits'):
                        cond_var = v.app_limits[n][t]
                    elif(cond_var_str.endswith('_buffer_bytes')):
                        if(cond_var_str == 'min_buffer_bytes'):
                            cond_var = v.min_buffer[n][t-1] * v.min_c[n][t-1]
                        elif(cond_var_str == 'max_buffer_bytes'):
                            cond_var = v.max_buffer[n][t-1] * v.max_c[n][t-1]
                        else:
                            assert False
                    # elif(cond_var_str == 't_last_loss'):
                    #     for dt in range(0, t-1):
                    #         st = t-dt-1
                    else:
                        cond_var = v.__getattribute__(cond_var_str)[n][t-1]

                    # TODO: convert all vars to bytes?
                    # if(cond_var_str.endswith('_c')):
                    #     # eval(f"v.{cond_var_str}[n][t]")
                    #     cond_var = v.__getattribute__(cond_var_str)
                    # elif(cond_var_str == "min_buffer"):
                    #     cond_var = v.min_buffer * v.min_c
                    # elif(cond_var_str == "max_buffer"):
                    #     cond_var = v.max_buffer * v.max_c
                    # else:
                    #     assert False

                    cond_lhs += get_product_ite(
                        cond_coeffs[ci][cvi], cond_var,
                        search_range_cond_coeffs)
                cond_lhs += cond_consts[ci]
                conds.append(cond_lhs > 0)

            for ei in range(n_expr):
                expr = get_product_ite(
                        expr_consts[ei], v.alpha,
                        search_range_expr_consts)
                for rv in rhs_vars:
                    expr += get_product_ite(
                        expr_coeffs[rv][ei], v.__getattribute__(rv)[n][t-1],
                        search_range_expr_coeffs)
                exprs.append(expr)

            rate = exprs[-1]
            assert isinstance(rate, z3.ArithRef)
            for ci in range(n_cond-1, -1, -1):
                rate = z3.If(conds[ci], exprs[ci], rate)
                assert isinstance(rate, z3.ArithRef)

            template_definitions.append(
                v.r_f[n][t] == z3_max(rate, v.alpha))

            # Rate based CCA.
            # template_definitions.append(
            #     v.c_f[n][t] == v.A_f[n][t-1] - v.S_f[n][t-1] + v.r_f[n][t] * 1000)
            # template_definitions.append(
            #     v.c_f[n][t] == 2 * v.r_f[n][t] * c.R)
            template_definitions.append(
                v.c_f[n][t] == 2 * v.max_c[n][t-1] * (c.R + c.D))
    return template_definitions


def get_solution_str(
        solution: z3.ModelRef,
        generator_vars: List[z3.ExprRef],
        n_cex: int) -> str:
    ret = f""

    def get_cond(ci):
        cond_str = ""
        for cvi, cond_var_str in enumerate(cond_vars):
            coeff = get_raw_value(solution.eval(cond_coeffs[ci][cvi]))
            if(coeff != 0):
                cond_str += f"+ {coeff}{cond_var_str} "
        const = get_raw_value(solution.eval(cond_consts[ci]))
        if(const != 0):
            cond_str += f"+ {const} "
        if(cond_str == ""):
            cond_str = "0 "
        cond_str += "> 0"
        return cond_str

    def get_expr(ei):
        expr_str = ""
        const = get_raw_value(solution.eval(expr_consts[ei]))
        for rv in rhs_vars:
            coeff = get_raw_value(solution.eval(expr_coeffs[rv][ei]))
            if(coeff != 0):
                expr_str += f" + {coeff}{rv}"
        if(const != 0):
            expr_str += f" + {const}alpha"
        if(expr_str == ""):
            expr_str = "0"
        return expr_str

    ret += f"if ({get_cond(0)}):"
    ret += f"\n    r_f[n][t] = max(alpha, {get_expr(0)})"
    for ci in range(1, n_cond):
        ret += f"\nelif ({get_cond(ci)}):"
        ret += f"\n    r_f[n][t] = max(alpha, {get_expr(ci)})"
    ret += f"\nelse:"
    ret += f"\n    r_f[n][t] = max(alpha, {get_expr(n_expr-1)})"
    return ret


# ----------------------------------------------------------------
# KNOWN SOLUTIONS
# (for debugging)
known_solution = None

"""
if(min_qdel > 0):
    r = 1/2 min_c
else:
    r = 2 min_c
"""
known_solution_list = [
    expr_coeffs['min_c'][0] == 1/2,
    cond_coeffs[0][cv_to_cvi['min_qdel']] == 1
]
for cv in cond_vars:
    if(cv != 'min_qdel'):
        known_solution_list.append(
            cond_coeffs[0][cv_to_cvi[cv]] == 0)
known_solution_list.extend(
    [expr_coeffs['min_c'][i] == 2 for i in range(1, n_expr)] +
    [expr_consts[i] == 0 for i in range(n_expr)] +
    [cond_consts[i] == 0 for i in range(n_cond)] +
    [cond_coeffs[i][cvi] == 0 for i in range(1, n_cond)
     for cvi in range(len(cond_vars))]
)
mimd = z3.And(*known_solution_list)

"""
if(min_qdel > 0):
    r = min_c - alpha
else:
    r = min_c + alpha
"""
known_solution_list = [
    expr_coeffs['min_c'][0] == 1,
    expr_consts[0] == -1,
    cond_coeffs[0][cv_to_cvi['min_qdel']] == 1,
]
for cv in cond_vars:
    if(cv != 'min_qdel'):
        known_solution_list.append(
            cond_coeffs[0][cv_to_cvi[cv]] == 0)
known_solution_list.extend(
    [expr_coeffs['min_c'][i] == 1 for i in range(1, n_expr)] +
    [expr_consts[i] == 1 for i in range(1, n_expr)] +
    [cond_consts[i] == 0 for i in range(n_cond)] +
    [cond_coeffs[i][cvi] == 0 for i in range(1, n_cond)
     for cvi in range(len(cond_vars))]
)
aiad = z3.And(*known_solution_list)

"""
if(2 * min_c - max_c > 0):
    r = min_c
elif(min_qdel - 2 > 0):
    r = 1/2 * min_c
else:
    r = 2 * min_c
"""
known_solution_list = [
    expr_coeffs['min_c'][0] == 1,
    expr_coeffs['min_c'][1] == 1/2,
    cond_coeffs[0][cv_to_cvi['min_c']] == 2,
    cond_coeffs[0][cv_to_cvi['max_c']] == -1,
    cond_consts[0] == 0,
    cond_consts[1] == -2,
    cond_coeffs[1][cv_to_cvi['min_qdel']] == 1,
]
for cv in cond_vars:
    if(cv not in ['min_c', 'max_c']):
        known_solution_list.append(
            cond_coeffs[0][cv_to_cvi[cv]] == 0)
    if(cv not in ['min_qdel']):
        known_solution_list.append(
            cond_coeffs[1][cv_to_cvi[cv]] == 0)
known_solution_list.extend(
    [expr_coeffs['min_c'][i] == 2 for i in range(2, n_expr)] +
    [expr_consts[i] == 0 for i in range(n_expr)] +
    [cond_consts[i] == 0 for i in range(2, n_cond)] +
    [cond_coeffs[i][cvi] == 0 for i in range(2, n_cond)
     for cvi in range(len(cond_vars))]
)
blast_then_minc = z3.And(*known_solution_list)

"""
if(-2 * min_c + max_c > 0):
    r = 2 * min_c
elif(min_qdel - 2 > 0):
    r = 1/2 * min_c
else:
    r = min_c
"""
known_solution_list = [
    expr_coeffs['min_c'][0] == 2,
    expr_coeffs['min_c'][1] == 1/2,
    cond_coeffs[0][cv_to_cvi['min_c']] == -2,
    cond_coeffs[0][cv_to_cvi['max_c']] == 1,
    cond_consts[0] == 0,
    cond_consts[1] == -2,
    cond_coeffs[1][cv_to_cvi['min_qdel']] == 1,
]
for cv in cond_vars:
    if(cv not in ['min_c', 'max_c']):
        known_solution_list.append(
            cond_coeffs[0][cv_to_cvi[cv]] == 0)
    if(cv not in ['min_qdel']):
        known_solution_list.append(
            cond_coeffs[1][cv_to_cvi[cv]] == 0)
known_solution_list.extend(
    [expr_coeffs['min_c'][i] == 1 for i in range(2, n_expr)] +
    [expr_consts[i] == 0 for i in range(n_expr)] +
    [cond_consts[i] == 0 for i in range(2, n_cond)] +
    [cond_coeffs[i][cvi] == 0 for i in range(2, n_cond)
     for cvi in range(len(cond_vars))]
)
blast_then_minc_qdel = z3.And(*known_solution_list)

"""
[01/11 02:56:44]  40: if (+ -1min_c + 1/2max_c > 0):
    r_f[n][t] = max(alpha, 2min_c)
elif (+ 1/2min_c + -1/2max_c + 10 > 0):
    r_f[n][t] = max(alpha, 1min_c + -1alpha)
else:
    r_f[n][t] = max(alpha, 1min_c)
"""
known_solution_list = [
    cond_coeffs[0][cv_to_cvi['min_c']] == -2,
    cond_coeffs[0][cv_to_cvi['max_c']] == 1,
    cond_consts[0] == 0,
    expr_coeffs['min_c'][0] == 2,
    expr_consts[0] == 0,

    cond_coeffs[1][cv_to_cvi['min_c']] == 1/2,
    cond_coeffs[1][cv_to_cvi['max_c']] == -1/2,
    cond_consts[1] == 10,
    expr_coeffs['min_c'][1] == 1,
    expr_consts[1] == -1
]
for cv in cond_vars:
    if(cv not in ['min_c', 'max_c']):
        known_solution_list.append(
            cond_coeffs[0][cv_to_cvi[cv]] == 0)
        known_solution_list.append(
            cond_coeffs[1][cv_to_cvi[cv]] == 0)
known_solution_list.extend(
    [expr_coeffs['min_c'][i] == 1 for i in range(2, n_expr)] +
    [expr_consts[i] == 0 for i in range(2, n_expr)] +
    [cond_consts[i] == 0 for i in range(2, n_cond)] +
    [cond_coeffs[i][cvi] == 0 for i in range(2, n_cond)
     for cvi in range(len(cond_vars))]
)
blast_then_medblast_then_minc_negalpha = z3.And(*known_solution_list)

"""
25: if (+ 1min_qdel + -2 > 0):
    r_f[n][t] = max(alpha, 1min_c + -1alpha)
elif (+ -1min_c + 1/2max_c > 0):
    r_f[n][t] = max(alpha, 2min_c)
else:
    r_f[n][t] = max(alpha, 1min_c)

If queue large, then drain, elseif beliefs large, then probe, otherwise minc.
"""
known_solution_list = [
    cond_coeffs[0][cv_to_cvi['min_qdel']] == 1,
    cond_consts[0] == -2,
    expr_coeffs['min_c'][0] == 1,
    expr_consts[0] == -1,

    cond_coeffs[1][cv_to_cvi['min_c']] == -1,
    cond_coeffs[1][cv_to_cvi['max_c']] == 1/2,
    cond_consts[1] == 0,
    expr_coeffs['min_c'][1] == 2,
    expr_consts[1] == 0
]
for cv in cond_vars:
    if(cv not in ['min_qdel']):
        known_solution_list.append(
            cond_coeffs[0][cv_to_cvi[cv]] == 0)
    if(cv not in ['min_c', 'max_c']):
        known_solution_list.append(
            cond_coeffs[1][cv_to_cvi[cv]] == 0)
known_solution_list.extend(
    [expr_coeffs['min_c'][i] == 1 for i in range(2, n_expr)] +
    [expr_consts[i] == 0 for i in range(2, n_expr)] +
    [cond_consts[i] == 0 for i in range(2, n_cond)] +
    [cond_coeffs[i][cvi] == 0 for i in range(2, n_cond)
     for cvi in range(len(cond_vars))]
)
drain_then_blast_then_stable = z3.And(*known_solution_list)

# """
# [01/10 22:51:36]  41: if (+ -1min_c + 1/2max_c > 0):
#     r_f[n][t] = max(alpha, 2min_c)
# elif (+ -1min_qdel + -1min_buffer + 9 > 0):
#     r_f[n][t] = max(alpha, 1min_c)
# else:
#     r_f[n][t] = max(alpha, 1/2min_c)
# """
# known_solution_list = [
#     cond_coeffs[0][cv_to_cvi['min_c']] == -2,
#     cond_coeffs[0][cv_to_cvi['max_c']] == 1,
#     cond_consts[0] == 0,
#     expr_coeffs[0] == 2,

#     cond_consts[1] == 9,
#     cond_coeffs[1][cv_to_cvi['min_qdel']] == -1,
#     cond_coeffs[1][cv_to_cvi['min_buffer']] == -1,
#     expr_coeffs[1] == 1,
# ]
# for cv in cond_vars:
#     if(cv not in ['min_c', 'max_c']):
#         known_solution_list.append(
#             cond_coeffs[0][cv_to_cvi[cv]] == 0)
#     if(cv not in ['min_qdel', 'min_buffer']):
#         known_solution_list.append(
#             cond_coeffs[1][cv_to_cvi[cv]] == 0)
# known_solution_list.extend(
#     [expr_coeffs[i] == 1/2 for i in range(2, n_expr)] +
#     [expr_consts[i] == 0 for i in range(n_expr)] +
#     [cond_consts[i] == 0 for i in range(2, n_cond)] +
#     [cond_coeffs[i][cvi] == 0 for i in range(2, n_cond)
#      for cvi in range(len(cond_vars))]
# )
# synth_min_buffer = z3.And(*known_solution_list)

solutions = [mimd, aiad, blast_then_minc, blast_then_minc_qdel,
             blast_then_medblast_then_minc_negalpha,
             drain_then_blast_then_stable
             # synth_min_buffer
             ]

# known_solution = z3.And(*known_solution_list)
# search_constraints = z3.And(search_constraints, known_solution)
# assert isinstance(search_constraints, z3.BoolRef)

# ----------------------------------------------------------------
# ADVERSARIAL LINK
cc = CegisConfig()
cc.name = "adv"
cc.synth_ss = False
cc.infinite_buffer = args.infinite_buffer
cc.dynamic_buffer = args.dynamic_buffer
cc.app_limited = args.app_limited
cc.buffer_size_multiplier = 1
cc.template_qdel = True
cc.template_queue_bound = False
cc.template_fi_reset = False
cc.template_beliefs = True
cc.template_beliefs_use_buffer = USE_BUFFER
cc.N = 1
cc.T = args.T
cc.history = cc.R
cc.cca = "none"

cc.use_belief_invariant = True
cc.fix_stale__min_c = args.fix_minc
cc.fix_stale__max_c = args.fix_maxc
cc.min_maxc_minc_gap_mult = (10+1)/(10-1)
cc.maxc_minc_change_mult = 1.1

cc.desired_util_f = 0.5
cc.desired_queue_bound_multiplier = 4
cc.desired_queue_bound_alpha = 3
if(cc.infinite_buffer):
    cc.desired_loss_count_bound = 0
    cc.desired_large_loss_count_bound = 0
    cc.desired_loss_amount_bound_multiplier = 0
    cc.desired_loss_amount_bound_alpha = 0
else:
    cc.desired_loss_count_bound = 4
    cc.desired_large_loss_count_bound = 4
    cc.desired_loss_amount_bound_multiplier = 3
    cc.desired_loss_amount_bound_alpha = 3


cc.ideal_link = False
cc.feasible_response = False

link = CCmatic(cc)
try_except(link.setup_config_vars)
c, _, v = link.c, link.s, link.v
template_definitions = get_template_definitions(cc, c, v)

link.setup_cegis_loop(
    search_constraints,
    template_definitions, generator_vars, get_solution_str)
link.critical_generator_vars = critical_generator_vars
logger.info("Adver: " + cc.desire_tag())

cc_ideal = None
ideal_link = None
if(ADD_IDEAL_LINK):
    cc_ideal = copy.copy(cc)
    cc_ideal.name = "ideal"

    cc_ideal.ideal_link = True

    ideal_link = CCmatic(cc_ideal)
    try_except(ideal_link.setup_config_vars)

    c, _, v = ideal_link.c, ideal_link.s, ideal_link.v
    template_definitions = get_template_definitions(cc_ideal, c, v)

    ideal_link.setup_cegis_loop(
        search_constraints,
        template_definitions, generator_vars, get_solution_str)
    ideal_link.critical_generator_vars = critical_generator_vars
    logger.info("Ideal: " + cc_ideal.desire_tag())


if(args.optimize):
    assert args.solution is not None
    solution = solutions[args.solution]
    assert isinstance(solution, z3.BoolRef)

    # Adversarial link
    cc.reset_desired_z3(link.v.pre)
    # metric_util = [
    #     Metric(cc.desired_util_f, 0.4, 1, 0.001, True)
    # ]
    # metric_queue = [
    #     Metric(cc.desired_queue_bound_multiplier, 0, 4, 0.001, False),
    #     Metric(cc.desired_queue_bound_alpha, 0, 3, 0.001, False),
    # ]
    # metric_loss = [
    #     Metric(cc.desired_loss_count_bound, 0, 4, 0.001, False),
    #     Metric(cc.desired_large_loss_count_bound, 0, 4, 0.001, False),
    #     Metric(cc.desired_loss_amount_bound_multiplier, 0, 3, 0.001, False),
    #     Metric(cc.desired_loss_amount_bound_alpha, 0, 3, 0.001, False)
    # ]
    # optimize_metrics_list = [metric_util, metric_queue]
    # os = OptimizationStruct(link, link.get_verifier_struct(),
    #                         metric_loss, optimize_metrics_list)

    metric_alpha = [
        Metric(cc.desired_loss_amount_bound_alpha, 0, 3, 0.1, False),
        Metric(cc.desired_queue_bound_alpha, 0, 3, 0.1, False),
    ]
    metric_non_alpha = [
        Metric(cc.desired_util_f, 0.4, 1, 0.01, True),
        Metric(cc.desired_queue_bound_multiplier, 0, 4, 0.1, False),
        Metric(cc.desired_loss_count_bound, 0, 4, 0.1, False),
        Metric(cc.desired_large_loss_count_bound, 0, 4, 0.1, False),
        Metric(cc.desired_loss_amount_bound_multiplier, 0, 3, 0.1, False),
    ]
    optimize_metrics_list = [[x] for x in metric_non_alpha]
    os = OptimizationStruct(link, link.get_verifier_struct(),
                            metric_alpha, optimize_metrics_list)
    oss = [os]

    if(ADD_IDEAL_LINK):
        assert isinstance(ideal_link, CCmatic)
        cc_ideal.reset_desired_z3(ideal_link.v.pre)
        metric_alpha = [
            Metric(cc_ideal.desired_loss_amount_bound_alpha, 0, 3, 0.001, False),
            Metric(cc_ideal.desired_queue_bound_alpha, 0, 3, 0.001, False),
        ]
        metric_non_alpha = [
            Metric(cc_ideal.desired_util_f, 0.4, 1, 0.001, True),
            Metric(cc_ideal.desired_queue_bound_multiplier, 0, 4, 0.001, False),
            Metric(cc_ideal.desired_loss_count_bound, 0, 4, 0.001, False),
            Metric(cc_ideal.desired_large_loss_count_bound, 0, 4, 0.001, False),
            Metric(cc_ideal.desired_loss_amount_bound_multiplier, 0, 3, 0.001, False),
        ]
        optimize_metrics_list = [[x] for x in metric_non_alpha]
        os = OptimizationStruct(ideal_link, ideal_link.get_verifier_struct(),
                                metric_alpha, optimize_metrics_list)
        oss = [os] + oss

    model = find_optimum_bounds(solution, oss)

elif(args.proofs):
    assert args.solution is not None
    solution = solutions[args.solution]
    assert isinstance(solution, z3.BoolRef)

    bp = BeliefProofs(link, solution)
    bp.proofs()
else:
    if(ADD_IDEAL_LINK):
        assert isinstance(ideal_link, CCmatic)
        links = [ideal_link, link]
        verifier_structs = [x.get_verifier_struct() for x in links]

        multicegis = MultiCegis(
            generator_vars, search_constraints, critical_generator_vars,
            verifier_structs, link.ctx, None, None)
        multicegis.get_solution_str = get_solution_str

        try_except(multicegis.run)
    else:
        link.run_cegis()
