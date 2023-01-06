import argparse
import copy
import logging
from fractions import Fraction
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import z3

import ccmatic.common  # Used for side effects
from ccac.config import ModelConfig
from ccac.variables import VariableNames, Variables
from ccmatic import CCmatic, OptimizationStruct
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, flatten_dict, get_product_ite, try_except
from ccmatic.verifier import get_cex_df
from cegis.multi_cegis import MultiCegis
from cegis.util import Metric, fix_metrics, get_raw_value, optimize_multi_var, z3_max, z3_min
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
# R = 1
# D = 1
# HISTORY = R

"""
if (cond):
    rate = choose [minc + eps, 2*minc, minc - eps, minc/2]
elif (cond):
    rate = ...
...
"""


n_expr = 4
n_cond = n_expr - 1
expr_coeffs = [z3.Real(f"Gen__coeff_expr{i}") for i in range(n_expr)]
expr_consts = [z3.Real(f"Gen__const_expr{i}") for i in range(n_expr)]

cond_vars = ['min_c', 'max_c', 'min_buffer',
             'max_buffer', 'min_qdel', 'max_qdel']
cond_coeffs = [[z3.Real(f"Gen__coeff_{cv}_cond{i}")
                for cv in cond_vars] for i in range(n_cond)]

critical_generator_vars = flatten(
    cond_coeffs) + flatten(expr_coeffs) + flatten(expr_consts)
generator_vars: List[z3.ExprRef] = critical_generator_vars

search_range_expr_coeffs = [1/2, 1, 2]
search_range_expr_consts = [-1, 0, 1]
search_range_cond_coeffs = [-1, 0, 1]

domain_clauses = []
for expr_coeff in expr_coeffs:
    domain_clauses.append(
        z3.Or(*[expr_coeff == x for x in search_range_expr_coeffs]))
for expr_const in expr_consts:
    domain_clauses.append(
        z3.Or(*[expr_const == x for x in search_range_expr_consts]))
for cond_coeff in flatten(cond_coeffs):
    domain_clauses.append(
        z3.Or(*[cond_coeff == x for x in search_range_cond_coeffs]))

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
                conds.append(cond_lhs > 0)

            for ei in range(n_expr):
                expr = \
                    + get_product_ite(
                        expr_coeffs[ei], v.min_c[n][t-1],
                        search_range_expr_coeffs) \
                    + get_product_ite(
                        expr_consts[ei], v.alpha,
                        search_range_expr_consts)
                exprs.append(expr)

            rate = exprs[-1]
            assert isinstance(rate, z3.ArithRef)
            for ci in range(n_cond-1, -1, -1):
                rate = z3.If(conds[ci], exprs[ci], rate)
                assert isinstance(rate, z3.ArithRef)

            template_definitions.append(
                v.r_f[n][t] == z3_max(rate, v.alpha))

            # Rate based CCA.
            template_definitions.append(
                v.c_f[n][t] == v.A_f[n][t-1] - v.S_f[n][t-1] + v.r_f[n][t] * 1000)
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
            cond_str += f" + {coeff}{cond_var_str}"
        return cond_str

    def get_expr(ei):
        expr_str = ""
        coeff = get_raw_value(solution.eval(expr_coeffs[ei]))
        const = get_raw_value(solution.eval(expr_consts[ei]))
        return f"{coeff}min_c + {const}alpha"

    ret += f"\nif ({get_cond(0)}):"
    ret += f"\n    r_f[n][t] = max(alpha, {get_expr(0)})"
    for ci in range(1, n_cond):
        ret += f"\nelif ({get_cond(ci)}):"
        ret += f"\n    r_f[n][t] = max(alpha, {get_expr(ci)})"
    ret += f"\nelse:"
    ret += f"\n    r_f[n][t] = max(alpha, {get_expr(n_expr-1)})"
    return ret


cc = CegisConfig()
cc.name = "adv"
cc.synth_ss = False
cc.infinite_buffer = True
cc.dynamic_buffer = False
cc.buffer_size_multiplier = 1
cc.template_qdel = True
cc.template_queue_bound = True
cc.template_fi_reset = False
cc.template_beliefs = True
cc.N = 1
cc.T = 6
cc.history = cc.R
cc.cca = "none"

cc.desired_util_f = 0.5
cc.desired_queue_bound_multiplier = 4
cc.desired_queue_bound_alpha = 3
cc.desired_loss_count_bound = 0
cc.desired_large_loss_count_bound = 0
cc.desired_loss_amount_bound_multiplier = 0
cc.desired_loss_amount_bound_alpha = 0

cc.ideal_link = False
cc.feasible_response = False

link = CCmatic(cc)
link.setup_config_vars()
c, _, v = link.c, link.s, link.v
template_definitions = get_template_definitions(cc, c, v)

link.setup_cegis_loop(
    search_constraints,
    template_definitions, generator_vars, get_solution_str)

link.run_cegis()
# TODO: redine desired