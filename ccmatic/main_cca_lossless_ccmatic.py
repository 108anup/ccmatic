import argparse
import copy
import logging
import os
import sys
from fractions import Fraction
from typing import List

import z3

import ccmatic.common  # Used for side effects
from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, get_product_ite_cc, try_except
from cegis.multi_cegis import MultiCegis
from cegis.quantified_smt import ExistsForall
from cegis.util import get_raw_value
from pyz3_utils.common import GlobalConfig

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)

# z3.set_param('arith.solver', 2)


def get_args():

    parser = argparse.ArgumentParser(description='Belief template')

    # parser.add_argument('--infinite-buffer', action='store_true')
    # parser.add_argument('--finite-buffer', action='store_true')
    # parser.add_argument('--dynamic-buffer', action='store_true')
    # parser.add_argument('-T', action='store', type=int, default=6)
    parser.add_argument('--ideal', action='store_true')
    # parser.add_argument('--optimize', action='store_true')
    # parser.add_argument('--proofs', action='store_true')
    # parser.add_argument('--solution', action='store', type=int, default=None)

    # optimizations test
    parser.add_argument('--opt-cegis-n', action='store_true')
    parser.add_argument('--opt-ve-n', action='store_true')
    parser.add_argument('--opt-pdt-n', action='store_true')
    parser.add_argument('--opt-wce-n', action='store_true')
    parser.add_argument('--opt-feasible-n', action='store_true')

    args = parser.parse_args()
    return args


args = get_args()
# assert args.infinite_buffer + args.finite_buffer + args.dynamic_buffer == 1
logger.info(args)

# ----------------------------------------------------------------
# TEMPLATE
# Generator search space
HISTORY = 4
rhs_var_symbols = ['S_f']
# rhs_var_symbols = ['c_f', 'S_f']
lhs_var_symbols = ['c_f']
lvar_lower_bounds = {
    'c_f': "v.alpha"
}
n_coeffs = len(rhs_var_symbols) * HISTORY
n_const = 1

# Coeff for determining rhs var, of lhs var, at shift t
coeffs = {
    lvar: [
        [z3.Real('Gen__coeff_{}_{}_{}'.format(lvar, rvar, h))
         for h in range(HISTORY)]
        for rvar in rhs_var_symbols
    ] for lvar in lhs_var_symbols
}
consts = {
    lvar: z3.Real('Gen__const_{}'.format(lvar))
    for lvar in lhs_var_symbols
}

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))
critical_generator_vars = flatten(list(coeffs.values()))

# Search constr
domain_clauses = []
search_range = [Fraction(i, 2) for i in range(-4, 5)]
# search_range = [-1, 0, 1]
for coeff in flatten(list(coeffs.values())) + flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range]))
search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))


# TODO(108anup): For multi-flow, need to add constraints for both CCAs.
def get_expr(cc, lvar_symbol, t, n) -> z3.ArithRef:
    term_list = []
    for rvar_idx in range(len(rhs_var_symbols)):
        rvar_symbol = rhs_var_symbols[rvar_idx]
        for h in range(HISTORY):
            this_coeff = coeffs[lvar_symbol][rvar_idx][h]
            time_idx = t - c.R - h
            rvar = eval(f'v.{rvar_symbol}[{n}]')
            this_term = get_product_ite_cc(
                cc, this_coeff, rvar[time_idx], search_range)
            term_list.append(this_term)
    expr = z3.Sum(*term_list) \
        + consts[lvar_symbol] * v.alpha
    # alpha is const for generator, no need to use the non-linear relaxation
    # get_product_ite_cc(cc, consts[lvar_symbol], v.alpha, search_range)
    assert isinstance(expr, z3.ArithRef)
    return expr


def get_template_definitions(cc, c, v):
    template_definitions = []
    first = HISTORY
    for n in range(c.N):
        for lvar_symbol in lhs_var_symbols:
            lower_bound = eval(lvar_lower_bounds[lvar_symbol])
            for t in range(first, c.T):
                lvar = eval(f'v.{lvar_symbol}[{n}]')
                rhs = get_expr(cc, lvar_symbol, t, n)
                template_definitions.append(
                    lvar[t] == z3.If(rhs >= lower_bound, rhs, lower_bound))
    return template_definitions


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
    rhs_expr += "+ {}alpha".format(this_const_val)
    ret = "{}[t] = max({}, {})".format(
        lvar_symbol, lvar_lower_bounds[lvar_symbol], rhs_expr)
    return ret


def desired_high_util_low_delay(c, v, first, util_frac, delay_bound):
    cond_list = []
    for t in range(first, c.T):
        cond_list.append(v.A[t] - v.L[t] - v.S[t] <= delay_bound)
    # Queue seen by a new packet should not be more that delay_bound
    low_delay = z3.And(*cond_list)
    # Serviced should be at least util_frac that could have been serviced
    high_util = v.S[-1] - v.S[first] >= util_frac * c.C * (c.T-1-first-c.D)
    # If the cwnd0 is very low then CCA should increase cwnd
    ramp_up = v.c_f[0][-1] > v.c_f[0][first]
    # If the queue is large to begin with then, CCA should cause queue to decrease.
    ramp_down = v.A[-1] - v.L[-1] - v.S[-1] < v.A[first] - v.L[first] - v.S[first]

    desired = z3.And(
        z3.Or(high_util, ramp_up),
        z3.Or(low_delay, ramp_down))
    return desired, high_util, low_delay, ramp_up, ramp_down


# ----------------------------------------------------------------
# KNOWN SOLUTIONS
# (for debugging)
known_solution = None
known_solution_list = []
# known_solution = z3.And(*known_solution_list)
# search_constraints = z3.And(search_constraints, known_solution)
# assert isinstance(search_constraints, z3.BoolRef)

# ----------------------------------------------------------------
# ADVERSARIAL LINK
cc = CegisConfig()
cc.name = "adv"
cc.synth_ss = False
cc.infinite_buffer = True  # args.infinite_buffer
cc.dynamic_buffer = False  # args.dynamic_buffer
cc.app_limited = False  # args.app_limited
cc.buffer_size_multiplier = 1
cc.template_qdel = False
cc.template_queue_bound = False
cc.template_fi_reset = False
cc.template_beliefs = False
cc.template_beliefs_use_buffer = False
cc.N = 1
cc.T = 7
cc.history = HISTORY
cc.cca = "paced"

cc.use_belief_invariant = False

cc.desired_util_f = 0.4
cc.desired_queue_bound_multiplier = 3
cc.desired_queue_bound_alpha = 4
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

cc.opt_cegis = not args.opt_cegis_n
cc.opt_ve = not args.opt_ve_n
cc.opt_pdt = not args.opt_pdt_n
cc.opt_wce = not args.opt_wce_n
cc.feasible_response = not args.opt_feasible_n
cc.ideal_link = False

link = CCmatic(cc)
try_except(link.setup_config_vars)
c, _, v = link.c, link.s, link.v
template_definitions = get_template_definitions(cc, c, v)
desired_adv = desired_high_util_low_delay(
    c, v, cc.history, cc.desired_util_f,
    cc.desired_queue_bound_multiplier * c.C * (c.R + c.D))[0]
link.desired = desired_adv

link.setup_cegis_loop(
    search_constraints,
    template_definitions, generator_vars, get_solution_str)
link.critical_generator_vars = critical_generator_vars
logger.info("Adver: " + cc.desire_tag())

cc_ideal = None
ideal_link = None
if(args.ideal):
    cc_ideal = copy.copy(cc)
    cc_ideal.name = "ideal"

    cc_ideal.ideal_link = True

    ideal_link = CCmatic(cc_ideal)
    try_except(ideal_link.setup_config_vars)

    c, _, v = ideal_link.c, ideal_link.s, ideal_link.v
    template_definitions = get_template_definitions(cc_ideal, c, v)
    desired_ideal = desired_high_util_low_delay(
        c, v, cc_ideal.history, cc_ideal.desired_util_f,
        cc_ideal.desired_queue_bound_multiplier * c.C * (c.R + c.D))[0]
    link.desired = desired_ideal

    ideal_link.setup_cegis_loop(
        search_constraints,
        template_definitions, generator_vars, get_solution_str)
    ideal_link.critical_generator_vars = critical_generator_vars
    logger.info("Ideal: " + cc_ideal.desire_tag())


fname = os.path.basename(sys.argv[0])
args_str = f"fname={fname}-"
# args_str += f"infinite_buffer={args.infinite_buffer}-"
# args_str += f"finite_buffer={args.finite_buffer}-"
# args_str += f"dynamic_buffer={args.dynamic_buffer}-"
args_str += f"opt_cegis={not args.opt_cegis_n}-"
args_str += f"opt_ve={not args.opt_ve_n}-"
args_str += f"opt_pdt={not args.opt_pdt_n}-"
args_str += f"opt_wce={not args.opt_wce_n}-"
args_str += f"opt_feasible={not args.opt_feasible_n}-"
args_str += f"opt_ideal={args.ideal}"
run_log_path = f'logs/optimizations/lossless-old_desired-T7_history4/{args_str}.csv'
logger.info(f"Run log at: {run_log_path}")

if(args.ideal):
    assert isinstance(ideal_link, CCmatic)
    links = [ideal_link, link]
    verifier_structs = [x.get_verifier_struct() for x in links]

    multicegis = MultiCegis(
        generator_vars, search_constraints, critical_generator_vars,
        verifier_structs, link.ctx, None, None, run_log_path=run_log_path)
    multicegis.get_solution_str = get_solution_str

    try_except(multicegis.run)
else:
    if(link.cc.opt_cegis):
        link.run_cegis(run_log_path=run_log_path)
    else:
        ef = ExistsForall(
            generator_vars, link.verifier_vars + link.definition_vars, search_constraints,
            z3.Implies(link.definitions,
                       link.specification), critical_generator_vars,
            get_solution_str, run_log_path=run_log_path)
        try_except(ef.run_all)
