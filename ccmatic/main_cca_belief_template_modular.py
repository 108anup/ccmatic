import argparse
import copy
import logging
import os
import sys
from typing import List, Union

import z3

import ccmatic.common  # Used for side effects
from ccac.config import ModelConfig
from ccac.variables import Variables
from ccmatic import (BeliefProofs, CCmatic, OptimizationStruct,
                     find_optimum_bounds)
from ccmatic.cegis import CegisConfig
from ccmatic.common import try_except
from ccmatic.generator import (SynthesisType, TemplateBuilder, TemplateTerm,
                               TemplateTermType, TemplateTermUnit,
                               TemplateType)
from cegis.multi_cegis import MultiCegis
from cegis.quantified_smt import ExistsForall
from cegis.util import Metric, z3_max
from pyz3_utils.common import GlobalConfig

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)

# z3.set_param('arith.solver', 2)


def get_args():

    parser = argparse.ArgumentParser(description='Belief template')

    parser.add_argument('--infinite-buffer', action='store_true')
    parser.add_argument('--finite-buffer', action='store_true')
    parser.add_argument('--dynamic-buffer', action='store_true')
    parser.add_argument('-T', action='store', type=int, default=6)
    parser.add_argument('--ideal', action='store_true')
    parser.add_argument('--app-limited', action='store_true')
    parser.add_argument('--fix-minc', action='store_true')
    parser.add_argument('--fix-maxc', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--proofs', action='store_true')
    parser.add_argument('--solution', action='store', type=int, default=None)
    parser.add_argument('--run-log-dir', action='store', default=None)
    parser.add_argument('--use-belief-invariant-n', action='store_true')
    parser.add_argument('--ideal-only', action='store_true')

    # optimizations test
    parser.add_argument('--opt-cegis-n', action='store_true')
    parser.add_argument('--opt-ve-n', action='store_true')
    parser.add_argument('--opt-pdt-n', action='store_true')
    parser.add_argument('--opt-wce-n', action='store_true')
    parser.add_argument('--opt-feasible-n', action='store_true')

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
NO_LARGE_LOSS = False
USE_CWND_CAP = False
SELF_AS_RVALUE = False

# synthesis_type = SynthesisType.CWND_ONLY
synthesis_type = SynthesisType.RATE_ONLY
template_type = TemplateType.IF_ELSE_CHAIN
# template_type = TemplateType.IF_ELSE_COMPOUND_DEPTH_1
# template_type = TemplateType.IF_ELSE_3LEAF_UNBALANCED

"""
if (cond):
    rate = choose [minc + eps, 2*minc, minc - eps, minc/2]
elif (cond):
    rate = ...
...
"""


n_exprs = 3
if(template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1):
    n_exprs = 4
elif(template_type == TemplateType.IF_ELSE_3LEAF_UNBALANCED):
    n_exprs = 3
else:
    if(args.infinite_buffer and not args.app_limited):
        n_exprs = 2
n_conds = n_exprs - 1


main_lhs_term = "r_f"
if(synthesis_type == SynthesisType.CWND_ONLY):
    main_lhs_term = "c_f"

search_range_expr_vars = (0, 1/2, 1, 2)
search_range_expr_consts = (-1, 0, 1)
expr_terms = [
    TemplateTerm('min_c', TemplateTermType.VAR,
                 TemplateTermUnit.BYTES_OR_RATE, search_range_expr_vars),
    TemplateTerm('alpha', TemplateTermType.CONST,
                 TemplateTermUnit.BYTES_OR_RATE, search_range_expr_consts)
]
if SELF_AS_RVALUE:
    expr_terms.append(
        TemplateTerm(main_lhs_term, TemplateTermType.VAR,
                     TemplateTermUnit.BYTES_OR_RATE, search_range_expr_vars))

search_range_cond_vars_time = (-1, 0, 1)
search_range_cond_vars_bytes = tuple(range(-2, 3))
search_range_cond_consts = tuple(range(-2, 3))
cond_terms = [
    TemplateTerm('min_c', TemplateTermType.VAR,
                 TemplateTermUnit.BYTES_OR_RATE, search_range_cond_vars_bytes),
    TemplateTerm('max_c', TemplateTermType.VAR,
                 TemplateTermUnit.BYTES_OR_RATE, search_range_cond_vars_bytes),
    TemplateTerm('min_qdel', TemplateTermType.VAR, TemplateTermUnit.TIME,
                 search_range_cond_vars_time),
    TemplateTerm('R', TemplateTermType.CONST,
                 TemplateTermUnit.TIME, search_range_cond_consts),
    TemplateTerm('alpha', TemplateTermType.CONST,
                 TemplateTermUnit.BYTES_OR_RATE, search_range_cond_consts),
]
if (SELF_AS_RVALUE):
    cond_terms.append(
        TemplateTerm(main_lhs_term, TemplateTermType.VAR,
                     TemplateTermUnit.BYTES_OR_RATE, search_range_cond_vars_bytes))
if (USE_MAX_QDEL):
    cond_terms.append(
        TemplateTerm('max_qdel', TemplateTermType.VAR, TemplateTermUnit.TIME,
                     search_range_cond_vars_time))


def get_value_for_term(
        tt: TemplateTerm, c: ModelConfig,
        v: Variables, n: int, t: int) -> Union[z3.ArithRef, float]:
    if (tt.name == "R"):
        return c.R
    elif (tt.name == "alpha"):
        return v.alpha
    else:
        return v.__getattribute__(tt.name)[n][t-1]


main_tb = TemplateBuilder(
        n_exprs, n_conds, template_type, expr_terms, cond_terms,
        get_value_for_term)

custom_search_constraints = []
if (len(expr_terms) == 2):
    # Limit the search space
    # Only 5 instead of 9 expressions.
    for ei in range(n_exprs):
        custom_search_constraints.append(
            z3.Or(*[
                z3.And(*[main_tb.get_expr_coeff(ei, 'min_c') == 2,
                       main_tb.get_expr_coeff(ei, 'alpha') == 0]),
                z3.And(*[main_tb.get_expr_coeff(ei, 'min_c') == 0.5,
                       main_tb.get_expr_coeff(ei, 'alpha') == 0]),
                main_tb.get_expr_coeff(ei, 'min_c') == 1,
            ]))

if (len(main_tb.expr_terms_by_type[TemplateTermType.VAR]) > 1):
    for ei in range(n_exprs):
        # Exactly one of the rhs_vars needs to be non zero:
        non_zero_list = [main_tb.get_expr_coeff(ei, et.name) != 0
                         for et in main_tb.expr_terms_by_type[TemplateTermType.VAR]]
        custom_search_constraints.append(z3.Sum(*non_zero_list) == 1)
        # If the const term is non zero, then the rhs_var must have coeff 1.
        coeff_list = [main_tb.get_expr_coeff(ei, et.name)
                      for et in main_tb.expr_terms_by_type[TemplateTermType.VAR]]
        custom_search_constraints.append(
            z3.Implies(
                main_tb.get_expr_coeff(ei, "alpha") != 0,
                z3.Sum(*coeff_list) == 1))

search_constraints = z3.And(
    *main_tb.get_search_space_constraints(),
    *main_tb.get_same_units_constraints(),
    *custom_search_constraints
)
assert(isinstance(search_constraints, z3.BoolRef))

critical_generator_vars = main_tb.get_critical_generator_vars()
generator_vars = main_tb.get_generator_vars()


def get_template_definitions(cc: CegisConfig, c: ModelConfig, v: Variables):
    template_definitions = []

    first = cc.history
    for n in range(c.N):
        for t in range(first, c.T):
            template_definitions.append(
                v.__getattribute__(main_lhs_term)[n][t]
                == z3_max(v.alpha,
                          main_tb.get_value_on_execution(cc, c, v, n, t)))

    if(synthesis_type == SynthesisType.RATE_ONLY):
        assert first >= 1
        for n in range(c.N):
            for t in range(first, c.T):
                if (USE_CWND_CAP):
                    template_definitions.append(
                        v.c_f[n][t] == 2 * v.max_c[n][t-1] * (c.R + c.D))
                else:
                    template_definitions.append(
                        v.c_f[n][t] == v.A_f[n][t-1] - v.S_f[n][t-1] + v.r_f[n][t] * 1000)

    return template_definitions


def get_solution_str(
        solution: z3.ModelRef,
        generator_vars: List[z3.ExprRef],
        n_cex: int) -> str:
    rhs_list = main_tb.get_str_on_execution(solution)
    rhs_str = "\n".join(rhs_list)
    return f"{main_lhs_term} = max alpha,\n{rhs_str}"


# ----------------------------------------------------------------
# KNOWN SOLUTIONS
# (for debugging)
known_solution = None
known_solution_list = []

# MIMD style solution
"""
min_qdel > 0:
    1/2min_c
else:
    2min_c
"""
if (n_exprs >= 2 and template_type == TemplateType.IF_ELSE_CHAIN):
    known_solution_list = [
        main_tb.get_cond_coeff(0, 'min_qdel') == 1,
        main_tb.get_expr_coeff(0, 'min_c') == 1/2,
    ]
    for ct in main_tb.cond_terms:
        if (ct.name != 'min_qdel'):
            known_solution_list.append(
                main_tb.get_cond_coeff(0, ct.name) == 0)
    for ei in range(n_exprs):
        for et in main_tb.expr_terms:
            if (et.name != 'min_c'):
                known_solution_list.append(
                    main_tb.get_expr_coeff(ei, et.name) == 0)
    known_solution_list.extend(
        [main_tb.get_cond_coeff(ci, ct.name) == 0
         for ci in range(1, n_conds)
         for ct in main_tb.cond_terms] +
        [main_tb.get_expr_coeff(ei, 'min_c') == 2 for ei in range(1, n_exprs)]
    )
mimd = z3.And(*known_solution_list)

solutions = [mimd]
known_solution = mimd
# search_constraints = z3.And(search_constraints, known_solution)
# assert isinstance(search_constraints, z3.BoolRef)

# ----------------------------------------------------------------
# ADVERSARIAL LINK
cc = CegisConfig()
# cc.DEBUG = True
cc.name = "adv"
cc.synth_ss = False
cc.infinite_buffer = args.infinite_buffer
cc.dynamic_buffer = args.dynamic_buffer
cc.buffer_size_multiplier = 1

cc.app_limited = args.app_limited
cc.app_fixed_avg_rate = True
cc.app_rate = None  # 0.5 * cc.C
cc.app_burst_factor = 1

cc.template_qdel = True
cc.template_queue_bound = False
cc.template_fi_reset = False
cc.template_beliefs = True
cc.template_beliefs_use_buffer = USE_BUFFER
cc.N = 1
cc.T = args.T
cc.history = cc.R
cc.cca = "none"

cc.rate_or_window = 'rate'
cc.use_belief_invariant = not args.use_belief_invariant_n
cc.fix_stale__min_c = args.fix_minc
cc.fix_stale__max_c = args.fix_maxc
cc.min_maxc_minc_gap_mult = (10+1)/(10-1)
cc.min_maxc_minc_gap_mult = 1
cc.maxc_minc_change_mult = 1.1

cc.desired_util_f = 0.5
cc.desired_queue_bound_multiplier = 4
cc.desired_queue_bound_alpha = 3
if(cc.infinite_buffer):
    cc.desired_loss_count_bound = 0
    cc.desired_large_loss_count_bound = 0
    cc.desired_loss_amount_bound_multiplier = 0
    cc.desired_loss_amount_bound_alpha = 0
elif(args.ideal_only):
    cc.desired_util_f = 0.6
    cc.desired_queue_bound_multiplier = 0
    cc.desired_queue_bound_alpha = 4
    cc.desired_loss_count_bound = 0
    cc.desired_large_loss_count_bound = 0
    cc.desired_loss_amount_bound_multiplier = 0
    cc.desired_loss_amount_bound_alpha = 4
else:
    cc.desired_loss_count_bound = (cc.T-1)/2
    cc.desired_large_loss_count_bound = 0   # if NO_LARGE_LOSS else (cc.T-1)/2
    # We don't expect losses in steady state. Losses only happen when beliefs
    # are changing.
    cc.desired_loss_amount_bound_multiplier = (cc.T-1)/2 - 1
    cc.desired_loss_amount_bound_alpha = (cc.T-1)/2 - 1

cc.opt_cegis = not args.opt_cegis_n
cc.opt_ve = not args.opt_ve_n
cc.opt_pdt = not args.opt_pdt_n
cc.opt_wce = not args.opt_wce_n
cc.feasible_response = not args.opt_feasible_n

cc.ideal_link = args.ideal_only
assert not (args.ideal_only and args.ideal)

link = CCmatic(cc)
try_except(link.setup_config_vars)
c, _, v = link.c, link.s, link.v
template_definitions = get_template_definitions(cc, c, v)

if(NO_LARGE_LOSS):
    # We don't want large loss even when probing for link rate.
    d = link.d
    desired = link.desired
    desired = z3.And(desired, d.bounded_large_loss_count)
    link.desired = desired

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

    bp = BeliefProofs(link, solution, args.solution)
    bp.proofs()
else:

    run_log_path = None
    if(args.run_log_dir):
        os.makedirs(args.run_log_dir, exist_ok=True)
        fname = os.path.basename(sys.argv[0])
        args_str = f"fname={fname}-"
        args_str += f"infinite_buffer={args.infinite_buffer}-"
        args_str += f"finite_buffer={args.finite_buffer}-"
        args_str += f"dynamic_buffer={args.dynamic_buffer}-"
        args_str += f"opt_cegis={not args.opt_cegis_n}-"
        args_str += f"opt_ve={not args.opt_ve_n}-"
        args_str += f"opt_pdt={not args.opt_pdt_n}-"
        args_str += f"opt_wce={not args.opt_wce_n}-"
        args_str += f"opt_feasible={not args.opt_feasible_n}-"
        args_str += f"opt_ideal={args.ideal}"
        run_log_path = os.path.join(args.run_log_dir, f'{args_str}.csv')
        logger.info(f"Run log at: {run_log_path}")

    if(ADD_IDEAL_LINK):
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
            link.run_cegis(known_solution=known_solution,
                           run_log_path=run_log_path)
        else:
            ef = ExistsForall(
                generator_vars, link.verifier_vars + link.definition_vars, search_constraints,
                z3.Implies(link.definitions,
                           link.specification), critical_generator_vars,
                get_solution_str, run_log_path=run_log_path)
            try_except(ef.run_all)