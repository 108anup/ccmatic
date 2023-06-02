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
from ccmatic.cegis import CegisConfig, VerifierType
from ccmatic.common import try_except
from ccmatic.generator import (SynthesisType, TemplateBuilder, TemplateTerm,
                               TemplateTermType, TemplateTermUnit,
                               TemplateType, solution_parser)
from ccmatic.solutions.solutions_belief_template_modular import get_solutions
from ccmatic.verifier.cbr_delay import CBRDelayLink
from ccmatic.verifier.proofs import CBRDelayProofs, CCACProofs
from cegis import get_unsat_core
from cegis.multi_cegis import MultiCegis
from cegis.quantified_smt import ExistsForall
from cegis.util import Metric, z3_max
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)

# z3.set_param('arith.solver', 2)


def get_args():

    parser = argparse.ArgumentParser(description='Belief template')

    parser.add_argument('--infinite-buffer', action='store_true')
    parser.add_argument('--finite-buffer', action='store_true')
    parser.add_argument('--dynamic-buffer', action='store_true')
    parser.add_argument('--large-buffer', action='store_true')
    parser.add_argument('-T', action='store', type=int, default=6)
    parser.add_argument('--ideal', action='store_true')
    parser.add_argument('--app-limited', action='store_true')
    parser.add_argument('--fix-minc', action='store_true')
    parser.add_argument('--fix-maxc', action='store_true')
    parser.add_argument('--use-belief-invariant-n', action='store_true')
    parser.add_argument('--verifier-type', action='store',
                        default=VerifierType.ccac, type=VerifierType, choices=list(VerifierType))
    parser.add_argument('--no-large-loss', action='store_true')

    parser.add_argument('--run-log-dir', action='store', default=None)
    parser.add_argument('--solution', action='store', type=str, default=None)
    parser.add_argument('--manual-query', action='store_true')
    parser.add_argument('--cegis-with-solution', action='store_true')
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--proofs', action='store_true')

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
assert (not args.large_buffer) or args.dynamic_buffer
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
NO_LARGE_LOSS = args.no_large_loss
USE_CWND_CAP = False
SELF_AS_RVALUE = False
CONVERGENCE_BASED_ON_BUFFER = False
assert (not CONVERGENCE_BASED_ON_BUFFER) or USE_BUFFER

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
# else:
#     if(not args.dynamic_buffer and not args.app_limited):
#         n_exprs = 2
n_conds = n_exprs - 1


main_lhs_term = "r_f"
if (synthesis_type == SynthesisType.CWND_ONLY):
    main_lhs_term = "c_f"

search_range_expr_vars = (0, 1/2, 1, 2)
search_range_expr_vars_min_c_lambda = (0, 1/2, 1, 2, 3)
search_range_expr_consts = (-1, 0, 1)
expr_terms = [
    TemplateTerm('alpha', TemplateTermType.CONST,
                 TemplateTermUnit.BYTES_OR_RATE, search_range_expr_consts)
]
if (args.verifier_type == VerifierType.cbrdelay):
    expr_terms.append(TemplateTerm('min_c_lambda', TemplateTermType.VAR,
                                   TemplateTermUnit.BYTES_OR_RATE, search_range_expr_vars_min_c_lambda))
    if(not args.finite_buffer):
        expr_terms.append(TemplateTerm('min_c', TemplateTermType.VAR,
                                       TemplateTermUnit.BYTES_OR_RATE, search_range_expr_vars))
else:
    expr_terms.append(TemplateTerm('min_c', TemplateTermType.VAR,
                                   TemplateTermUnit.BYTES_OR_RATE, search_range_expr_vars))

if SELF_AS_RVALUE:
    expr_terms.append(
        TemplateTerm(main_lhs_term, TemplateTermType.VAR,
                     TemplateTermUnit.BYTES_OR_RATE, search_range_expr_vars))

search_range_cond_vars_time = (-1, 0, 1)
# search_range_cond_vars_bytes = tuple(list(range(-2, 3)) + [1.5, -1.5])
search_range_cond_vars_bytes = tuple(list(range(-2, 3)))
search_range_cond_consts = tuple(range(-6, 7))
# search_range_cond_consts = tuple(range(-2, 3))
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
if (args.verifier_type == VerifierType.cbrdelay and not args.infinite_buffer):
    cond_terms.append(
        TemplateTerm('bq_belief', TemplateTermType.VAR,
                     TemplateTermUnit.BYTES_OR_RATE, search_range_cond_consts))
if (USE_BUFFER and args.dynamic_buffer):
    cond_terms.append(TemplateTerm('min_buffer', TemplateTermType.VAR, TemplateTermUnit.TIME,
                                   search_range_cond_vars_time))
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
    elif (tt.name == "bq_belief"):
        assert isinstance(v, CBRDelayLink.LinkVariables)
        return v.bq_belief1[n][t-1]
        # return v.bq_belief2[n][t-1]
    else:
        return v.__getattribute__(tt.name)[n][t-1]


main_tb = TemplateBuilder(
        n_exprs, n_conds, template_type, expr_terms, cond_terms,
        get_value_for_term)

custom_search_constraints = []
# if (len(expr_terms) == 2):
#     # Limit the search space
#     # Only 5 instead of 9 expressions.
#     for ei in range(n_exprs):
#         custom_search_constraints.append(
#             z3.Or(*[
#                 z3.And(*[main_tb.get_expr_coeff(ei, 'min_c') == 2,
#                        main_tb.get_expr_coeff(ei, 'alpha') == 0]),
#                 z3.And(*[main_tb.get_expr_coeff(ei, 'min_c') == 0.5,
#                        main_tb.get_expr_coeff(ei, 'alpha') == 0]),
#                 main_tb.get_expr_coeff(ei, 'min_c') == 1,
#             ]))

# if (len(main_tb.expr_terms_by_type[TemplateTermType.VAR]) > 1):
#     for ei in range(n_exprs):
#         # Exactly one of the rhs_vars needs to be non zero:
#         non_zero_list = [main_tb.get_expr_coeff(ei, et.name) != 0
#                          for et in main_tb.expr_terms_by_type[TemplateTermType.VAR]]
#         custom_search_constraints.append(z3.Sum(*non_zero_list) == 1)
#         # If the const term is non zero, then the rhs_var must have coeff 1.
#         coeff_list = [main_tb.get_expr_coeff(ei, et.name)
#                       for et in main_tb.expr_terms_by_type[TemplateTermType.VAR]]
#         custom_search_constraints.append(
#             z3.Implies(
#                 main_tb.get_expr_coeff(ei, "alpha") != 0,
#                 z3.Sum(*coeff_list) == 1))

if (args.verifier_type == VerifierType.cbrdelay and not args.finite_buffer):
    # Should only use min_c or min_c_lambda (not both in RHS expr)
    for ei in range(n_exprs):
        non_zero_list = [main_tb.get_expr_coeff(ei, 'min_c') != 0,
                         main_tb.get_expr_coeff(ei, 'min_c_lambda') != 0]
        custom_search_constraints.append(z3.Sum(*non_zero_list) <= 1)

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
    elif(synthesis_type == SynthesisType.CWND_ONLY):
        for n in range(c.N):
            for t in range(first, c.T):
                template_definitions.append(
                    v.r_f[n][t] == v.c_f[n][t] / c.R)
    else:
        assert False

    return template_definitions


def get_solution_str(
        solution: z3.ModelRef,
        generator_vars: List[z3.ExprRef],
        n_cex: int) -> str:
    rhs_list = main_tb.get_str_on_execution(solution)
    rhs_str = "\n".join(rhs_list)
    return f"{main_lhs_term} = max alpha,\n{rhs_str}"


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
cc.template_beliefs_use_buffer = USE_BUFFER and args.dynamic_buffer
cc.template_beliefs_use_max_buffer = False
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

cc.desired_no_large_loss = args.no_large_loss
cc.desired_util_f = 0.5
cc.desired_queue_bound_multiplier = 4
cc.desired_queue_bound_alpha = 3
if(cc.infinite_buffer):
    cc.desired_loss_count_bound = 0
    cc.desired_large_loss_count_bound = 0
    cc.desired_loss_amount_bound_multiplier = 0
    cc.desired_loss_amount_bound_alpha = 0
# elif(args.ideal_only):
#     cc.desired_util_f = 0.6
#     cc.desired_queue_bound_multiplier = 0
#     cc.desired_queue_bound_alpha = 4
#     cc.desired_loss_count_bound = 0
#     cc.desired_large_loss_count_bound = 0
#     cc.desired_loss_amount_bound_multiplier = 0
#     cc.desired_loss_amount_bound_alpha = 4
else:
    cc.desired_loss_count_bound = (cc.T-1)/2 + 1
    cc.desired_large_loss_count_bound = 0   # if NO_LARGE_LOSS else (cc.T-1)/2
    # We don't expect losses in steady state. Losses only happen when beliefs
    # are changing.
    cc.desired_loss_amount_bound_multiplier = (cc.T-1)/2 + 1
    cc.desired_loss_amount_bound_alpha = (cc.T-1)  # (cc.T-1)/2 - 1

cc.opt_cegis = not args.opt_cegis_n
cc.opt_ve = not args.opt_ve_n
cc.opt_pdt = not args.opt_pdt_n
cc.opt_wce = not args.opt_wce_n
cc.feasible_response = not args.opt_feasible_n

cc.verifier_type = args.verifier_type
assert not (args.verifier_type == "ideal" and args.add_ideal)

cc.send_min_alpha = True

link = CCmatic(cc)
try_except(link.setup_config_vars)
c, _, v = link.c, link.s, link.v
template_definitions = get_template_definitions(cc, c, v)

# if(NO_LARGE_LOSS):
#     # We don't want large loss even when probing for link rate.
#     d = link.d
#     desired = link.desired
#     desired = z3.And(desired,
#                      z3.Or(d.bounded_large_loss_count, d.ramp_down_cwnd,
#                            d.ramp_down_queue, d.ramp_down_bq))
#     link.desired = desired

if (CONVERGENCE_BASED_ON_BUFFER):
    """
    If buffer is large and CCA knows it,
    we must have fast convergence.
    """
    buf_multiplier = 3 if args.verifier_type == VerifierType.ideal else 6
    assert c.N == 1
    d = link.d
    first = link.cc.history
    desired = link.desired
    desired = z3.And(desired,
                     z3.Implies(v.min_buffer[0][0] > buf_multiplier * (c.R + c.D),
                                z3.Implies(
                                    v.r_f[0][first] <= 0.1 * c.C, v.r_f[0][c.T-1] >= 1.1 * v.r_f[0][first])))
    link.desired = desired

if(args.large_buffer):
    link.environment = z3.And(link.environment, c.buf_min >= 3 * c.C * (c.R + c.D))

# ----------------------------------------------------------------
# KNOWN SOLUTIONS
# (for debugging)
known_solution = None
solution_dict = get_solutions(cc, main_tb, main_lhs_term)
if(args.cegis_with_solution):
    assert args.solution is not None
    assert args.solution in solution_dict
    known_solution = solution_dict[args.solution]
    search_constraints = z3.And(search_constraints, known_solution)
    assert isinstance(search_constraints, z3.BoolRef)

# ----------------------------------------------------------------
# SETUP LINK
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

# ----------------------------------------------------------------
# RUN
if(args.optimize):
    assert args.solution is not None
    solution = solution_dict[args.solution]
    assert isinstance(solution, z3.BoolRef)

    # Adversarial link
    cc.reset_desired_z3(link.v.pre)
    metric_alpha = [
        Metric(cc.desired_queue_bound_alpha, 0, 3, 0.1, False),
        Metric(cc.desired_loss_amount_bound_alpha, 0, (cc.T-1), 0.1, False),
    ]
    metric_non_alpha = [
        Metric(cc.desired_util_f, 0.4, 1, 0.01, True),
        Metric(cc.desired_queue_bound_multiplier, 0, 4, 0.1, False),
        Metric(cc.desired_loss_count_bound, 0, (cc.T-1)/2 + 1, 0.1, False),
        Metric(cc.desired_loss_amount_bound_multiplier, 0, (cc.T-1)/2 + 1, 0.1, False),
        Metric(cc.desired_large_loss_count_bound, 0, (cc.T-1)/2 + 1, 0.1, False),
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
    solution = solution_dict[args.solution]
    assert isinstance(solution, z3.BoolRef)

    ProofClass = CCACProofs \
        if args.verifier_type == VerifierType.ccac \
        else CBRDelayProofs
    bp = ProofClass(link, solution, args.solution)
    bp.proofs()

elif(args.manual_query):
    assert args.solution is not None
    solution = solution_dict[args.solution]
    assert isinstance(solution, z3.BoolRef)

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.search_constraints)
    verifier.add(solution)

    sat = verifier.check()
    if(str(sat) != "sat"):
        logger.error("Solution not in search space")
        uc = get_unsat_core(verifier)
        import ipdb; ipdb.set_trace()

    verifier.add(link.environment)
    verifier.add(link.definitions)
    # verifier.add(v.min_buffer[0][0] == 2)
    # verifier.add(v.min_c[0][0] <= 0.05 * c.C)
    # verifier.add(v.min_c[0][1] <= 0.05 * c.C)
    # # verifier.add(v.A[0] - v.L[0] - v.S[0] == 0)
    # verifier.add(v.r_f[0][1] > v.min_c[0][0])
    # verifier.add(v.alpha == 1/100)

    # assert isinstance(v, CBRDelayLink.LinkVariables)
    # verifier.add(v.alpha == 1)
    # verifier.add(v.min_c_lambda[0][0] == 5 * v.alpha)
    # verifier.add(v.min_c[0][0] == 0.8 * c.C)
    # verifier.add(v.max_c[0][0] == 1.2 * c.C)

    verifier.add(z3.Not(link.desired))

    # # Sanity check if we get traces for all scenarios. I.e., we have not
    # # inadvertently removed these intial conditions.
    # assert isinstance(v, CBRDelayLink.LinkVariables)
    # # verifier.add(z3.Not(v.initial_minc_lambda_consistent))
    # # verifier.add(z3.Not(v.initial_bq_consistent))
    # verifier.add(v.initial_minc_lambda_consistent)
    # verifier.add(v.initial_bq_consistent)

    sat = verifier.check()
    print(sat)
    if(str(sat) == "sat"):
        model = verifier.model()
        print(link.get_counter_example_str(model, link.verifier_vars))
    import ipdb; ipdb.set_trace()

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
