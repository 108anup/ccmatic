import z3
from ccmatic.cegis import CegisConfig, VerifierType
from ccmatic.common import try_except_wrapper

from ccmatic.generator import TemplateBuilder, TemplateType, solution_parser


def get_solutions_cbr_delay(main_tb: TemplateBuilder,
                            main_lhs_term: str):
    n_exprs = main_tb.n_exprs
    n_conds = main_tb.n_conds
    template_type = main_tb.template_type

    solution_dict = {}

    if (n_exprs >= 3 and
        template_type == TemplateType.IF_ELSE_3LEAF_UNBALANCED and
            main_lhs_term == 'r_f' and
            main_tb.get_expr_coeff(0, 'min_c') is not None):
        drain_probe_steady = solution_parser(
            """
            r_f = max alpha,
            if (+ 1max_c + -2min_c + 2alpha > 0):
                if (+ 1bq_belief + -1alpha > 0):
                    + 1alpha
                else:
                    + 3min_c_lambda + 1alpha
            else:
                + 1min_c + -1alpha
            """, main_tb)
        solution_dict['drain_probe_steady'] = drain_probe_steady

    if (n_exprs >= 2 and
        template_type == TemplateType.IF_ELSE_CHAIN and
        main_lhs_term == 'r_f' and
            main_tb.get_cond_coeff(0, 'bq_belief') is not None):
        drain_probe = solution_parser(
            """
            r_f = max alpha,
            if (+ 1bq_belief + -1alpha > 0):
                + 1alpha
            else:
                + 3min_c_lambda + 1alpha
            """, main_tb)
        solution_dict['drain_probe'] = drain_probe

    if (n_exprs >= 2 and
        template_type == TemplateType.IF_ELSE_CHAIN and
        main_lhs_term == 'r_f' and
            main_tb.get_cond_coeff(0, 'bq_belief') is not None and
            main_tb.get_expr_coeff(0, 'bq_belief') is not None):
        drain_bq_probe = solution_parser(
            """
            r_f = max alpha,
            if (+ 1bq_belief + -1alpha > 0):
                + 1min_c_lambda + -1bq_belief
            else:
                + 3min_c_lambda + 1alpha
            """, main_tb)
        solution_dict['drain_bq_probe'] = drain_bq_probe

    if (n_exprs >= 2 and
        template_type == TemplateType.IF_ELSE_CHAIN and
        main_lhs_term == 'r_f' and
            main_tb.get_cond_coeff(0, 'bq_belief') is not None):
        # This does not pass the verifier. Can perhaps only gaurantee lower
        # utilization. Because minc_lambda can be farther away from minc with
        # MI=1. CCA ends up sending at minc_lambda/2 for longer time than if
        # the CCA sent at 0.
        slow_drain_probe = solution_parser(
            """
            r_f = max alpha,
            if (+ 1bq_belief + -1alpha > 0):
                + 0.5min_c_lambda
            else:
                + 3min_c_lambda + 1alpha
            """, main_tb)
        solution_dict['slow_drain_probe'] = slow_drain_probe

    if (n_exprs >= 2 and
            template_type == TemplateType.IF_ELSE_CHAIN and
            main_lhs_term == 'r_f' and
            main_tb.get_expr_coeff(0, 'min_c') is not None):
        probe_until_shrink = solution_parser(
            """
            r_f = max alpha,
            if (+ -2min_c + 1max_c + 2alpha > 0):
                + 2min_c
            else:
                + 1min_c + -1alpha
            """, main_tb)
        solution_dict['probe_until_shrink'] = probe_until_shrink

    if (n_exprs >= 3 and
            template_type == TemplateType.IF_ELSE_CHAIN and
            main_lhs_term == 'r_f' and
            main_tb.get_expr_coeff(0, 'min_c') is not None):
        two_probes = solution_parser(
            """
            r_f = max alpha,
            if (+ 2min_c + -1.5max_c + -1alpha > 0):
                + 1min_c + -1alpha
            elif (+ -2min_c + 1max_c > 0):
                + 2min_c
            else:
                + 1min_c
            """, main_tb)
        solution_dict['two_probes'] = two_probes

    if (n_exprs >= 4 and
       template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1 and
       main_lhs_term == 'r_f' and
            main_tb.get_expr_coeff(0, 'min_c') is not None):
        buffer_based = solution_parser(
            """
            r_f = max alpha,
            if (+ -5R + 1min_buffer > 0):
                if (+ -2min_c + 2alpha + 1max_c > 0):
                    + 2min_c
                else:
                    + 1min_c + -1alpha
            else:
                if (+ 1bq_belief + -1alpha > 0):
                    + 1alpha
                else:
                    + 3min_c_lambda + 1alpha
            """, main_tb)
        solution_dict['buffer_based'] = buffer_based

        buffer_based_qdel = solution_parser(
            """
            r_f = max alpha,
            if (+ -6R + 1min_buffer > 0):
                if (+ 1min_qdel > 0):
                    + 0.5min_c
                else:
                    + 2min_c
            else:
                if (+ 1bq_belief + -1alpha > 0):
                    + 1alpha
                else:
                    + 3min_c_lambda + 1alpha
            """, main_tb)
        solution_dict['buffer_based_qdel'] = buffer_based_qdel

        """
        r_f = max alpha,
        if (+ 1min_c + 3bq_belief > 0):
            if (+ -3alpha + 3bq_belief > 0):
                + 1alpha + 1/2min_c_lambda + -1bq_belief
            else:
                + 1alpha + 3min_c_lambda + 1bq_belief
        else:
            + 1alpha + 2min_c_lambda + -1bq_belief
        """

    return solution_dict


def get_solutions_ccac(main_tb: TemplateBuilder,
                       main_lhs_term: str):
    n_exprs = main_tb.n_exprs
    n_conds = main_tb.n_conds
    template_type = main_tb.template_type

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
            [main_tb.get_expr_coeff(ei, 'min_c') ==
             2 for ei in range(1, n_exprs)]
        )
    mimd = z3.And(*known_solution_list)

    # MIMD style solution
    """
    2min_c
    """
    if (n_exprs >= 1 and template_type == TemplateType.IF_ELSE_CHAIN):
        known_solution_list = []
        for ei in range(n_exprs):
            for et in main_tb.expr_terms:
                if (et.name != 'min_c'):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'min_c') == 2 for ei in range(n_exprs)]
        )
    minc2 = z3.And(*known_solution_list)

    """
    c_f = max alpha,
    if (+ 2max_c + -1c_f > 0):
        + 1alpha + 1c_f
    else:
        + 2min_c
    """
    if (n_exprs >= 1
        and template_type == TemplateType.IF_ELSE_CHAIN
            and main_lhs_term == 'c_f'):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'max_c') == 2,
            main_tb.get_cond_coeff(0, 'c_f') == -1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['max_c', 'c_f']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'alpha') == 1,
            main_tb.get_expr_coeff(0, 'c_f') == 1,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['alpha', 'c_f']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(1, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'min_c') ==
             2 for ei in range(1, n_exprs)]
        )
        for ei in range(1, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name != 'min_c'):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    ai_probe = z3.And(*known_solution_list)

    """
    20: c_f = max alpha,
    if (+ 2min_c + -1max_c + -1alpha > 0):
        + 2min_c
    else:
        + 1alpha + 1c_f
    """
    if (n_exprs >= 1
        and template_type == TemplateType.IF_ELSE_CHAIN
            and main_lhs_term == 'c_f'):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'min_c') == 2,
            main_tb.get_cond_coeff(0, 'max_c') == -1,
            main_tb.get_cond_coeff(0, 'alpha') == -1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['max_c', 'min_c', 'alpha']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'min_c') == 2,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(1, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'alpha') == 1 for ei in range(1, n_exprs)] +
            [main_tb.get_expr_coeff(ei, 'c_f') ==
             1 for ei in range(1, n_exprs)]
        )
        for ei in range(1, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name not in ['alpha', 'c_f']):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    ai_until_shrink = z3.And(*known_solution_list)

    """
    r_f = max alpha,
    if (+ -1min_qdel + 1R > 0):
        + 2min_c
    else:
        + 1min_c + -1alpha
    """
    if (n_exprs >= 1
        and template_type == TemplateType.IF_ELSE_CHAIN
            and main_lhs_term == 'r_f'):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'min_qdel') == -1,
            main_tb.get_cond_coeff(0, 'R') == 1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['min_qdel', 'R']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'min_c') == 2,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(1, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'min_c') == 1 for ei in range(1, n_exprs)] +
            [main_tb.get_expr_coeff(ei, 'alpha') == -
             1 for ei in range(1, n_exprs)]
        )
        for ei in range(1, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name not in ['min_c', 'alpha']):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    ideal_fast = z3.And(*known_solution_list)

    """
    r_f = max alpha,
    if (+ -1min_qdel + 1R > 0):
        + 1min_c + 1alpha
    else:
        + 1min_c + -1alpha
    """
    if (n_exprs >= 1 and
        template_type == TemplateType.IF_ELSE_CHAIN and
            main_lhs_term == 'r_f'):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'min_qdel') == -1,
            main_tb.get_cond_coeff(0, 'R') == 1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['min_qdel', 'R']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'min_c') == 1,
            main_tb.get_expr_coeff(0, 'alpha') == 1,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c', 'alpha']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(1, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'min_c') == 1 for ei in range(1, n_exprs)] +
            [main_tb.get_expr_coeff(ei, 'alpha') == -
             1 for ei in range(1, n_exprs)]
        )
        for ei in range(1, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name not in ['alpha', 'min_c']):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    ideal_slow = z3.And(*known_solution_list)

    """
    r_f = max alpha,
    if (+ 1max_c + -2alpha + -1r_f > 0):
        + 1alpha + 1r_f
    else:
        + 1min_c + -1alpha
    """
    if (n_exprs >= 1 and
        template_type == TemplateType.IF_ELSE_CHAIN and
            main_tb.get_cond_coeff(0, 'r_f') is not None and
            main_lhs_term == 'r_f'):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'max_c') == 1,
            main_tb.get_cond_coeff(0, 'alpha') == -2,
            main_tb.get_cond_coeff(0, 'r_f') == -1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['max_c', 'alpha', 'r_f']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'alpha') == 1,
            main_tb.get_expr_coeff(0, 'r_f') == 1,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['alpha', 'r_f']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(1, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'min_c') == 1 for ei in range(1, n_exprs)] +
            [main_tb.get_expr_coeff(ei, 'alpha') == -
             1 for ei in range(1, n_exprs)]
        )
        for ei in range(1, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name not in ['alpha', 'min_c']):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    rate_ai_probe = z3.And(*known_solution_list)

    """
    r_f = max alpha,
    if (+ -1min_c + 1alpha + 2r_f > 0):
        if (+ 2min_c + 2alpha + -2r_f > 0):
            + 1min_c + 1alpha
        else:
            + 1/2r_f
    else:
        + 1alpha + 1r_f
    """
    if (n_exprs >= 3 and
        template_type == TemplateType.IF_ELSE_3LEAF_UNBALANCED and
            main_lhs_term == 'r_f'):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'min_c') == -1,
            main_tb.get_cond_coeff(0, 'alpha') == 1,
            main_tb.get_cond_coeff(0, 'r_f') == 2,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['min_c', 'alpha', 'r_f']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'min_c') == 1,
            main_tb.get_expr_coeff(0, 'alpha') == 1,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c', 'alpha']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        # Cond 1
        known_solution_list.extend([
            main_tb.get_cond_coeff(1, 'min_c') == 2,
            main_tb.get_cond_coeff(1, 'alpha') == 2,
            main_tb.get_cond_coeff(1, 'r_f') == -2,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['min_c', 'alpha', 'r_f']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(1, ct.name) == 0)
        # Expr 1
        known_solution_list.extend([
            main_tb.get_expr_coeff(1, 'r_f') == 1/2,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['r_f']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(1, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(2, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'alpha') == 1 for ei in range(2, n_exprs)] +
            [main_tb.get_expr_coeff(ei, 'r_f') == 1 for ei in range(2, n_exprs)]
        )
        for ei in range(2, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name not in ['alpha', 'r_f']):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    convergence_based_on_buffer_first_try = z3.And(known_solution_list)

    """
    r_f = max alpha,
    if (+ -2min_c + 2alpha + 1max_c > 0):
        if (+ 1min_buffer + -3R > 0):
            + 2min_c
        else:
            + 1min_c + 1alpha
    else:
        + 1min_c + -1alpha
    """
    if (n_exprs >= 3 and
        template_type == TemplateType.IF_ELSE_3LEAF_UNBALANCED and
            main_lhs_term == 'r_f' and
            main_tb.get_cond_coeff(0, 'min_buffer') is not None):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'min_c') == -2,
            main_tb.get_cond_coeff(0, 'alpha') == 2,
            main_tb.get_cond_coeff(0, 'max_c') == 1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['min_c', 'alpha', 'max_c']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'min_c') == 2,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        # Cond 1
        known_solution_list.extend([
            main_tb.get_cond_coeff(1, 'min_buffer') == 1,
            main_tb.get_cond_coeff(1, 'R') == -3,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['min_buffer', 'R']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(1, ct.name) == 0)
        # Expr 1
        known_solution_list.extend([
            main_tb.get_expr_coeff(1, 'min_c') == 1,
            main_tb.get_expr_coeff(1, 'alpha') == 1,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c', 'alpha']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(1, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(2, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'min_c') == 1 for ei in range(2, n_exprs)] +
            [main_tb.get_expr_coeff(ei, 'alpha') == -1 for ei in range(2, n_exprs)]
        )
        for ei in range(2, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name not in ['min_c', 'alpha']):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    convergence_based_on_buffer_manual = z3.And(known_solution_list)

    """
    r_f = max alpha,
    if (+ 1max_c + -2alpha + -1r_f > 0):
        if (+ -1min_qdel + 1R > 0):
            + 1min_c + 1alpha
        else:
            + 1/2min_c
    else:
        + 1/2min_c
    """
    if (n_exprs >= 3 and
        template_type == TemplateType.IF_ELSE_3LEAF_UNBALANCED and
            main_lhs_term == 'r_f' and
            main_tb.get_cond_coeff(0, 'min_buffer') is not None):
        known_solution_list = []
        # Cond 0
        known_solution_list.extend([
            main_tb.get_cond_coeff(0, 'max_c') == 1,
            main_tb.get_cond_coeff(0, 'alpha') == -2,
            main_tb.get_cond_coeff(0, 'r_f') == -1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['max_c', 'alpha', 'r_f']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(0, ct.name) == 0)
        # Expr 0
        known_solution_list.extend([
            main_tb.get_expr_coeff(0, 'min_c') == 1,
            main_tb.get_expr_coeff(0, 'alpha') == 1,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c', 'alpha']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(0, et.name) == 0)
        # Cond 1
        known_solution_list.extend([
            main_tb.get_cond_coeff(1, 'min_qdel') == -1,
            main_tb.get_cond_coeff(1, 'R') == 1,
        ])
        for ct in main_tb.cond_terms:
            if (ct.name not in ['min_qdel', 'R']):
                known_solution_list.append(
                    main_tb.get_cond_coeff(1, ct.name) == 0)
        # Expr 1
        known_solution_list.extend([
            main_tb.get_expr_coeff(1, 'min_c') == 1/2,
        ])
        for et in main_tb.expr_terms:
            if (et.name not in ['min_c']):
                known_solution_list.append(
                    main_tb.get_expr_coeff(1, et.name) == 0)
        known_solution_list.extend(
            [main_tb.get_cond_coeff(ci, ct.name) == 0
             for ci in range(2, n_conds)
             for ct in main_tb.cond_terms] +
            [main_tb.get_expr_coeff(ei, 'min_c') == 1/2 for ei in range(2, n_exprs)]
        )
        for ei in range(2, n_exprs):
            for et in main_tb.expr_terms:
                if (et.name not in ['min_c']):
                    known_solution_list.append(
                        main_tb.get_expr_coeff(ei, et.name) == 0)
    convergence_based_on_buffer_second_try = z3.And(known_solution_list)

    """
    TODO: Understand why these work.
    Ideal link, finite buffer. No large loss.

    r_f = max alpha,
    if (+ -2min_c + -2alpha + 2r_f > 0):
        if (+ -2min_c + -1max_c > 0):
            + 1alpha + 1r_f
        else:
            + 1min_c + -1alpha
    else:
        + 1alpha + 1r_f

    139: r_f = max alpha,
    if (+ 1min_c + -1max_c + -2alpha + 1r_f > 0):
        + 1min_c + -1alpha
    else:
        + 1alpha + 1r_f
    """

    solution_dict = {
        'mimd': mimd,
        'minc2': minc2,
        'ai_probe': ai_probe,
        'ai_until_shrink': ai_until_shrink,
        'ideal_fast': ideal_fast,
        'ideal_slow': ideal_slow,
        'rate_ai_probe': rate_ai_probe,
        'convergence_based_on_buffer_first_try': convergence_based_on_buffer_first_try,
        'convergence_based_on_buffer_manual': convergence_based_on_buffer_manual,
        'convergence_based_on_buffer_second_try': convergence_based_on_buffer_second_try,
    }

    if (n_exprs >= 3 and
            template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1 and
            main_lhs_term == 'r_f' and
            main_tb.get_cond_coeff(0, 'r_f') is not None):
        rate_ai_probe_2minc = solution_parser(
            """
            r_f = max alpha,
            if (+ 2min_c + -1max_c > 0):
                if (+ 1min_qdel + -3R > 0):
                    + 1min_c + -1alpha
                else:
                    + 1min_c
            else:
                if (+ 2min_c + -1r_f > 0):
                    + 1r_f + 1alpha
                else:
                    + 2min_c
            """, main_tb)
        solution_dict['rate_ai_probe_2minc'] = rate_ai_probe_2minc

    if (n_exprs >= 3 and
            template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1 and
            main_lhs_term == 'r_f' and
            main_tb.get_cond_coeff(0, 'r_f') is not None):
        rate_probe_2minc = solution_parser(
            """
            r_f = max alpha,
            if (+ 2min_c + -1max_c > 0):
                if (+ 1min_qdel + -3R > 0):
                    + 1min_c + -1alpha
                else:
                    + 1min_c
            else:
                if (+ 2min_c + -1r_f > 0):
                    + 2min_c
                else:
                    + 2min_c
            """, main_tb)
        solution_dict['rate_probe_2minc'] = rate_probe_2minc

    # This solution requires T=11.
    if (n_exprs >= 2 and
            template_type == TemplateType.IF_ELSE_CHAIN and
            main_lhs_term == 'r_f'):
        probe_until_shrink = solution_parser(
            """
            r_f = max alpha,
            if (+ -2min_c + 1max_c + 2alpha > 0):
                + 2min_c
            else:
                + 1min_c + -1alpha
            """, main_tb)
        solution_dict['probe_until_shrink'] = probe_until_shrink

    # This works with T=6
    if (n_exprs >= 2 and
            template_type == TemplateType.IF_ELSE_CHAIN and
            main_lhs_term == 'r_f'):
        probe_qdel = solution_parser(
            """
            r_f = max alpha,
            if (+ 1min_qdel > 0):
                + 0.5min_c
            else:
                + 2min_c
            """, main_tb)
        solution_dict['probe_qdel'] = probe_qdel

    # I don't htink this solution works ever. On a high queue, if qdel is low,
    # then queue increases. I twill take a lot of time to drain this.
    if (n_exprs >= 2 and
            template_type == TemplateType.IF_ELSE_CHAIN and
            main_lhs_term == 'r_f'):
        probe_qdel_paced = solution_parser(
            """
            r_f = max alpha,
            if (+ 1min_qdel > 0):
                + 1min_c + -1alpha
            else:
                + 2min_c
            """, main_tb)
        solution_dict['probe_qdel_paced'] = probe_qdel_paced

    if (n_exprs >= 3 and
        template_type == TemplateType.IF_ELSE_CHAIN and
            main_lhs_term == 'r_f'):
        probe_until_shrink_3expr = solution_parser(
            """
            r_f = max alpha,
            if (+ -2min_c + 1max_c > 0):
                + 2min_c
            elif (-2min_c + 1.5max_c + 1alpha):
                + 1min_c
            else:
                + 1min_c + -1alpha
            """, main_tb)
        solution_dict['probe_until_shrink_3expr'] = probe_until_shrink_3expr

    if (n_exprs >= 4 and
        template_type == TemplateType.IF_ELSE_COMPOUND_DEPTH_1 and
            main_lhs_term == 'r_f'):
        probe_until_shrink_qdel = solution_parser(
            """
            r_f = max alpha,
            if (+ -2min_c + 1max_c > 0):
                if (+ 1min_qdel + -1R > 0):
                    + 0.5min_c
                else:
                    + 2min_c
            else:
                if (+ -2min_c + 1max_c + 2alpha > 0):
                    + 1min_c
                else:
                    + 1min_c + -1alpha
            """, main_tb)
        solution_dict['probe_until_shrink_qdel'] = probe_until_shrink_qdel

    return solution_dict


def get_solutions(cc: CegisConfig, main_tb: TemplateBuilder,
                  main_lhs_term: str):
    if(cc.verifier_type == VerifierType.cbrdelay):
        return get_solutions_cbr_delay(main_tb, main_lhs_term)
    elif(cc.verifier_type == VerifierType.ccac):
        return get_solutions_ccac(main_tb, main_lhs_term)
    else:
        return get_solutions_ccac(main_tb, main_lhs_term)
