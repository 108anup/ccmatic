import matplotlib.pyplot as plt
import time
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, List, Tuple, Union

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import z3
from cegis import rename_vars
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

logger = logging.getLogger('analyse_assumptions')
GlobalConfig().default_logger_setup(logger)


# def extract_vars(e: z3.ExprRef) -> Set[z3.ExprRef]:
#     if(z3.is_app(e)):
#         if(len(e.children()) == 0):
#             return set([e])
#         else:
#             union = set()
#             for child in e.children():
#                 union = union.union(extract_vars(child))
#             return union
#     return set()


def extract_vars(e: z3.ExprRef) -> List[z3.ExprRef]:
    if e.children() == []:
        if str(e)[:4] == "Var(":
            return []
        elif type(e) == z3.ArithRef or type(e) == z3.BoolRef\
                or type(e) == z3.FuncDeclRef:
            return [e]
        else:
            return []
    else:
        res = []
        for x in e.children():
            res += extract_vars(x)
        return res


def parse_and_create_assumptions(assumption_records: pd.DataFrame,
                                 assumption_template: z3.ExprRef
                                 ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
    # Assumes that two varaiables with different sorts
    # always have different names.
    var_dict = {x.decl().name(): x
                for x in extract_vars(assumption_template)}
    gen_var_names = list(assumption_records.columns)

    # gen_var_names should be subset of var_dict
    all_varnames = set(var_dict.keys())
    # import ipdb; ipdb.set_trace()
    assert set(gen_var_names).union(all_varnames) == all_varnames

    gen_vars = [var_dict[x] for x in gen_var_names]

    def get_assumption_assign_from_record(assumption_record: pd.Series,
                                          name_template: str):
        expr_list = []
        for vname, val in assumption_record.items():
            v = var_dict[vname]
            new_name = name_template.format(v.decl().name())
            expr_list.append(z3.Const(new_name, v.sort()) == val)
        return z3.And(*expr_list)

    assumption_assignments: List[z3.ExprRef] = []
    assumption_expressions: List[z3.ExprRef] = []
    for aid, assumption_record in assumption_records.iterrows():
        name_template = f"Assumption{aid}___" + "{}"
        assumption_assignment = get_assumption_assign_from_record(
            assumption_record, name_template)
        assert isinstance(assumption_assignment, z3.ExprRef)
        assumption_assignments.append(assumption_assignment)
        this_assumption_template = rename_vars(assumption_template,
                                               gen_vars, name_template)
        assumption_expressions.append(this_assumption_template)

    return assumption_assignments, assumption_expressions


def threadsafe_compare(a, b, lemmas, ctx: z3.Context):
    # For parallel calls, need to use unique context.
    # From https://github.com/Z3Prover/z3/blob/master/examples/python/parallel.py
    solver = MySolver(ctx=ctx)
    solver.warn_undeclared = False
    solver.s.add(lemmas)

    def x_implies_y(x, y):
        # import ipdb; ipdb.set_trace()
        solver.push()
        solver.s.add(z3.And(x, z3.Not(y)))
        start = time.time()
        logger.info("Started cmp")
        ret = str(solver.check())
        end = time.time()
        logger.info(f"Cmp took {end - start} secs, returned {ret}")
        # model = solver.model()
        # import ipdb; ipdb.set_trace()
        solver.pop()
        # if unsat then x implies y
        if(ret == "unsat"):
            return True  # x is stronger.
        else:
            return False

    a_implies_b = x_implies_y(a, b)
    # b_implies_a = x_implies_y(b, a)
    return a_implies_b


def build_adj_matrix(assumption_assignments: List[z3.ExprRef],
                     assumption_expressions: List[z3.ExprRef],
                     lemmas: z3.ExprRef) -> npt.NDArray[np.bool8]:
    assert len(assumption_expressions) == len(assumption_assignments)
    n = len(assumption_expressions)
    logger.info(f"Building adj matrix ({n})")
    # Does a imply b
    adj: List[List[Union[bool, Future]]] = \
        [[False for _ in range(n)] for _ in range(n)]

    PARALLEL: bool = True
    if(PARALLEL):
        logger.info("Created thread pool")
        thread_pool_executor = ThreadPoolExecutor(max_workers=32)
    for ia, a in enumerate(assumption_expressions):
        for ib, b in enumerate(assumption_expressions):
            if(ia == ib):
                adj[ia][ib] = True
            else:
                # _adj[ia][ib] = threadsafe_compare(a, b)
                ctx = z3.Context()
                assert ctx != z3.main_ctx
                _a = a.translate(ctx)
                _b = b.translate(ctx)
                _lemmas = lemmas.translate(ctx)
                assignments = z3.And(
                        assumption_assignments[ia],
                        assumption_assignments[ib])
                assert isinstance(assignments, z3.ExprRef)
                _assignments = assignments.translate(ctx)
                _all_lemmas = z3.And(_lemmas, _assignments)
                assert isinstance(_all_lemmas, z3.ExprRef)
                if(PARALLEL):
                    adj[ia][ib] = thread_pool_executor.submit(
                        threadsafe_compare, _a, _b, _all_lemmas, ctx)
                else:
                    adj[ia][ib] = threadsafe_compare(_a, _b, _all_lemmas, ctx)

    if(PARALLEL):
        logger.info("Waiting on thread pool")
        for ia in range(n):
            for ib in range(n):
                if(ia != ib):
                    future = adj[ia][ib]
                    assert isinstance(future, Future)
                    adj[ia][ib] = future.result()
        thread_pool_executor.shutdown()

    logger.info("Built adj")
    return np.array(adj).astype(bool)


def get_unique_assumptions(adj: npt.NDArray[np.bool8]):
    n_assumptions = len(adj)
    assumption_ids = list(range(n_assumptions))
    uf = nx.utils.UnionFind(assumption_ids)
    for i in range(n_assumptions):
        for j in range(i+1, n_assumptions):
            if(adj[i][j] and adj[j][i]):
                uf.union(i, j)

    unique_assumption_ids: List[int] = []
    # Group assumptions and choose representative assumption
    groups = list(uf.to_sets())
    for alist in groups:
        unique_assumption_ids.append(sorted(alist)[0])

    return unique_assumption_ids, groups


def get_topo_sort(unique_assumption_ids: List[int], adj: npt.NDArray[np.bool8]):
    g = nx.DiGraph()
    g.add_nodes_from(unique_assumption_ids)
    edges = []
    for i in unique_assumption_ids:
        for j in unique_assumption_ids:
            if(i != j and adj[i][j]):
                edges.append((i, j))
    g.add_edges_from(edges)
    nx.draw(nx.transitive_reduction(g), with_labels=True)
    plt.savefig('tmp/tmp.pdf')
    return nx.topological_sort(g)


def sort_print_assumptions(
        assumption_records: pd.DataFrame,
        assumption_template: z3.ExprRef, lemmas: z3.ExprRef,
        get_solution_str: Callable):
    # sort assumption assignments
    # sorted_assumption_records = assumption_records.sort_values(
    #     by=list(assumption_records.columns))
    sorted_assumption_records = assumption_records
    logger.info(sorted_assumption_records)
    # del assumption_records

    assumption_assignments, assumption_expressions = \
        parse_and_create_assumptions(sorted_assumption_records,
                                     assumption_template)

    adj = build_adj_matrix(
        assumption_assignments, assumption_expressions, lemmas)
    logger.info(f"Adj: {adj}")

    # Remove duplicates
    unique_assumption_ids, groups = get_unique_assumptions(adj)
    logger.info(f"{unique_assumption_ids}, {groups}")

    # Topological sort
    # Stronger to weaker (as adj matrix encodes implication relation)
    sorted_order = get_topo_sort(unique_assumption_ids, adj)

    solution_strs = []
    for i, uid in enumerate(sorted_order):
        solution_strs.append(
            f"{i}, {uid}: " +
            get_solution_str(sorted_assumption_records.iloc[uid],
                             None, None))
    logger.info("Sorted solutions: \n"+"\n".join(solution_strs))