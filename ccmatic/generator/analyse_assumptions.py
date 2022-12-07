import os
import gc
import itertools
import logging
import pickle
import time
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from enum import unique
from typing import Callable, Dict, List, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import z3
from ccmatic.common import substitute_values_df
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

    assumption_assignments: List[z3.ExprRef] = []
    assumption_expressions: List[z3.ExprRef] = []
    for aid, assumption_record in assumption_records.iterrows():
        name_template = f"Assumption{aid}___" + "{}"
        assumption_assignment = substitute_values_df(
            assumption_record, name_template, var_dict)
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
    del solver
    del a
    del b
    del lemmas
    del ctx
    return a_implies_b


def get_weakest_assumptions(assumption_assignments: List[z3.ExprRef],
                            assumption_expressions: List[z3.ExprRef],
                            lemmas: z3.ExprRef) -> List[int]:
    assert len(assumption_expressions) == len(assumption_assignments)
    n = len(assumption_expressions)
    logger.info(f"Finding weakest assumptions ({n})")
    filtered = set([i for i in range(n)])

    # TODO: duplicate function!
    def create_inputs(ia, ib):
        ctx = z3.Context()
        assert ctx != z3.main_ctx
        _a = assumption_expressions[ia].translate(ctx)
        _b = assumption_expressions[ib].translate(ctx)
        _lemmas = lemmas.translate(ctx)
        assignments = z3.And(assumption_assignments[ia],
                             assumption_assignments[ib])
        assert isinstance(assignments, z3.ExprRef)
        _assignments = assignments.translate(ctx)
        _all_lemmas = z3.And(_lemmas, _assignments)
        assert isinstance(_all_lemmas, z3.ExprRef)
        del assignments
        del _assignments
        del _lemmas
        return _a, _b, _all_lemmas, ctx

    PARALLEL = False
    WORKERS = 54

    def check(ia: int, nodelist: List[int]) -> Set[int]:
        ib_implies_ia: List[Union[bool, Future]] = [
            False for _ in range(len(nodelist))]

        if(not PARALLEL):
            for iib, ib in enumerate(nodelist):
                assert ib != ib
                ib_implies_ia[iib] = threadsafe_compare(*create_inputs(ib, ia))
        else:
            thread_pool_executor = ThreadPoolExecutor(max_workers=WORKERS)

            def create_and_submit_job(iib, ib):
                assert ia != ib
                inputs = create_inputs(ib, ia)
                # * ^ order of input is reversed
                # ib implies ia
                ib_implies_ia[iib] = thread_pool_executor.submit(
                    threadsafe_compare, *inputs)

            def post_process_job(iib):
                future = ib_implies_ia[iib]
                assert isinstance(future, Future)
                ib_implies_ia[iib] = future.result()
                del future

            for iib, ib, in enumerate(nodelist):
                create_and_submit_job(iib, ib)

            for iib, ib, in enumerate(nodelist):
                post_process_job(iib)

            thread_pool_executor.shutdown()

        ret = set()
        for iib, ib in enumerate(nodelist):
            if(ib_implies_ia[iib]):
                # ib implies ia, remove ib
                pass
            else:
                # keep ib
                ret.add(ib)
        return ret

    for ia in range(n):
        if(ia in filtered):
            nodelist = list(filtered - set([ia]))
            ret = check(ia, nodelist)
            filtered = filtered.intersection(ret)

    return list(filtered)


def build_adj_matrix(assumption_assignments: List[z3.ExprRef],
                     assumption_expressions: List[z3.ExprRef],
                     lemmas: z3.ExprRef) -> npt.NDArray[np.bool8]:
    assert len(assumption_expressions) == len(assumption_assignments)
    n = len(assumption_expressions)
    logger.info(f"Building adj matrix ({n})")
    # Does a imply b
    adj: List[List[Union[bool, Future]]] = \
        [[False for _ in range(n)] for _ in range(n)]

    def create_inputs(ia, ib):
        ctx = z3.Context()
        assert ctx != z3.main_ctx
        _a = assumption_expressions[ia].translate(ctx)
        _b = assumption_expressions[ib].translate(ctx)
        _lemmas = lemmas.translate(ctx)
        assignments = z3.And(assumption_assignments[ia],
                             assumption_assignments[ib])
        assert isinstance(assignments, z3.ExprRef)
        _assignments = assignments.translate(ctx)
        _all_lemmas = z3.And(_lemmas, _assignments)
        assert isinstance(_all_lemmas, z3.ExprRef)
        del assignments
        del _assignments
        del _lemmas
        return _a, _b, _all_lemmas, ctx

    PARALLEL: bool = True
    if not PARALLEL:
        for ia in range(n):
            for ib in range(n):
                if(ia == ib):
                    adj[ia][ib] = True
                else:
                    inputs = create_inputs(ia, ib)
                    adj[ia][ib] = threadsafe_compare(*inputs)
    else:
        WORKERS = 54
        BATCH_SIZE = WORKERS * 32
        logger.info("Created thread pool")
        thread_pool_executor = ThreadPoolExecutor(max_workers=WORKERS)

        def create_and_submit_job(ia, ib):
            if(ia == ib):
                adj[ia][ib] = True
            else:
                inputs = create_inputs(ia, ib)
                adj[ia][ib] = thread_pool_executor.submit(
                    threadsafe_compare, *inputs)

        def post_process_job(ia, ib):
            if(ia != ib):
                future = adj[ia][ib]
                assert isinstance(future, Future)
                adj[ia][ib] = future.result()
                del future

        # # Batch by batch
        # worklist = itertools.product(range(n), range(n))
        # itr1, itr2 = itertools.tee(worklist, 2)
        # done = 0
        # batch_i = 0

        # while done < n * n:
        #     for ia, ib in itertools.islice(itr1, 0, BATCH_SIZE):
        #         create_and_submit_job(ia, ib)

        #     for ia, ib in itertools.islice(itr2, 0, BATCH_SIZE):
        #         done += 1
        #         post_process_job(ia, ib)

        #     gc.collect()
        #     logger.info(f"Done batch {batch_i}")
        #     batch_i += 1

        # Pipeline
        worklist = itertools.product(range(n), range(n))
        itr1, itr2 = itertools.tee(worklist, 2)

        for ia, ib in itertools.islice(itr1, 0, BATCH_SIZE):
            create_and_submit_job(ia, ib)

        done = 0
        done_submit = False
        # TODO: Use done order instead of sorted order to process futures
        #  use: https://stackoverflow.com/a/41654240/5039326
        for ia, ib in itr2:
            post_process_job(ia, ib)
            done += 1
            logger.info(f"Done {done}/{n*n}.")
            if(not done_submit):
                try:
                    nia, nib = next(itr1)
                    create_and_submit_job(nia, nib)
                except StopIteration:
                    done_submit = True

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


# def write_adj_disk(unique_assumption_ids, edges):
#     f = open("tmp/graph.txt", 'w')
#     n, m = len(unique_assumption_ids), len(edges)
#     f.write(n)
#     f.write(m)
#     for i in range(n):
#         f.write()
#     f.close()

def get_graph(nodes, adj):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    edges = []
    for i in nodes:
        for j in nodes:
            if(i != j and adj[i][j]):
                edges.append((i, j))
    g.add_edges_from(edges)
    return g


def write_draw_graph(g: nx.DiGraph, outdir: str = "tmp", suffix: str = ""):
    f = open(os.path.join(outdir, f'graph{suffix}.pickle'), 'wb')
    pickle.dump(g, f)
    f.close()
    nx.draw(g, with_labels=True)
    plt.savefig(os.path.join(outdir, f'graph{suffix}.pdf'))


def write_assumptions(
        assumption_ids: List[int],
        assumption_records: pd.DataFrame,
        get_solution_str: Callable,
        outdir: str = "tmp", suffix: str = ""):
    f = open(os.path.join(outdir, f'assumptions{suffix}.txt'), 'w')
    for i, ia in enumerate(assumption_ids):
        assumption_str = get_solution_str(
            assumption_records.iloc[ia], None, None)
        f.write(f"{ia}, {i}.\n{assumption_str}\n")
    f.close()


def sort_print_assumptions(
        assumption_records: pd.DataFrame,
        assumption_template: z3.ExprRef, lemmas: z3.ExprRef,
        get_solution_str: Callable,
        outdir: str = "tmp", suffix: str = ""):
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
    g = get_graph(unique_assumption_ids, adj)
    rg = nx.transitive_reduction(g)
    write_draw_graph(rg, outdir, suffix)
    sorted_order = nx.topological_sort(rg)

    solution_strs = []
    for i, uid in enumerate(sorted_order):
        solution_strs.append(
            f"{uid}, {i} -- \n" +
            get_solution_str(sorted_assumption_records.iloc[uid],
                             None, None))
    logger.info("Sorted solutions: \n"+"\n".join(solution_strs))


def filter_print_assumptions(
        assumption_records: pd.DataFrame,
        assumption_template: z3.ExprRef, lemmas: z3.ExprRef,
        get_solution_str: Callable,
        outdir: str = "tmp", suffix: str = ""):
    assumption_assignments, assumption_expressions = \
        parse_and_create_assumptions(assumption_records,
                                     assumption_template)
    filtered_list = get_weakest_assumptions(
        assumption_assignments, assumption_expressions, lemmas)

    solution_strs = []
    for i, uid in enumerate(filtered_list):
        solution_strs.append(
            f"{uid}, {i} -- \n" +
            get_solution_str(assumption_records.iloc[uid],
                             None, None))
    logger.info("Sorted solutions: \n"+"\n".join(solution_strs))

    write_assumptions(
        filtered_list, assumption_records,
        get_solution_str, outdir, f'_filtered{suffix}')
    # For reference write all assumptions also:
    write_assumptions(
        list(range(len(assumption_records))), assumption_records,
        get_solution_str, outdir, f'_all{suffix}')
