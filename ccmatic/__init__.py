import pandas as pd
import copy
from dataclasses import dataclass
import functools
import logging
from typing import Callable, List, Optional

import z3
from ccac.variables import VariableNames

from ccmatic.cegis import CegisCCAGen, CegisConfig, CegisMetaData
from ccmatic.common import flatten, get_renamed_vars, try_except
from ccmatic.verifier import (get_belief_invariant, get_cex_df, get_desired_necessary,
                              get_desired_ss_invariant, get_gen_cex_df,
                              run_verifier_incomplete, setup_cegis_basic)
from ccmatic.verifier.assumptions import AssumptionVerifier
from ccmatic.verifier.ideal import IdealLink
from cegis import NAME_TEMPLATE
from cegis.multi_cegis import VerifierStruct
from cegis.util import Metric, fix_metrics, optimize_multi_var, z3_min
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver


logger = logging.getLogger('ccmatic')
GlobalConfig().default_logger_setup(logger)


class CCmatic():
    # Boilerplate for common steps
    # Ad-hoc, created from past runs...

    # Call setup_config_vars
    # Define template
    # Call setup_cegis_loop
    # Define method overrides (for printing candidate solutions)
    # Call run_cegis

    cc: CegisConfig

    def __init__(self, cc: CegisConfig):
        self.cc = cc

    def get_fast_convergence(self):
        # Increases and decreases must be fast during link rate variations. In
        # case of multi-flow, just putting at least one flow has fast
        # variations.

        cc = self.cc
        v = self.v
        c = self.c
        first = cc.history
        mmBDP = c.C * (c.R + c.D)
        BDP = c.C * c.R
        fast_decrease = z3.Or(*[z3.Implies(
            v.c_f[n][first] >= 20 * mmBDP,
            v.c_f[n][c.T-1] <= v.c_f[n][first]/2) for n in range(c.N)])

        # Increase should be such that additive increase of alpha can't justify
        # the increase. Hence, one fast increase must have happened.
        increase_time_steps = c.T - first - 1
        fast_increase_list = []
        for n in range(c.N):
            initial_cwnd = z3_min(v.c_f[n][first], v.c_f[n][first+1])
            final_cwnd = z3_min(v.c_f[n][-1], v.c_f[n][-2])
            final_cwnd = v.c_f[n][-1]
            this_fi = z3.Implies(
                initial_cwnd < 0.1 * mmBDP,
                z3.Or(final_cwnd > initial_cwnd +
                      increase_time_steps*v.alpha,
                      final_cwnd >= 0.5 * BDP,
                      final_cwnd > 3/2 * initial_cwnd))
            fast_increase_list.append(this_fi)
        fast_increase = z3.Or(*fast_increase_list)

        assert isinstance(fast_decrease, z3.BoolRef)
        assert isinstance(fast_increase, z3.BoolRef)
        return fast_decrease, fast_increase

    def get_desired(self):
        cc = self.cc
        c = self.c
        v = self.v

        if(cc.synth_ss):
            d = get_desired_ss_invariant(cc, c, v)
            desired = d.desired_invariant
        elif(cc.use_belief_invariant):
            d = get_belief_invariant(cc, c, v)
            desired = d.desired_belief_invariant
        else:
            d = get_desired_necessary(cc, c, v)
            desired = d.desired_necessary

        fd, fi = self.get_fast_convergence()
        if(cc.desired_fast_decrease):
            d.fast_decrease = fd
            desired = z3.And(desired, fd)
        if(cc.desired_fast_increase):
            d.fast_increase = fi
            desired = z3.And(desired, fi)

        assert isinstance(desired, z3.ExprRef)
        return d, desired

    def setup_config_vars(self):
        cc = self.cc
        if(cc.assumption_verifier):
            (c, s, v,
             ccac_domain, ccac_definitions, environment,
             verifier_vars, definition_vars) = AssumptionVerifier.setup_cegis_basic(cc)
        elif(cc.ideal_link):
            (c, s, v,
             ccac_domain, ccac_definitions, environment,
             verifier_vars, definition_vars) = IdealLink.setup_cegis_basic(cc)
        else:
            (c, s, v,
             ccac_domain, ccac_definitions, environment,
             verifier_vars, definition_vars) = setup_cegis_basic(cc)

        self.c = c
        self.s = s
        self.v = v
        self.ccac_domain = ccac_domain
        self.ccac_definitions = ccac_definitions
        self.environment = environment
        self.verifier_vars = verifier_vars
        self.definition_vars = definition_vars

        d, desired = self.get_desired()
        self.d = d
        self.desired = desired

    def setup_cegis_loop(
            self, search_constraints: z3.ExprRef,
            template_definitions: List[z3.ExprRef],
            generator_vars: List[z3.ExprRef],
            get_solution_str: Callable[[z3.ModelRef, List[z3.ExprRef], int], str]):
        vn = VariableNames(self.v)

        self.search_constraints = search_constraints
        self.ctx = z3.main_ctx()
        self.specification = z3.Implies(self.environment, self.desired)
        assert(isinstance(self.specification, z3.ExprRef))
        _definitions = z3.And(
            self.ccac_domain, self.ccac_definitions, *template_definitions)
        assert(isinstance(_definitions, z3.ExprRef))
        self.definitions = _definitions

        if(self.d.steady_state_variables):
            for sv in self.d.steady_state_variables:
                generator_vars.append(sv.lo)
                generator_vars.append(sv.hi)
        self.generator_vars = generator_vars
        self.critical_generator_vars = generator_vars

        # Closures
        v = self.v
        c = self.c
        cc = self.cc
        d = self.d

        def get_counter_example_str(
                counter_example: z3.ModelRef,
                verifier_vars: List[z3.ExprRef]) -> str:
            df = get_cex_df(counter_example, v, vn, c)
            desired_string = d.to_string(cc, c, counter_example)
            buf_size = c.buf_min
            if(cc.dynamic_buffer):
                buf_size = counter_example.eval(c.buf_min)

            start_state_str = ""
            if(c.beliefs):
                start_state_str = ", start_state={}".format(
                    [counter_example.eval(v.start_state_f[n])
                     for n in range(c.N)])

            ret = "{}\n{}, alpha={}, buf_size={}{}.".format(
                df, desired_string, counter_example.eval(v.alpha),
                buf_size, start_state_str)
            return ret

        def get_verifier_view(
                counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
                definition_vars: List[z3.ExprRef]) -> str:
            return get_counter_example_str(counter_example, verifier_vars)

        def get_generator_view(
                solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                definition_vars: List[z3.ExprRef], n_cex: int) -> str:
            df = get_gen_cex_df(solution, v, vn, n_cex, c)

            dcopy = copy.copy(d)
            name_template = NAME_TEMPLATE + str(n_cex)
            dcopy.rename_vars(
                self.verifier_vars + self.definition_vars, name_template)
            desired_string = dcopy.to_string(cc, c, solution)

            renamed_alpha = get_renamed_vars([v.alpha], n_cex)[0]
            buf_size = c.buf_min
            if(cc.dynamic_buffer):
                renamed_buffer = get_renamed_vars([c.buf_min], n_cex)[0]
                buf_size = solution.eval(renamed_buffer)
            gen_view_str = "{}\n{}, alpha={}, buf_size={}.".format(
                df, desired_string, solution.eval(renamed_alpha), buf_size)
            return gen_view_str

        self.get_counter_example_str = get_counter_example_str
        self.get_verifier_view = get_verifier_view
        self.get_generator_view = get_generator_view
        self.get_solution_str = get_solution_str

    @staticmethod
    def get_pretty_term(val, suffix: str):
        if(val == 0):
            return None
        if(val == 1):
            return f'+{suffix}'
        if(val == -1):
            return f'-{suffix}'
        if(val > 0):
            return f'+{val}{suffix}'
        if(val < 0):
            return f'-{-val}{suffix}'

    def run_cegis(self, known_solution: Optional[z3.ExprRef]=None):
        # Directly update any closures or critical_generator_vars
        # or any other expression before calling run function.

        assert self.get_solution_str is not None, \
            "This needs to be set manually based on template."

        # Debugging:
        debug_known_solution = None
        if self.cc.DEBUG:
            debug_known_solution = known_solution
            assert known_solution is not None
            # self.search_constraints = z3.And(
            #     self.search_constraints, known_solution)
            # assert(isinstance(self.search_constraints, z3.ExprRef))

            # Definitions (including template)
            with open('tmp/definitions.txt', 'w') as f:
                assert(isinstance(self.definitions, z3.ExprRef))
                f.write(self.definitions.sexpr())

        md = CegisMetaData(self.critical_generator_vars)
        cg = CegisCCAGen(
            self.generator_vars, self.verifier_vars, self.definition_vars,
            self.search_constraints, self.definitions, self.specification,
            self.ctx, debug_known_solution, md)
        cg.get_solution_str = self.get_solution_str
        cg.get_counter_example_str = self.get_counter_example_str
        cg.get_generator_view = self.get_generator_view
        cg.get_verifier_view = self.get_verifier_view
        run_verifier = functools.partial(
            run_verifier_incomplete, c=self.c, v=self.v, ctx=self.ctx)
        cg.run_verifier = run_verifier
        try_except(cg.run)

    def get_verifier_struct(self):
        # Directly update any closures or any other expression before calling
        # this function.

        name = self.cc.name
        assert name

        vs = VerifierStruct(
            name, self.verifier_vars, self.definition_vars,
            self.definitions, self.specification)
        vs.run_verifier = functools.partial(
            run_verifier_incomplete, c=self.c, v=self.v, ctx=self.ctx)
        vs.get_counter_example_str = self.get_counter_example_str
        vs.get_verifier_view = self.get_verifier_view
        vs.get_generator_view = self.get_generator_view
        return vs


@dataclass
class OptimizationStruct:
    ccmatic: CCmatic
    vs: VerifierStruct

    fixed_metrics: List[Metric]
    optimize_metrics_list: List[List[Metric]]


def find_optimum_bounds(
        solution: z3.BoolRef, optimization_structs: List[OptimizationStruct],
        extra_constraints: List[z3.BoolRef] = []):

    for ops in optimization_structs:
        link = ops.ccmatic
        vs = ops.vs
        cc = link.cc
        _, desired = link.get_desired()
        logger.info(f"Testing link: {cc.name}")

        verifier = MySolver()
        verifier.warn_undeclared = False
        verifier.add(link.definitions)
        verifier.add(link.environment)
        verifier.add(solution)
        verifier.add(z3.Not(desired))
        verifier.add(z3.And(*extra_constraints))
        fix_metrics(verifier, ops.fixed_metrics)

        verifier.push()
        fix_metrics(verifier, flatten(ops.optimize_metrics_list))
        sat = verifier.check()

        if(str(sat) == "sat"):
            model = verifier.model()
            logger.error("Objective violted. Cex:\n" +
                         vs.get_counter_example_str(model, link.verifier_vars))
            logger.critical("Note, the desired string in above output is based "
                            "on cegis metrics instead of optimization metrics.")
            import ipdb; ipdb.set_trace()

        else:
            logger.info(f"Solver gives {str(sat)} with loosest bounds.")
            verifier.pop()

            # Change logging levels used by optimize_multi_var
            GlobalConfig().logging_levels['cegis'] = logging.INFO
            cegis_logger = logging.getLogger('cegis')
            GlobalConfig().default_logger_setup(cegis_logger)

            for metric_list in ops.optimize_metrics_list:
                verifier.push()
                other_metrics = ops.optimize_metrics_list.copy()
                other_metrics.remove(metric_list)
                fix_metrics(verifier, flatten(other_metrics))
                ret = optimize_multi_var(verifier, metric_list)
                verifier.pop()
                if(len(ret) > 0):
                    df = pd.DataFrame(ret)
                    sort_columns = [x.name() for x in metric_list]
                    sort_order = [x.maximize for x in metric_list]
                    df = df.sort_values(by=sort_columns, ascending=sort_order)
                    logger.info(df)
                    logger.info("-"*80)

            logger.info("="*80)


def long_term_proof_belief_template():
    """
    Each lemma has two types of constraints:
    (1) Recursivity (if you are in a good state, you don't become bad)
    (2) Performant (if you are in good state, you get to narrower good state) OR
                   (if you are in good state, you get performance objectives)

    The good state might have a range (this needs to be computed),
    getting to a narrower good state may take time (this needs to be a bounded using a finite time)
    """

    # Lemma 1
    """
    If inconsistent then beliefs
        eventually shrink to small range or
        eventually become consistent

    In this function:
    * Eventually needs to be finite time.
    * Small range needs to be computed
    """

    # Lemma 1.1
    """
    If inconsistent in small range (computed above) then beliefs
        don't increase beyond small range and
        evetually become consistent

    * Eventually needs to be finite time.
    """

    # Lemma 2
    """
    If beliefs are consistent then
        they don't become inconsistent and
        they eventually converge to small range

    * The small range needs to be computed (this may be exact same as above range)
    * Eventually needs to be finite time.
    """

    # Lemma 3
    """
    If the beliefs are consistent and in small range (computed above) then
        the beliefs don't become inconsistent and
        remain in small range and
        the performance metrics eventually stabilize into a small steady range

    * The small steady range needs to be computed
    * Eventually needs to be finite
    """

    # Lemma 4
    """
    If the beliefs are consistent, in small range (computed above)
    and performance metrics are in steady range, then
        the beliefs don't become inconsistent and
        remain in small range and
        performance metrics remain in steady range
        the performance metrics satisfy desired objectives.
    """