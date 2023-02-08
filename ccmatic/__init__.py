import math
import pandas as pd
import copy
from dataclasses import dataclass
import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import z3
from ccac.variables import VariableNames

from ccmatic.cegis import CegisCCAGen, CegisConfig, CegisMetaData
from ccmatic.common import flatten, get_renamed_vars, try_except
from ccmatic.verifier import (SteadyStateVariable, get_belief_invariant, get_cex_df, get_desired_necessary,
                              get_desired_ss_invariant, get_gen_cex_df,
                              run_verifier_incomplete, setup_cegis_basic)
from ccmatic.verifier.assumptions import AssumptionVerifier
from ccmatic.verifier.ideal import IdealLink
from cegis import NAME_TEMPLATE, get_unsat_core
from cegis.multi_cegis import VerifierStruct
from cegis.util import Metric, fix_metrics, get_raw_value, optimize_multi_var, z3_max_list, z3_min, z3_min_list
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver


logger = logging.getLogger('ccmatic')
GlobalConfig().default_logger_setup(logger)

EPS = 1e-1


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

            app_rate = counter_example.eval(cc.app_rate)
            ret = "{}\n{}, alpha={}, buf_size={}{}, app_rate={}.".format(
                df, desired_string, counter_example.eval(v.alpha),
                buf_size, start_state_str, app_rate)
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

    def run_cegis(self, known_solution: Optional[z3.ExprRef]=None,
                  solution_log_path: Optional[str] = None,
                  run_log_path: Optional[str] = None):
        # Directly update any closures or critical_generator_vars
        # or any other expression before calling run function.

        assert self.get_solution_str is not None, \
            "This needs to be set manually based on template."

        cc = self.cc

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
            self.ctx, debug_known_solution, md, solution_log_path=solution_log_path,
            run_log_path=run_log_path)
        if(not cc.opt_ve):
            vvars = self.verifier_vars + self.definition_vars
            spec = z3.Implies(self.definitions, self.specification)
            cg = CegisCCAGen(
                self.generator_vars, vvars, [],
                self.search_constraints, True, spec,
                self.ctx, debug_known_solution, md,
                solution_log_path=solution_log_path, run_log_path=run_log_path)
        cg.get_solution_str = self.get_solution_str
        cg.get_counter_example_str = self.get_counter_example_str
        cg.get_generator_view = self.get_generator_view
        cg.get_verifier_view = self.get_verifier_view
        if(cc.opt_wce):
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
    _desired: Optional[z3.BoolRef] = None
    _get_counter_example_str_function: Optional[Callable] = None

    def get_desired(self) -> z3.BoolRef:
        if(self._desired is None):
            _, desired = self.ccmatic.get_desired()
            return desired
        return self._desired

    def get_get_counter_example_str_function(self):
        if(self._get_counter_example_str_function is None):
            return self.vs.get_counter_example_str
        return self._get_counter_example_str_function


def find_optimum_bounds(
        solution: z3.BoolRef, optimization_structs: List[OptimizationStruct],
        extra_constraints: List[z3.BoolRef] = []):

    for ops in optimization_structs:
        link = ops.ccmatic
        cc = link.cc
        desired = ops.get_desired()
        get_cex_str_fun = ops.get_get_counter_example_str_function()
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
                         get_cex_str_fun(model, link.verifier_vars))
            logger.critical("Note, the desired string in above output is based "
                            "on cegis metrics instead of optimization metrics.")
            import ipdb; ipdb.set_trace()
            return model

        else:
            # uc = get_unsat_core(verifier)
            # import ipdb; ipdb.set_trace()

            logger.info(f"Solver gives {str(sat)} with loosest bounds.")
            verifier.pop()

            # Change logging levels used by optimize_multi_var
            # GlobalConfig().logging_levels['cegis'] = logging.INFO
            # cegis_logger = logging.getLogger('cegis')
            # GlobalConfig().default_logger_setup(cegis_logger)

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


def steady_state_variables_improve(
        svs: List[SteadyStateVariable],
        movement_mult: Optional[Union[z3.ExprRef, float]] = None):
    # Improves =
    #     none should degrade AND
    #     atleast one that is outside must move towards inside
    none_degrade = z3.And([x.does_not_degrage() for x in svs])
    atleast_one_improves = z3.Or(
        [z3.And(x.init_outside(), x.strictly_improves(movement_mult))
         for x in svs])
    final_improves = z3.And(atleast_one_improves, none_degrade)
    return final_improves


class Proofs:
    link: CCmatic
    solution: z3.BoolRef
    recursive: Dict[Optional[z3.ExprRef], float] = {}
    cache: Dict[str, Any] = {}

    def __init__(self, link: CCmatic, solution: z3.BoolRef, solution_id: int):
        self.link = link
        self.solution = solution
        self.solution_id = solution_id


class BeliefProofs(Proofs):

    """
    Each lemma has two types of constraints:
    (1) Recursivity (if you are in a good state, you don't become bad)
    (2) Performant (if you are in good state, you get to narrower good state) OR
                   (if you are in good state, you get performance objectives)

    The good state might have a range (this needs to be computed),
    getting to a narrower good state may take time (this needs to be a bounded using a finite time)
    """

    def setup_steady_variables_functions(self):
        link = self.link
        c, v = self.link.c, self.link.v
        # For now only looking at link rate in belief state (not buffer size).
        assert c.beliefs_use_buffer is False
        # I have only thought about single flow case for these proofs.
        assert c.N == 1

        # Steady states
        first = link.cc.history
        self.steady__min_c = SteadyStateVariable(
            "steady__min_c",
            v.min_c[0][0],
            v.min_c[0][-1],
            z3.Real("steady__min_c__lo"),
            None)
        self.steady__max_c = SteadyStateVariable(
            "steady__max_c",
            v.max_c[0][0],
            v.max_c[0][-1],
            None,
            z3.Real("steady__max_c__hi"))
        self.steady__minc_maxc_mult_gap = SteadyStateVariable(
            "steady__minc_maxc_mult_gap",
            None,
            None,
            None,
            z3.Real("steady__minc_maxc_mult_gap__hi"))
        self.steady__minc_maxc_add_gap = SteadyStateVariable(
            "steady__minc_maxc_add_gap",
            self.steady__max_c.initial-self.steady__min_c.initial,
            self.steady__max_c.final-self.steady__min_c.final,
            None,
            z3.Real("steady__minc_maxc_add_gap__hi"))
        self.steady__rate = SteadyStateVariable(
            "steady__rate",
            v.r_f[0][first],
            v.r_f[0][-1],
            z3.Real("steady__rate__lo"),
            z3.Real("steady__rate__hi"))
        self.steady__bottle_queue = SteadyStateVariable(
            "steady__bottle_queue",
            (v.A[first] - v.L[first]) - (v.C0 + c.C * (first) - v.W[first]),
            (v.A[c.T-1] - v.L[c.T-1]) - (v.C0 + c.C * (c.T-1) - v.W[c.T-1]),
            None,
            z3.Real("steady__bottle_queue__hi"))
        svs = [self.steady__min_c, self.steady__max_c,
               self.steady__minc_maxc_mult_gap, self.steady__minc_maxc_add_gap,
               self.steady__rate, self.steady__bottle_queue]

        # Movements
        self.movement_mult__consistent = z3.Real('movement_mult__consistent')
        self.movement_mult__minc_maxc_mult_gap = z3.Real(
            'movement_mult__minc_maxc_mult_gap')
        self.movement_mult__rate = z3.Real('movement_mult__rate_queue')
        self.movement_mult__minc_maxc = z3.Real('movement_mult__minc_maxc')
        movements = [
            self.movement_mult__consistent,
            self.movement_mult__minc_maxc_mult_gap,
            self.movement_mult__rate,
            self.movement_mult__minc_maxc
        ]

        vs = link.get_verifier_struct()
        def get_counter_example_str(counter_example: z3.ModelRef,
                                    verifier_vars: List[z3.ExprRef]) -> str:
            ret = vs.get_counter_example_str(counter_example, verifier_vars)
            sv_strs = []
            for sv in svs:
                lo = sv.lo
                hi = sv.hi
                if(lo is not None):
                    val = get_raw_value(counter_example.eval(lo))
                    if(not math.isnan(val)):
                        sv_strs.append(f"{lo.decl().name()}="
                                       f"{val}")
                if(hi is not None):
                    val = get_raw_value(counter_example.eval(hi))
                    if(not math.isnan(val)):
                        sv_strs.append(f"{hi.decl().name()}="
                                       f"{val}")
            ret += "\n"
            ret += ", ".join(sv_strs)
            movement_strs = []
            for movement in movements:
                val = get_raw_value(counter_example.eval(movement))
                if(not math.isnan(val)):
                    movement_strs.append(f"{movement.decl().name()}="
                                         f"{val}")
            ret += "\n"
            ret += ", ".join(movement_strs)
            return ret

        def debug_verifier(lemma: z3.BoolRef, extra_constraints):
            verifier = MySolver()
            verifier.warn_undeclared = False
            verifier.add(link.definitions)
            verifier.add(link.environment)
            verifier.add(self.solution)
            verifier.add(z3.Not(lemma))
            verifier.add(z3.And(extra_constraints))
            sat = verifier.check()
            if(str(sat) == "sat"):
                model = verifier.model()
                logger.error("Lemma violated. Cex:\n" +
                             get_counter_example_str(model, link.verifier_vars))
                logger.critical("Note, the desired string in above output is based "
                                "on cegis metrics instead of optimization metrics.")
                import ipdb; ipdb.set_trace()
                return model
            else:
                logger.info("Lemma passes")

        self.vs = link.get_verifier_struct()
        self.debug_verifier = debug_verifier
        self.get_counter_example_str = get_counter_example_str

    def setup_conditions(self):
        link = self.link
        c, v = link.c, link.v

        # Beliefs consistent
        _initial_minc_consistent = z3.And([v.min_c[n][0] <= c.C
                                           for n in range(c.N)])
        _initial_maxc_consistent = z3.And([v.max_c[n][0] >= c.C
                                           for n in range(c.N)])
        self.initial_beliefs_consistent = z3.And(
            _initial_minc_consistent, _initial_maxc_consistent)
        _final_minc_consistent = z3.And([v.min_c[n][-1] <= c.C
                                        for n in range(c.N)])
        _final_maxc_consistent = z3.And([v.max_c[n][-1] >= c.C
                                        for n in range(c.N)])
        self.final_beliefs_consistent = z3.And(
            _final_minc_consistent, _final_maxc_consistent)

        # first = 0  # cc.history
        # self.final_moves_consistent = z3.Or(
        #     z3.And(v.max_c[0][first] < c.C, v.max_c[0][-1] > v.max_c[0][first]),
        #     z3.And(v.min_c[0][first] > c.C, v.min_c[0][-1] < v.min_c[0][first]))
        # self.final_beliefs_invalid = v.max_c[0][-1] <= v.min_c[0][-1]
        # ^^^ We assume if min_c and max_c become exactly same, then that is
        # also invalid, and we reset beleifs in this case... I think there was
        # some issue when max_c and min_c become same.
        # TODO: Check if this concern is still there. Ideally, we'd not want
        # this, as in ideal links this can very well happen.

        # Beliefs in steady range
        self.initial_beliefs_inside = z3.And(
            self.steady__max_c.init_inside(), self.steady__min_c.init_inside())
        self.final_beliefs_inside = z3.And(
            self.steady__max_c.final_inside(), self.steady__min_c.final_inside())
        # self.final_beliefs_improve = steady_state_variables_improve(
        #     [self.steady__min_c, self.steady__max_c], self.movement_mult__minc_maxc)

        # Belief gaps small
        # TODO: see if we ever need these.
        self.initial_beliefs_close_mult = \
            self.steady__max_c.initial <= self.steady__minc_maxc_mult_gap.hi * \
            self.steady__min_c.initial
        self.initial_beliefs_close_add = z3.And(
            self.steady__minc_maxc_add_gap.initial
            <= self.steady__minc_maxc_add_gap.hi)
        self.final_beliefs_close_add = \
            self.steady__minc_maxc_add_gap.final <= self.steady__minc_maxc_add_gap.hi
        self.final_beliefs_close_mult = \
            self.steady__max_c.final <= self.steady__minc_maxc_mult_gap.hi * \
            self.steady__min_c.final
        self.final_beliefs_shrink_mult = \
            self.steady__max_c.final * self.steady__min_c.initial \
            < self.steady__max_c.initial * self.steady__min_c.final
        self.final_beliefs_shrink_add = self.steady__minc_maxc_add_gap.final < self.steady__minc_maxc_add_gap.initial

        # Rate/bottle_queue
        self.initial_inside = z3.And(
            self.steady__rate.init_inside(),
            self.steady__bottle_queue.init_inside())
        self.final_inside = z3.And(
            self.steady__rate.final_inside(),
            self.steady__bottle_queue.final_inside())
        # self.final_improves = steady_state_variables_improve(
        #         [self.steady__rate, self.steady__bottle_queue])

    def setup_offline_cache(self):
        link = self.link
        c = link.c

        """
        [01/26 15:46:34]  Lemma 1
        [01/26 15:46:34]  Testing link: adv
        [01/26 15:57:56]  Solver gives unsat with loosest bounds.
        [01/26 15:57:56]  --------------------------------------------------------------------------------
        [01/26 15:57:56]  This Try: {'movement_mult__consistent': 1.1}
        [01/26 15:57:56]  Finding bounds for movement_mult__consistent
        [01/26 15:57:56]  Optimizing movement_mult__consistent.
        [01/26 16:13:19]  Found bounds for movement_mult__consistent, 1.1, (1.1, None, 1.1008789062500002)
        [01/26 16:13:19]  {'movement_mult__consistent': {1.1}}
        [01/26 16:13:19]  ================================================================================
        """
        self.recursive[self.movement_mult__consistent] = 1.1

        # drain_then_blast_then_stable
        if(self.solution_id == 5):
            self.recursive[self.steady__min_c.lo] = c.C/3
            self.recursive[self.steady__max_c.hi] = 4.5 * c.C
            self.recursive[self.movement_mult__minc_maxc] = 1.5

            # self.recursive[self.steady__rate.lo] = c.C/2
            # self.recursive[self.steady__rate.hi] = 2 * c.C
            # self.recursive[self.steady__bottle_queue.hi] = 4 * c.C * (c.R + c.D)

            # self.recursive[self.movement_mult__rate] = 2

        # blast_then_medblast_then_minc_negalpha_correct_units
        if(self.solution_id == 6):
            """
            We find 38.8 and 300 as the recursive region for minc and maxc.
            The lower bound probably depends on:
                (1) how much data we had when we timed out min_c
                (2) when probing is done by CCA, i.e.,
                    largest min_c at which probing won't be done (= c.C/2).
            """
            self.recursive[self.steady__min_c.lo] = c.C/3
            self.recursive[self.steady__max_c.hi] = 3 * c.C

            self.recursive[self.movement_mult__minc_maxc] = 2

            """
            Rate lies in range [50, 200], i.e, [C/2, 2C], while bottleneck queue is
            below 800, i.e., 4 * c.C * (c.R + c.D).
            """

            """
            Deprecated output:
            Since the mult_gap or add_gap is not recursive,
            below states don't matter.
            If we keep minc_maxc_mult_gap to 2:
                Smallest recursive state is rate in [60, 200],
                bottle_queue in 300.
            i.e., anything larger should still be recursive.

            If we keep minc_maxc_add_gap to 10:
                Smallest recursive state is rate in [60, 200],
                bottle_queue is recursive in 300.
            """

            self.recursive[self.steady__rate.lo] = c.C/2
            self.recursive[self.steady__rate.hi] = 2 * c.C
            self.recursive[self.steady__bottle_queue.hi] = 4 * c.C * (c.R + c.D)

            self.recursive[self.movement_mult__rate] = 2

            """
            [01/23 16:47:06]  Lemma 4 - Objectives in steady state
            [01/23 16:47:06]  Testing link: adv
            [01/23 17:05:39]  Solver gives unsat with loosest bounds.
            [01/23 17:05:39]  --------------------------------------------------------------------------------
            [01/23 17:05:39]  This Try: {'adv__Desired__util_f': 0.01}
            [01/23 17:05:39]  Finding bounds for adv__Desired__util_f
            [01/23 17:05:39]  Optimizing adv__Desired__util_f.
            [01/23 23:01:13]  Found bounds for adv__Desired__util_f, 0.499, (0.49919921874999995, None, 0.500166015625)
            [01/23 23:01:13]  {'adv__Desired__util_f': {0.499, 0.01}}
            [01/23 23:01:13]     adv__Desired__util_f
            0                 0.499
            [01/23 23:01:13]  --------------------------------------------------------------------------------
            [01/23 23:01:13]  --------------------------------------------------------------------------------
            [01/23 23:01:13]  This Try: {'adv__Desired__queue_bound_multiplier': 16}
            [01/23 23:01:13]  Finding bounds for adv__Desired__queue_bound_multiplier
            [01/23 23:01:13]  Optimizing adv__Desired__queue_bound_multiplier.
            [01/24 00:57:02]  Found bounds for adv__Desired__queue_bound_multiplier, 6.0, (5.9375, None, 6.0)
            [01/24 00:57:02]  {'adv__Desired__queue_bound_multiplier': {16, 6.0}}
            [01/24 00:57:02]     adv__Desired__queue_bound_multiplier
            0                                   6.0
            [01/24 00:57:02]  --------------------------------------------------------------------------------
            [01/24 00:57:02]  --------------------------------------------------------------------------------
            [01/24 00:57:02]  This Try: {'adv__Desired__loss_count_bound': 10}
            [01/24 00:57:02]  Finding bounds for adv__Desired__loss_count_bound
            [01/24 00:57:02]  Optimizing adv__Desired__loss_count_bound.
            [01/24 01:50:58]  Found bounds for adv__Desired__loss_count_bound, 4.1000000000000005, (3.984375, None, 4.0625)
            [01/24 01:50:58]  {'adv__Desired__loss_count_bound': {10, 4.1000000000000005}}
            [01/24 01:50:58]     adv__Desired__loss_count_bound
            0                             4.1
            [01/24 01:50:58]  --------------------------------------------------------------------------------
            [01/24 01:50:58]  --------------------------------------------------------------------------------
            [01/24 01:50:58]  This Try: {'adv__Desired__large_loss_count_bound': 10}
            [01/24 01:50:58]  Finding bounds for adv__Desired__large_loss_count_bound
            [01/24 01:50:58]  Optimizing adv__Desired__large_loss_count_bound.
            [01/24 01:57:47]  Found bounds for adv__Desired__large_loss_count_bound, 0.0, (0, None, 0)
            [01/24 01:57:47]  {'adv__Desired__large_loss_count_bound': {0.0, 10}}
            [01/24 01:57:47]     adv__Desired__large_loss_count_bound
            0                                   0.0
            [01/24 01:57:47]  --------------------------------------------------------------------------------
            [01/24 01:57:47]  --------------------------------------------------------------------------------
            [01/24 01:57:47]  This Try: {'adv__Desired__loss_amount_bound': 10}
            [01/24 01:57:47]  Finding bounds for adv__Desired__loss_amount_bound
            [01/24 01:57:47]  Optimizing adv__Desired__loss_amount_bound.        [01/24 03:40:34]  Found bounds for adv__Desired__loss_amount_bound, 2.1, (1.953125, None, 2.03125)
            [01/24 03:40:34]  {'adv__Desired__loss_amount_bound': {10, 2.1}}
            [01/24 03:40:34]     adv__Desired__loss_amount_bound
            0                              2.1
            [01/24 03:40:34]  --------------------------------------------------------------------------------
            [01/24 03:40:34]  ================================================================================
            """

            """
            The one with correct units also gets the same bounds as above.
            """

        # blast_then_medblast_then_minc_negalpha_correct_units_higher_util
        if(self.solution_id == 7):
            self.recursive[self.steady__min_c.lo] = c.C/2
            self.recursive[self.steady__max_c.hi] = 2.5 * c.C

            self.recursive[self.movement_mult__minc_maxc] = 1.2

        # blast_then_medblast_then_minc_negalpha_correct_units_lower_loss
        if(self.solution_id == 8):
            pass

    def lemma1(self):
        """
        Lemma 1: If inconsistent and outside small range then beliefs
            eventually shrink to small range or
            eventually become consistent

        In this function:
        * Eventually needs to be finite time.
        * Small range needs to be computed
        """
        link = self.link
        c = link.c
        v = link.v
        d = link.d
        first = 0  # cc.history
        final_moves_consistent = z3.Or(
            z3.And(
                v.max_c[0][first] < c.C,
                z3.Or(v.max_c[0][-1] > self.movement_mult__consistent * v.max_c[0][first],
                      v.min_c[0][-1] > self.movement_mult__consistent * v.min_c[0][first])),
            z3.And(
                v.min_c[0][first] > c.C,
                z3.Or(v.min_c[0][-1] * self.movement_mult__consistent < v.min_c[0][first],
                      v.max_c[0][-1] * self.movement_mult__consistent < v.max_c[0][first]))
        )
        final_valid = v.min_c[0][-1] * c.min_maxc_minc_gap_mult <= v.max_c[0][-1]

        # # Alternate for this would have been that you count beliefs as changed
        # # only if they changed by a sufficient amount.
        # initial_bottleneck_queue_empty = v.A[first] - \
        #     v.L[first] - (v.C0 + c.C * first - v.W[first]) <= 0
        # final_bottleneck_queue_empty = v.A[c.T-1] - \
        #     v.L[c.T-1] - (v.C0 + c.C * (c.T-1) - v.W[c.T-1]) <= 0
        # lemma1_a = z3.Implies(
        #     z3.And(z3.Not(self.initial_beliefs_consistent),
        #            z3.Not(initial_bottleneck_queue_empty)),
        #     z3.Or(self.final_beliefs_consistent,
        #           final_moves_consistent,
        #           final_invalid,
        #           final_bottleneck_queue_empty,
        #           # d.desired_in_ss
        #           ))
        # lemma1_b = z3.Implies(
        #     z3.And(z3.Not(self.initial_beliefs_consistent),
        #            initial_bottleneck_queue_empty),
        #     z3.Or(self.final_beliefs_consistent,
        #           final_moves_consistent,
        #           final_invalid,
        #           # d.desired_in_ss
        #           ))
        # lemma1 = z3.And(lemma1_a, lemma1_b)

        lemma1 = z3.Implies(
            z3.Not(self.initial_beliefs_consistent),
            z3.And(final_valid,
                   z3.Or(self.final_beliefs_consistent,
                         final_moves_consistent,
                         # d.desired_in_ss
                         )))
        metric_lists = [
            [Metric(self.movement_mult__consistent,
                    c.maxc_minc_change_mult, 2, 1e-3, True)]
        ]
        os = OptimizationStruct(
            self.link, self.vs, [], metric_lists,
            lemma1, self.get_counter_example_str)
        logger.info("Lemma 1: move quickly towards consistent")
        if(self.movement_mult__consistent not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        ss_assignments = [
            self.movement_mult__consistent == self.recursive[self.movement_mult__consistent]
        ]
        model = self.debug_verifier(lemma1, ss_assignments)

    def deprecated_lemma1_2(self):
        """
        Lemma 1.2: If inconsistent in small range (computed here) then
            beliefs don't increase beyond small range and
            we get desired properties
        """

        link = self.link
        d = link.d
        lemma1_2 = z3.Implies(
            z3.And(
                z3.Not(self.initial_beliefs_consistent),
                self.initial_beliefs_close_mult
            ),
            z3.And(self.final_beliefs_close_mult, d.desired_in_ss)
        )

        metric_lists = [
            [Metric(self.movement_mult__minc_maxc_mult_gap, 1.2, 5, 1e-2, True)]
        ]
        os = OptimizationStruct(
            self.link, self.vs, [], metric_lists,
            lemma1_2, self.get_counter_example_str)
        logger.info("Lemma 1.2")
        find_optimum_bounds(self.solution, [os])

    def deprecated_lemma1_1(self):
        # If we timeout beliefs when they get too close, init conditions can
        # never be beliefs in small range.
        """
        Lemma 1.1: If inconsistent in small range (computed here) then beliefs
            don't increase beyond small range and
            evetually become consistent

        * Eventually needs to be finite time.
        """
        link = self.link
        d = link.d
        # d, _ = link.get_desired()  # this recomputes d.
        c = link.c
        v = link.v
        first = 0  # link.cc.history

        # There may be cases when beliefs are inconsistent in small range and remain like that.
        # In these cases, the CCA still delivers on desired properties.

        # Since beliefs are inconsistent in this lemma. We use the mult_gap
        # between maxc and minc to gauge the size of the belief state.
        final_moves_consistent = z3.Or(
            z3.And(v.max_c[0][first] < c.C, v.max_c[0][-1] >
                   self.movement_mult__consistent * v.max_c[0][first]),
            z3.And(v.min_c[0][first] > c.C,
                   v.min_c[0][-1] * self.movement_mult__consistent < v.min_c[0][first]))
        lemma1_1 = z3.Implies(
            z3.And(z3.Not(self.initial_beliefs_consistent),
                   self.initial_beliefs_close_mult),
            z3.Or(self.final_beliefs_consistent, # self.final_beliefs_invalid,
                  #   z3.And(
                  #       # self.final_moves_consistent,
                  #       final_moves_consistent,
                  #       self.final_beliefs_close_mult),
                  d.desired_in_ss))
        # fixed_metrics = [
        #     Metric(self.steady__minc_maxc_mult_gap.hi, 1.2, 2, 1e-2, True)
        # ]
        # metric_lists = [
        #     [Metric(self.movement_mult__consistent, 1.2, 10, 1e-2, True)]
        # ]
        # os = OptimizationStruct(
        #     link, self.vs, fixed_metrics, metric_lists, lemma1_1, self.get_counter_example_str)
        # logger.info("Lemma 1.1")
        # find_optimum_bounds(self.solution, [os])
        # self.cache['inconsistent_but_close__maxc_minc_mult_gap'] = 1.2

        lemma1_1 = z3.Implies(self.initial_beliefs_close_mult, False)
        ss_assignments = [
            self.steady__minc_maxc_mult_gap.hi == 1.2,
        ]
        logger.info("Lemma 1.1")
        self.debug_verifier(lemma1_1, ss_assignments)
        """
        Gap comes from what we consider invalid.

        movement_mult_consistent is not needed at all...
        it becomes consistent or invalid immediately.
        """

    def lemma2(self):
        """
        Lemma 2: If beliefs are consistent then
            they don't become inconsistent and
            they eventually converge to small range

        * The small range needs to be computed (this may be exact same as above range)
        * Eventually needs to be finite time.
        """
        link = self.link
        c = link.c

        # For final beliefs improve, they may first shrink and then come within
        # range. So we allow a steady variable to improve even though it is
        # already in range.
        svs = [self.steady__min_c, self.steady__max_c]
        none_degrade = z3.And([x.does_not_degrage() for x in svs])
        atleast_one_improves = z3.Or(
            [x.strictly_improves(self.movement_mult__minc_maxc)  # we have remove init_outside constraint here.
             for x in svs])
        final_beliefs_improve = z3.And(atleast_one_improves, none_degrade)

        lemma2 = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   z3.Not(self.initial_beliefs_inside)),
            z3.And(self.final_beliefs_consistent,
                   z3.Or(self.final_beliefs_inside, final_beliefs_improve)))
        fixed_metrics = [
            Metric(self.steady__min_c.lo, EPS,
                   self.recursive[self.steady__min_c.lo], EPS, False),
            Metric(self.steady__max_c.hi,
                   self.recursive[self.steady__max_c.hi], 10 * c.C, EPS, True),
        ]
        metric_lists = [
            [Metric(self.movement_mult__minc_maxc, 1.2, 5, EPS, True)]
        ]
        os = OptimizationStruct(
            link, self.vs, fixed_metrics, metric_lists,
            lemma2, self.get_counter_example_str)
        logger.info(
            "Lemma 2: move quickly towards maxc/minc.")
        if(self.movement_mult__minc_maxc not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        ss_assignments = [
            self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
            self.steady__max_c.hi == self.recursive[self.steady__max_c.hi],
            self.movement_mult__minc_maxc == self.recursive[self.movement_mult__minc_maxc]
        ]
        model = self.debug_verifier(lemma2, ss_assignments)

    def deprecated_recursive_mult_gap(self):
        # Lemma 2, Step 1.1: Find smallest recursive state for mult gap.
        desired = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_close_mult),
            self.final_beliefs_close_mult)
        metric_lists = [
            [Metric(self.steady__minc_maxc_mult_gap.hi, 1.2, 10, EPS, False)]]
        os = OptimizationStruct(
            self.link, self.vs, [], metric_lists, desired, self.get_counter_example_str)
        logger.info("Lemma 2: Find recursive state for mult gap")
        model = find_optimum_bounds(self.solution, [os])
        """
        [01/24 23:03:53]  Lemma 2: Find recursive state for mult gap
        [01/24 23:03:53]  Testing link: adv
        [01/24 23:09:51]  Solver gives unsat with loosest bounds.
        [01/24 23:09:51]  --------------------------------------------------------------------------------
        [01/24 23:09:51]  This Try: {'steady__minc_maxc_mult_gap__hi': 10}
        [01/24 23:09:51]  Finding bounds for steady__minc_maxc_mult_gap__hi
        [01/24 23:09:51]  Optimizing steady__minc_maxc_mult_gap__hi.        [01/25 00:27:32]  Found bounds for steady__minc_maxc_mult_gap__hi, 3.9000000000000004, (3.74375, None, 3.8125)
        [01/25 00:27:32]  {'steady__minc_maxc_mult_gap__hi': {10, 3.9000000000000004}}
        [01/25 00:27:32]     steady__minc_maxc_mult_gap__hi
        0                             3.9
        [01/25 00:27:32]  --------------------------------------------------------------------------------
        """

    def deprecated_recursive_add_gap(self):
        # Lemma 2, Step 1.2: Find smallest recursive state for add gap.
        link = self.link
        c = link.c
        desired = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_close_add),
            self.final_beliefs_close_add)
        metric_lists = [
            [Metric(self.steady__minc_maxc_add_gap.hi, EPS, 10 * c.C, EPS, False)]]
        os = OptimizationStruct(
            link, self.vs, [], metric_lists, desired, self.get_counter_example_str)
        logger.info("Lemma 2: Find recursive state for add gap")
        find_optimum_bounds(self.solution, [os])

    def lemma2_step1_recursive_minc_maxc(self):
        """
        Lemma 2, Step 1: Find smallest recursive state for minc and maxc
        """
        link = self.link
        c = link.c

        recursive_beliefs = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside),
            z3.And(self.final_beliefs_consistent,
                   self.final_beliefs_inside))
        metric_lists = [
            [Metric(self.steady__min_c.lo, EPS, c.C-EPS, EPS, True),
             Metric(self.steady__max_c.hi, c.C+EPS, 10 * c.C, EPS, False)]
        ]
        os = OptimizationStruct(link, self.vs, [], metric_lists,
                                recursive_beliefs, self.get_counter_example_str)
        logger.info("Lemma 2: recursive state for minc and maxc")

        if(self.steady__min_c.lo not in self.recursive or
           self.steady__max_c.hi not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        ss_assignments = [
            self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
            self.steady__max_c.hi == self.recursive[self.steady__max_c.hi]
        ]
        model = self.debug_verifier(recursive_beliefs, ss_assignments)

    def lemma2_step2_possible_perf_with_recursive_minc_maxc(self):
        """
        Lemma 2, Step 2: Find what performance is possible with above recursive
        state for minc/maxc.
        """
        link = self.link
        c = link.c

        # Recompute desired after resetting.
        cc_old = link.cc
        cc = copy.copy(cc_old)
        cc.reset_desired_z3(link.v.pre)
        link.cc = cc
        d, _ = link.get_desired()
        link.cc = cc_old

        desired = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside),
            z3.And(self.final_beliefs_consistent, self.final_beliefs_inside, d.desired_necessary))
        fixed_metrics = [
            Metric(self.steady__min_c.lo, EPS, 1/3 * c.C, EPS, False),
            Metric(self.steady__max_c.hi, 3 * c.C, 10 * c.C, EPS, True),
            Metric(cc.desired_loss_amount_bound_alpha, 0, 3, 0.001, False),
            Metric(cc.desired_queue_bound_alpha, 0, 3, 0.001, False),
        ]
        metric_lists = [
            [Metric(cc.desired_util_f, 0.01, 1, 1e-3, True)],
            [Metric(cc.desired_queue_bound_multiplier, 0, 16, EPS, False)],
            [Metric(cc.desired_loss_count_bound, 0, c.T, EPS, False)],
            [Metric(cc.desired_large_loss_count_bound, 0, c.T, EPS, False)],
            [Metric(cc.desired_loss_amount_bound_multiplier, 0, c.T, EPS, False)],
        ]
        os = OptimizationStruct(link, self.vs, fixed_metrics, metric_lists,
                                desired, self.get_counter_example_str)
        logger.info("Lemma 2, Step 2: What are the best metrics possible for"
                    "the recursive range of minc and maxc")
        # model = find_optimum_bounds(self.solution, [os])

        """
        [01/22 18:54:40]  --------------------------------------------------------------------------------
        [01/22 18:54:40]  This Try: {'adv__Desired__util_f': 0.01}
        [01/22 18:54:40]  Finding bounds for adv__Desired__util_f
        [01/22 19:31:34]  Found bounds for adv__Desired__util_f, 0.499, (0.49919921874999995, None, 0.500166015625)
        [01/22 19:31:34]     adv__Desired__util_f
        0                 0.499
        [01/22 19:31:34]  --------------------------------------------------------------------------------
        [01/22 19:31:34]  --------------------------------------------------------------------------------
        [01/22 19:31:34]  This Try: {'adv__Desired__queue_bound_multiplier': 16}
        [01/22 19:31:34]  Finding bounds for adv__Desired__queue_bound_multiplier
        [01/22 19:58:28]  Found bounds for adv__Desired__queue_bound_multiplier, 4.5, (4.4990234375, None, 4.5)
        [01/22 19:58:28]     adv__Desired__queue_bound_multiplier
        0                                   4.5
        [01/22 19:58:28]  --------------------------------------------------------------------------------
        [01/22 19:58:28]  --------------------------------------------------------------------------------
        [01/22 19:58:28]  This Try: {'adv__Desired__loss_count_bound': 9}
        [01/22 19:58:28]  Finding bounds for adv__Desired__loss_count_bound
        [01/22 20:23:31]  Found bounds for adv__Desired__loss_count_bound, 4.001, (3.99957275390625, None, 4.0001220703125)
        [01/22 20:23:31]     adv__Desired__loss_count_bound
        0                           4.001
        [01/22 20:23:31]  --------------------------------------------------------------------------------
        [01/22 20:23:31]  --------------------------------------------------------------------------------
        [01/22 20:23:31]  This Try: {'adv__Desired__large_loss_count_bound': 9}
        [01/22 20:23:31]  Finding bounds for adv__Desired__large_loss_count_bound
        [01/22 21:02:05]  Found bounds for adv__Desired__large_loss_count_bound, 4.001, (3.99957275390625, None, 4.0001220703125)
        [01/22 21:02:05]     adv__Desired__large_loss_count_bound
        0                                 4.001
        [01/22 21:02:05]  --------------------------------------------------------------------------------
        [01/22 21:02:05]  --------------------------------------------------------------------------------
        [01/22 21:02:05]  This Try: {'adv__Desired__loss_amount_bound': 9}
        [01/22 21:02:05]  Finding bounds for adv__Desired__loss_amount_bound
        [01/22 21:50:31]  Found bounds for adv__Desired__loss_amount_bound, 2.001, (1.99951171875, None, 2.00006103515625)
        [01/22 21:50:31]     adv__Desired__loss_amount_bound
        0                            2.001
        [01/22 21:50:31]  --------------------------------------------------------------------------------
        """

    def lemma3(self):
        """
        Lemma 3: If the beliefs are consistent and
        in small range (computed above) and
        performance metrics not in steady range then
            the beliefs don't become inconsistent and
            remain in small range and
            the performance metrics eventually stabilize into a small steady range

        * The small steady range needs to be computed
        * Eventually needs to be finite
        """

        link = self.link
        c = link.c

        # final_improves by a decent amount:
        svs = [self.steady__rate, self.steady__bottle_queue]
        # Queue might only move slowly (e.g., decrease by T * alpha).
        # Separately find movement speed of queue and rate.
        # final_improves = steady_state_variables_improve(
        #     svs, self.movement_mult__rate)
        none_degrade = z3.And([x.does_not_degrage() for x in svs])
        r = self.steady__rate
        bq = self.steady__bottle_queue
        atleast_one_improves = z3.Or([
            z3.And(r.init_outside(), r.strictly_improves(self.movement_mult__rate)),
            z3.And(bq.init_outside(), bq.strictly_improves())
        ])
        final_improves = z3.And(none_degrade, atleast_one_improves)

        lemma3 = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside, z3.Not(self.initial_inside)),
            z3.And(self.final_beliefs_consistent, self.final_beliefs_inside,
                   z3.Or(self.final_inside, final_improves)))

        fixed_metrics = [
            Metric(self.steady__rate.lo,
                   self.recursive[self.steady__rate.lo], c.C-EPS, EPS, True),
            Metric(self.steady__rate.hi, c.C+EPS,
                   self.recursive[self.steady__rate.hi], EPS, False),
            Metric(self.steady__bottle_queue.hi, c.C * c.R,
                   self.recursive[self.steady__bottle_queue.hi], EPS, False),
            Metric(self.steady__min_c.lo, EPS,
                   self.recursive[self.steady__min_c.lo], EPS, True),
            Metric(self.steady__max_c.hi,
                   self.recursive[self.steady__max_c.hi], 10 * c.C, EPS, False)
        ]
        metric_lists = [
            [Metric(self.movement_mult__rate, 2, 3, EPS, True)]
        ]
        os = OptimizationStruct(link, self.vs, fixed_metrics,
                                metric_lists, lemma3,
                                self.get_counter_example_str)
        logger.info("Lemma 3: move quickly towards rate/queue")
        if(self.movement_mult__rate not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        ss_assignments = [
            self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
            self.steady__max_c.hi == self.recursive[self.steady__max_c.hi],
            self.steady__rate.lo == self.recursive[self.steady__rate.lo],
            self.steady__rate.hi == self.recursive[self.steady__rate.hi],
            self.steady__bottle_queue.hi == self.recursive[self.steady__bottle_queue.hi],
            self.movement_mult__rate == self.recursive[self.movement_mult__rate]
        ]
        model = self.debug_verifier(lemma3, ss_assignments)

    def lemma3_1_recursive_rate_queue(self):
        """
        Lemma 3, Step 1: find smallest rate/queue state that is recursive
        """
        link = self.link
        c = link.c
        recur_rate_queue = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside,
                   self.initial_inside),
            self.final_inside)

        fixed_metrics = [
            Metric(self.steady__min_c.lo, EPS,
                   self.recursive[self.steady__min_c.lo], EPS, True),
            Metric(self.steady__max_c.hi, self.recursive[self.steady__max_c.hi],
                   10 * c.C, EPS, False)]

        metric_lists = [
            [Metric(self.steady__rate.lo, EPS, c.C-EPS, EPS, True)],
            [Metric(self.steady__rate.hi, c.C+EPS, 10 * c.C, EPS, False)],
            [Metric(self.steady__bottle_queue.hi, c.C * c.R, 10 * c.C * (c.R + c.D), EPS, False)]
        ]
        os = OptimizationStruct(link, self.vs, fixed_metrics,
                                metric_lists, recur_rate_queue, self.get_counter_example_str)
        logger.info("Lemma 3, Step 1: recursive state for rate, queue")
        if(self.steady__rate.lo not in self.recursive or
           self.steady__rate.hi not in self.recursive or
           self.steady__bottle_queue.hi not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        ss_assignments = [
            self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
            self.steady__max_c.hi == self.recursive[self.steady__max_c.hi],
            self.steady__rate.lo == self.recursive[self.steady__rate.lo],
            self.steady__rate.hi == self.recursive[self.steady__rate.hi],
            self.steady__bottle_queue.hi == self.recursive[self.steady__bottle_queue.hi]
        ]
        model = self.debug_verifier(recur_rate_queue, ss_assignments)

    def deprecated_lemma3_step1_1(self):
        """
        Lemma 3, Step 1.1: find largest minc/maxc range for which a rate/queue
        recursive state is possible.
        """
        link = self.link
        c = link.c
        desired = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside,
                   self.initial_inside),
            self.final_inside)
        fixed_metrics = flatten([
            [Metric(self.steady__rate.lo, EPS, c.C-EPS, EPS, True)],
            [Metric(self.steady__rate.hi, c.C+EPS, 10 * c.C, EPS, False)],
            [Metric(self.steady__bottle_queue.hi, c.C * c.R,
                    10 * c.C * (c.R + c.D), EPS, False)]
        ])
        metric_lists = [
            # [Metric(self.steady__minc_maxc_mult_gap.hi, 1+EPS, 10, EPS, True)],
            [Metric(self.steady__min_c.lo, EPS, c.C-EPS, EPS, False),
             Metric(self.steady__max_c.hi, c.C+EPS, 10*c.C, EPS, True)]
        ]
        os = OptimizationStruct(link, self.vs, fixed_metrics,
                                metric_lists, desired, self.get_counter_example_str)
        logger.info("Lemma 3, Step 1.1 - Largest minc/maxc for recursive rate/queue.")
        find_optimum_bounds(self.solution, [os])

        """
        We get bounds as:
        EPS to 10 * c.C

        After putting cwnd cap, this is moot. I think for all minc/maxc ranges,
        a resursive rate/queue state is possible.

        No need to run this function again...
        """

    def deprecated_lemma3_step2(self):
        """
        Lemma 3, Step 2: Find largest performant state
        """
        link = self.link
        d = link.d
        c = link.c
        desired = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_close_mult, self.initial_inside),
            d.desired_in_ss)
        fixed_metrics = []
        metric_lists = [
            [Metric(self.steady__minc_maxc_mult_gap.hi, 1+EPS, 10, EPS, True)],
            [Metric(self.steady__rate.lo, EPS, c.C-EPS, EPS, False)],
            [Metric(self.steady__rate.hi, c.C+EPS, 10 * c.C, EPS, True)],
            [Metric(self.steady__bottle_queue.hi, c.C * c.R, 10 * c.C * (c.R + c.D), EPS, True)]
        ]
        os = OptimizationStruct(link, self.vs, fixed_metrics,
                                metric_lists, desired, self.get_counter_example_str)
        logger.info("Lemma 3 - Performant state for rate, queue")
        find_optimum_bounds(self.solution, [os])

        # debug_verifier(False,
        #                [initial_consistent, initial_beliefs_close,
        #                 v.r_f[0][first] == 100-EPS])

        """
        Deprecated. Since the mult_gap or add_gap is not recursive.
        If we keep minc_maxc_mult_gap to 2
            performant state does not require bounds on rate, i.e., beliefs
            close is sufficient.
            bottle_queue needs to be within 700.

        If we keep minc_maxc_add_gap to 10
            performant state does not require bounds on rate, i.e., beliefs
            close is sufficient.
            bottle_queue needs to be within 700.
        """

    def lemma4(self):
        """
        Lemma 4: If the beliefs are consistent, in small range (computed above)
        and performance metrics are in steady range, then
            the beliefs don't become inconsistent and
            remain in small range and
            performance metrics remain in steady range
            the performance metrics satisfy desired objectives.
        """
        link = self.link
        c = link.c
        # lemma4 = z3.Implies(
        #     z3.And(self.initial_beliefs_consistent,
        #            self.initial_beliefs_close_add, self.initial_inside),
        #     z3.And(self.final_beliefs_consistent, self.final_beliefs_close_add,
        #            self.final_inside, d.desired_in_ss))

        # Recompute desired after resetting.
        cc_old = link.cc
        cc = copy.copy(cc_old)
        cc.reset_desired_z3(link.v.pre)
        link.cc = cc
        d, _ = link.get_desired()
        link.cc = cc_old

        # cc = link.cc
        # cc.reset_desired_z3(link.v.pre)
        # d, _ = link.get_desired()

        lemma4 = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside, self.initial_inside),
            z3.And(self.final_beliefs_consistent, self.final_beliefs_inside,
                   self.final_inside, d.desired_in_ss))
        fixed_metrics = [
            Metric(self.steady__rate.lo,
                   self.recursive[self.steady__rate.lo], c.C-EPS, EPS, True),
            Metric(self.steady__rate.hi, c.C+EPS,
                   self.recursive[self.steady__rate.hi], EPS, False),
            Metric(self.steady__bottle_queue.hi, c.C * c.R,
                   self.recursive[self.steady__bottle_queue.hi], EPS, False),
            Metric(self.steady__min_c.lo, EPS,
                   self.recursive[self.steady__min_c.lo], EPS, True),
            Metric(self.steady__max_c.hi,
                   self.recursive[self.steady__max_c.hi], 10 * c.C, EPS, False),

            Metric(cc.desired_loss_amount_bound_alpha, 0, 3, 0.001, False),
            Metric(cc.desired_queue_bound_alpha, 0, 3, 0.001, False),
        ]
        metric_lists = [
            [Metric(cc.desired_util_f, 0.01, 1, 1e-3, True)],
            [Metric(cc.desired_queue_bound_multiplier, 0, 16, EPS, False)],
            [Metric(cc.desired_loss_count_bound, 0, c.T, EPS, False)],
            [Metric(cc.desired_large_loss_count_bound, 0, c.T, EPS, False)],
            [Metric(cc.desired_loss_amount_bound_multiplier, 0, c.T, EPS, False)],
        ]
        os = OptimizationStruct(link, self.vs, fixed_metrics,
                                metric_lists, lemma4, self.get_counter_example_str)
        logger.info("Lemma 4: objectives in steady state (no link rate variations)")
        model = find_optimum_bounds(self.solution, [os])

    def proofs(self):
        self.setup_steady_variables_functions()
        self.setup_conditions()
        self.setup_offline_cache()

        # TODO: There are many cases where lemmas can be in terms of gaps or
        # range. We should programatically figure these things out.

        # self.deprecated_lemma1_1()
        # self.lemma2_step2_possible_perf_with_recursive_minc_maxc()

        # self.lemma1()  # movement
        # self.deprecated_recursive_mult_gap()
        # self.lemma2_step1_recursive_minc_maxc()  # recursion
        # self.lemma2()  # movement
        self.lemma3_1_recursive_rate_queue()  # recursion
        self.lemma3()  # movement
        self.lemma4()  # performance
