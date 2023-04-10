import logging
import z3

from ccmatic import CCmatic, OptimizationStruct, Proofs, find_optimum_bounds
from ccmatic.verifier import SteadyStateVariable, initial_beliefs
from ccmatic.verifier.cbr_delay import CBRDelayLink
from cegis.util import Metric
from pyz3_utils.common import GlobalConfig


logger = logging.getLogger('proofs')
GlobalConfig().default_logger_setup(logger)


class CBRDelayProofs(Proofs):

    check_lemmas = False

    def setup_steady_variables(self):
        super().setup_steady_variables()

        link = self.link
        c, v = self.link.c, self.link.v
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)
        assert c.N == 1

        # Steady state variables
        self.steady__minc_c_lambda = SteadyStateVariable(
            "steady__minc_c_lambda",
            v.min_c_lambda[0][0],
            v.min_c_lambda[0][-1],
            z3.Real("steady__minc_c_lambda__lo"),
            None)
        self.steady__bq_belief2 = SteadyStateVariable(
            "steady__bq_belief2",
            v.bq_belief2[0][0],
            v.bq_belief2[0][-1],
            None,
            z3.Real("steady__bq_belief2__hi"))

        self.movement_mult__consistent = z3.Real(
            'movement_mult__consistent')
        self.movement_add__min_c_lambda = z3.Real(
            'movement_mult__min_c_lambda')

        self.svs.append(self.steady__minc_c_lambda)
        self.movements.append(self.movement_mult__consistent)

    def setup_conditions(self):
        super().setup_conditions()
        link = self.link
        c, v = self.link.c, self.link.v
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        # Consistency expressions are already defined in
        # CBRDelayLink.LinkVariables
        self.initial_beliefs_valid = z3.And(
            v.initial_minc_lambda_valid, v.initial_bq_valid)
        self.initial_beliefs_consistent = z3.And(
            v.initial_minc_lambda_consistent, v.initial_bq_consistent)
        self.final_beliefs_valid = z3.And(
            v.final_minc_lambda_valid, v.final_bq_valid)
        self.final_beliefs_consistent = z3.And(
            v.final_minc_lambda_consistent, v.final_bq_consistent)

        self.stale_minc_lambda_improves = z3.And(
            [v.min_c_lambda[n][-1] * self.movement_mult__consistent < v.min_c_lambda[n][0]
             for n in range(c.N)])

        self.all_beliefs_improve = z3.And(
            z3.Or(self.stale_minc_lambda_improves,
                  v.final_minc_lambda_consistent),
            z3.Or(v.stale_bq_belief_improves, v.final_bq_consistent))

        self.initial_beliefs_inside = z3.And(
            self.steady__minc_c_lambda.init_inside(),
            self.steady__bq_belief2.init_inside())
        self.final_beliefs_inside = z3.And(
            self.steady__minc_c_lambda.final_inside(),
            self.steady__bq_belief2.final_inside())

    def setup_offline_cache(self):
        if(self.solution_id == "drain_probe"):
            self.recursive[self.movement_mult__consistent] = 1.9
            self.recursive[self.steady__minc_c_lambda.lo] = 24
            self.recursive[self.steady__bq_belief2.hi] = 321

    def lemma1__beliefs_become_consistent(self,):
        link = self.link
        c, v = self.link.c, self.link.v
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        queue_reduces = v.q(c.T-1) < v.q(0) - c.C * (c.R + c.D)
        queue_low = v.q(0) < c.C * (c.R + c.D)

        lemma1_1 = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   z3.Not(self.initial_beliefs_consistent)),
            z3.And(self.final_beliefs_valid,
                   z3.Or(self.final_beliefs_consistent,
                         self.all_beliefs_improve,
                         link.d.desired_in_ss,
                         queue_reduces
                         )))

        lemma1_2 = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   z3.Not(self.initial_beliefs_consistent),
                   queue_low),
            z3.And(self.final_beliefs_valid,
                   z3.Or(self.final_beliefs_consistent,
                         self.all_beliefs_improve,
                         link.d.desired_in_ss)))

        lemma1 = z3.And(lemma1_1, lemma1_2)

        metric_lists = [
            [Metric(self.movement_mult__consistent,
                    1.1, 3, 1e-1, True)]]

        os = OptimizationStruct(
            self.link, self.vs, [], metric_lists,
            lemma1, self.get_counter_example_str)
        logger.info("Lemma 1: beliefs become consistent")
        if(self.movement_mult__consistent not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            ss_assignments = [
                self.movement_mult__consistent ==
                self.recursive[self.movement_mult__consistent]
            ]
            model = self.debug_verifier(lemma1, ss_assignments)

    def lemma21__beliefs_recursive(self,):
        link = self.link
        c, v = self.link.c, self.link.v
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        lemma21 = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   self.initial_beliefs_consistent,
                   self.initial_beliefs_inside),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   self.final_beliefs_inside))

        metric_lists = [
            [Metric(self.steady__minc_c_lambda.lo, 0.1*c.C, c.C, 1, True)],
            [Metric(self.steady__bq_belief2.hi, 0, 10 * c.C * (c.R + c.D), 1, False)]]

        os = OptimizationStruct(
            self.link, self.vs, [], metric_lists,
            lemma21, self.get_counter_example_str)
        logger.info("Lemma 2.1: initial beliefs steady implies final steady")
        if(self.steady__minc_c_lambda.lo not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            assert self.steady__bottle_queue.hi in self.recursive
            ss_assignments = [
                self.steady__minc_c_lambda.lo ==
                self.recursive[self.steady__minc_c_lambda.lo],
                self.steady__bq_belief2.hi ==
                self.recursive[self.steady__bq_belief2.hi]
            ]
            model = self.debug_verifier(lemma21, ss_assignments)

    def lemma2__beliefs_steady(self,):
        pass

    def lemma31__rate_recursive(self,):
        pass

    def lemma3__rate_steady(self,):
        pass

    def lemma4__objectives(self,):
        pass

    def proofs(self):
        self.setup_steady_variables()
        self.setup_functions()
        self.setup_conditions()
        self.setup_offline_cache()

        self.lemma1__beliefs_become_consistent()
        self.lemma21__beliefs_recursive()
