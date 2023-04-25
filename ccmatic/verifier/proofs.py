import copy
import logging
import z3

from ccmatic import CCmatic, OptimizationStruct, Proofs, find_optimum_bounds, find_optimum_bounds_nopushpop
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
        self.bq_belief = v.bq_belief1
        self.steady__bq_belief = SteadyStateVariable(
            "steady__bq_belief",
            self.bq_belief[0][0],
            self.bq_belief[0][-1],
            None,
            z3.Real("steady__bq_belief__hi"))

        self.movement_mult__consistent = z3.Real(
            'movement_mult__consistent')
        self.movement_add__min_c_lambda = z3.Real(
            'movement_add__min_c_lambda')

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

        self.all_beliefs_improve_towards_consistency = z3.And(
            z3.Or(self.stale_minc_lambda_improves,
                  v.final_minc_lambda_consistent),
            z3.Or(v.stale_bq_belief_improves, v.final_bq_consistent))

        self.initial_beliefs_inside = z3.And(
            self.steady__minc_c_lambda.init_inside(),
            self.steady__bq_belief.init_inside())
        self.final_beliefs_inside = z3.And(
            self.steady__minc_c_lambda.final_inside(),
            self.steady__bq_belief.final_inside())

    def setup_offline_cache(self):
        if(self.solution_id == "drain_probe"):
            self.recursive[self.movement_mult__consistent] = 1.9
            self.recursive[self.steady__minc_c_lambda.lo] = 24
            self.recursive[self.steady__bq_belief.hi] = 321

            """
            [04/11 18:07:22]     adv__Desired__util_f
            0                 0.199
            [04/11 18:07:49]     adv__Desired__queue_bound_multiplier
            0                                   1.5
            [04/11 18:08:02]     adv__Desired__loss_count_bound
            0                             3.0
            [04/11 18:08:06]  {'adv__Desired__large_loss_count_bound': {0}}
            [04/11 18:08:35]     adv__Desired__loss_amount_bound
            0                              0.3
            """

    def lemma1__beliefs_become_consistent(self,):
        link = self.link
        c, v = self.link.c, self.link.v
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        queue_reduces = v.q(c.T-1) < v.q(0) - c.C * (c.R + c.D)
        queue_low = v.q(0) < c.C * (c.R + c.D)

        lemma1a = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   z3.Not(self.initial_beliefs_consistent)),
            z3.And(self.final_beliefs_valid,
                   z3.Or(self.final_beliefs_consistent,
                         self.all_beliefs_improve_towards_consistency,
                         link.d.desired_in_ss,
                         queue_reduces
                         )))

        lemma1b = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   z3.Not(self.initial_beliefs_consistent),
                   queue_low),
            z3.And(self.final_beliefs_valid,
                   z3.Or(self.final_beliefs_consistent,
                         self.all_beliefs_improve_towards_consistency,
                         link.d.desired_in_ss)))

        lemma1 = z3.And(lemma1a, lemma1b)
        assert isinstance(lemma1, z3.BoolRef)

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
        d = link.d
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        lemma21 = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   self.initial_beliefs_consistent,
                   self.initial_beliefs_inside),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   self.final_beliefs_inside,
                   d.bounded_large_loss_count))

        metric_lists = [
            [Metric(self.steady__minc_c_lambda.lo, 0.1*c.C, c.C, 1, True)],
            [Metric(self.steady__bq_belief.hi, 0, 10 * c.C * (c.R + c.D), 1, False)]]

        os = OptimizationStruct(
            self.link, self.vs, [], metric_lists,
            lemma21, self.get_counter_example_str)
        logger.info("Lemma 2.1: initial beliefs steady implies final steady")
        if(self.steady__minc_c_lambda.lo not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            assert self.steady__bq_belief.hi in self.recursive
            ss_assignments = [
                self.steady__minc_c_lambda.lo ==
                self.recursive[self.steady__minc_c_lambda.lo],
                self.steady__bq_belief.hi ==
                self.recursive[self.steady__bq_belief.hi]
            ]
            model = self.debug_verifier(lemma21, ss_assignments)

    def lemma2__beliefs_steady(self,):
        link = self.link
        c, v = self.link.c, self.link.v
        d = link.d
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        some_belief_improves_towards_shrinking = False
        if(self.solution_id == "drain_probe"):
            movement_add = self.movement_add__min_c_lambda * v.alpha
            svs = [self.steady__minc_c_lambda, self.steady__bq_belief]
            none_degrade = z3.And(*[x.does_not_degrage() for x in svs])
            at_least_one_improves = z3.Or(
                *[z3.And(x.init_outside(), x.strictly_improves_add(movement_add))
                  for x in svs])
            some_belief_improves_towards_shrinking = z3.And(
                none_degrade, at_least_one_improves)

        queue_reduces = v.q(c.T-1) < v.q(0) - c.C * (c.R + c.D)
        queue_low = v.q(0) < c.C * (c.R + c.D)

        lemma2a = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   self.initial_beliefs_consistent),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   z3.Or(self.final_beliefs_inside,
                         some_belief_improves_towards_shrinking,
                         queue_reduces),
                   d.bounded_large_loss_count))

        lemma2b = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   self.initial_beliefs_consistent,
                   queue_low),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   z3.Or(self.final_beliefs_inside,
                         some_belief_improves_towards_shrinking),
                   d.bounded_large_loss_count))

        lemma2 = z3.And(lemma2a, lemma2b)
        assert isinstance(lemma2, z3.BoolRef)

        fixed_metrics = [
            Metric(self.steady__minc_c_lambda.lo,
                   self.recursive[self.steady__minc_c_lambda.lo], c.C, 1, True),
            Metric(self.steady__bq_belief.hi, 0,
                   self.recursive[self.steady__bq_belief.hi], 1, False)]

        metric_lists = [
            [Metric(self.movement_add__min_c_lambda, 0.1, c.C, 0.1, True)]
        ]

        os = OptimizationStruct(
            self.link, self.vs, fixed_metrics, metric_lists,
            lemma2, self.get_counter_example_str)
        logger.info(
            "Lemma 2: initial beliefs consistent implies they "
            "eventually become steady and remain consistent.")
        if(self.movement_add__min_c_lambda not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            ss_assignments = [
                self.steady__minc_c_lambda.lo ==
                self.recursive[self.steady__minc_c_lambda.lo],
                self.steady__bq_belief.hi ==
                self.recursive[self.steady__bq_belief.hi],
                self.movement_add__min_c_lambda ==
                self.recursive[self.movement_add__min_c_lambda]
            ]
            model = self.debug_verifier(lemma2, ss_assignments)

    def lemma31__rate_recursive(self,):
        link = self.link
        c, v = self.link.c, self.link.v
        d = link.d
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        lemma31 = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   self.initial_beliefs_consistent,
                   self.initial_beliefs_inside,
                   self.initial_inside),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   self.final_beliefs_inside,
                   self.final_inside,
                   d.bounded_large_loss_count))

        fixed_metrics = [
            Metric(self.steady__minc_c_lambda.lo,
                   self.recursive[self.steady__minc_c_lambda.lo], c.C, 1, True),
            Metric(self.steady__bq_belief.hi, 0,
                   self.recursive[self.steady__bq_belief.hi], 1, False)]

        metric_lists = [
            [Metric(self.steady__rate.lo, 0.1*c.C, c.C, 1, True)],
            [Metric(self.steady__rate.hi, c.C, 2 * c.C, 1, False)],
            [Metric(self.steady__bottle_queue.hi, 0, 10 * c.C * (c.R + c.D), 1, False)],
        ]

        os = OptimizationStruct(
            self.link, self.vs, fixed_metrics, metric_lists,
            lemma31, self.get_counter_example_str)
        logger.info("Lemma 3.1: initial rate/queue steady implies final steady")
        if(self.steady__rate.lo not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            assert self.steady__bottle_queue.hi in self.recursive
            ss_assignments = [
                self.steady__minc_c_lambda.lo ==
                self.recursive[self.steady__minc_c_lambda.lo],
                self.steady__bq_belief.hi ==
                self.recursive[self.steady__bq_belief.hi],
                self.steady__rate.lo ==
                self.recursive[self.steady__rate.lo],
                self.steady__rate.hi ==
                self.recursive[self.steady__rate.hi],
                self.steady__bottle_queue.hi ==
                self.recursive[self.steady__bottle_queue.hi]
            ]
            model = self.debug_verifier(lemma31, ss_assignments)

    def lemma3__rate_steady(self,):
        pass

    def lemma4__objectives(self,):
        link = self.link
        c, v = self.link.c, self.link.v
        assert isinstance(c, CBRDelayLink.LinkModelConfig)
        assert isinstance(v, CBRDelayLink.LinkVariables)

        # Recompute desired after resetting.
        cc_old = link.cc
        cc = copy.copy(cc_old)
        cc.reset_desired_z3(link.v.pre)
        link.cc = cc
        d, _ = link.get_desired()
        link.cc = cc_old

        lemma4 = z3.Implies(
            z3.And(self.initial_beliefs_valid,
                   self.initial_beliefs_consistent,
                   self.initial_beliefs_inside),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   self.final_beliefs_inside,
                   d.desired_in_ss))

        fixed_metrics = [
            Metric(self.steady__minc_c_lambda.lo,
                   self.recursive[self.steady__minc_c_lambda.lo], c.C, 1, True),
            Metric(self.steady__bq_belief.hi, 0,
                   self.recursive[self.steady__bq_belief.hi], 1, False),

            Metric(cc.desired_queue_bound_alpha, 0, 3, 0.001, False),
            Metric(cc.desired_loss_amount_bound_alpha, 0, (cc.T-1)/2 + 1, 0.001, False)
        ]

        metric_lists = [
            [Metric(cc.desired_util_f, 0.01, 1, 1e-3, True)],
            [Metric(cc.desired_queue_bound_multiplier, 0, 16, 0.1, False)],
            [Metric(cc.desired_loss_count_bound, 0, (cc.T-1)/2 + 1, 0.1, False)],
            [Metric(cc.desired_large_loss_count_bound, 0, 0, 0.1, False)],
            [Metric(cc.desired_loss_amount_bound_multiplier, 0, (cc.T-1)/2 + 1, 0.1, False)],
        ]

        os = OptimizationStruct(
            self.link, self.vs, fixed_metrics, metric_lists,
            lemma4, self.get_counter_example_str)
        logger.info("Lemma 4: initial beliefs inside implies we get performance")
        if(self.steady__rate.lo not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            assert self.steady__bottle_queue.hi in self.recursive
            ss_assignments = [
                self.steady__minc_c_lambda.lo ==
                self.recursive[self.steady__minc_c_lambda.lo],
                self.steady__bq_belief.hi ==
                self.recursive[self.steady__bq_belief.hi],
                # self.steady__rate.lo ==
                # self.recursive[self.steady__rate.lo],
                # self.steady__rate.hi ==
                # self.recursive[self.steady__rate.hi],
                # self.steady__bottle_queue.hi ==
                # self.recursive[self.steady__bottle_queue.hi],
            ]
            model = self.debug_verifier(lemma4, ss_assignments)

    def proofs(self):
        self.setup_steady_variables()
        self.setup_functions()
        self.setup_conditions()
        self.setup_offline_cache()

        self.lemma1__beliefs_become_consistent()
        self.lemma21__beliefs_recursive()
        self.lemma2__beliefs_steady()
        # self.lemma31__rate_recursive()
        # self.lemma4__objectives()


class CCACProofs(Proofs):

    check_lemmas = False

    def setup_steady_variables(self):
        super().setup_steady_variables()

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
        self.svs.extend([self.steady__min_c, self.steady__max_c,
               self.steady__minc_maxc_mult_gap])

        # Movements
        self.movement_mult__consistent = z3.Real('movement_mult__consistent')
        self.movement_mult__minc_maxc = z3.Real('movement_mult__minc_maxc')
        self.movements.extend([
            self.movement_mult__consistent,
            self.movement_mult__minc_maxc,
        ])

    def setup_conditions(self):
        super().setup_conditions()

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
        self.final_beliefs_close_mult = \
            self.steady__max_c.final <= self.steady__minc_maxc_mult_gap.hi * \
            self.steady__min_c.final
        self.final_beliefs_shrink_mult = \
            self.steady__max_c.final * self.steady__min_c.initial \
            < self.steady__max_c.initial * self.steady__min_c.final

        self.final_beliefs_valid = v.min_c[0][-1] * c.min_maxc_minc_gap_mult <= v.max_c[0][-1]

        self.initial_bq_inside = self.steady__bottle_queue.init_inside()
        self.final_bq_inside = self.steady__bottle_queue.final_inside()

    def setup_offline_cache(self):
        link = self.link
        c = link.c

        self.recursive[self.movement_mult__consistent] = 1.1

        if(self.solution_id == "probe_qdel"):
            self.recursive[self.steady__min_c.lo] = 69
            self.recursive[self.steady__max_c.hi] = 301
            self.recursive[self.movement_mult__minc_maxc] = 1.7
            self.recursive[self.steady__bottle_queue.hi] = 3.3 * c.C * (c.R + c.D)

            # We get 0.87 util, 3.3 queue, 0 loss, when beliefs are consistent
            # with desired necessary.

    def lemma1__beliefs_become_consistent(self):
        """
        Lemma 1: If inconsistent
            move towards consistent or become consistent.

        At some point you can't move towards consistent, so beliefs will have
        to become consistent.
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

        lemma1 = z3.Implies(
            z3.Not(self.initial_beliefs_consistent),
            z3.And(self.final_beliefs_valid,
                   z3.Or(self.final_beliefs_consistent,
                         final_moves_consistent,
                         d.desired_in_ss
                         )))
        metric_lists = [
            [Metric(self.movement_mult__consistent,
                    c.maxc_minc_change_mult, 2, 1e-3, True)]
        ]
        os = OptimizationStruct(
            self.link, self.vs, [], metric_lists,
            lemma1, self.get_counter_example_str)
        logger.info("Lemma 1: beliefs become consistent")
        if(self.movement_mult__consistent not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            ss_assignments = [
                self.movement_mult__consistent == self.recursive[self.movement_mult__consistent]
            ]
            model = self.debug_verifier(lemma1, ss_assignments)

    def lemma21__beliefs_recursive(self):
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
        EPS = 1
        metric_lists = [
            [Metric(self.steady__min_c.lo, EPS, c.C-EPS, EPS, True)],
            [Metric(self.steady__max_c.hi, c.C+EPS, 10 * c.C, EPS, False)]
        ]
        os = OptimizationStruct(link, self.vs, [], metric_lists,
                                recursive_beliefs, self.get_counter_example_str)
        logger.info("Lemma 2.1: initial beliefs steady implies final steady")
        if(self.steady__min_c.lo not in self.recursive or
           self.steady__max_c.hi not in self.recursive):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            ss_assignments = [
                self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
                self.steady__max_c.hi == self.recursive[self.steady__max_c.hi]
            ]
            model = self.debug_verifier(recursive_beliefs, ss_assignments)

    def lemma2__beliefs_steady(self):
        """
        Lemma 2: If beliefs are consistent then
            they don't become inconsistent and
            they eventually converge to small range
        """
        link = self.link
        c = link.c
        d = link.d

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
                   z3.Or(self.final_beliefs_inside,
                         final_beliefs_improve,
                         d.desired_in_ss)))
        EPS = 1
        fixed_metrics = [
            Metric(self.steady__min_c.lo, EPS,
                   self.recursive[self.steady__min_c.lo], EPS, False),
            Metric(self.steady__max_c.hi,
                   self.recursive[self.steady__max_c.hi], 10 * c.C, EPS, True),
        ]
        metric_lists = [
            [Metric(self.movement_mult__minc_maxc, 1.2, 5, 0.1, True)]
        ]
        os = OptimizationStruct(
            link, self.vs, fixed_metrics, metric_lists,
            lemma2, self.get_counter_example_str)
        logger.info(
            "Lemma 2: initial beliefs consistent implies they "
            "eventually become steady and remain consistent.")
        if(self.movement_mult__minc_maxc not in self.recursive):
            model = find_optimum_bounds_nopushpop(self.solution, [os])
            return

        if(self.check_lemmas):
            ss_assignments = [
                self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
                self.steady__max_c.hi == self.recursive[self.steady__max_c.hi],
                self.movement_mult__minc_maxc == self.recursive[self.movement_mult__minc_maxc]
            ]
            model = self.debug_verifier(lemma2, ss_assignments)

    def lemma3__objectives(self):
        """
        Lemma 3: Find what performance is possible with above recursive
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
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   self.final_beliefs_inside,
                   d.desired_necessary))
        EPS = 1
        fixed_metrics = [
            Metric(self.steady__min_c.lo, self.recursive[self.steady__min_c.lo], c.C, EPS, True),
            Metric(self.steady__max_c.hi, c.C, self.recursive[self.steady__max_c.hi], EPS, False),
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
        metric_lists = [[x] for x in metric_non_alpha]
        os = OptimizationStruct(link, self.vs, fixed_metrics, metric_lists,
                                desired, self.get_counter_example_str)
        logger.info("Lemma 3: What are the best metrics possible for"
                    "the recursive range of minc and maxc")
        model = find_optimum_bounds(self.solution, [os])

    def lemma31__rate_recursive(self):
        """
        Lemma 3, Step 1: find smallest rate/queue state that is recursive
        """
        link = self.link
        c = link.c

        recur_rate_queue = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside,
                   self.initial_bq_inside),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   self.final_beliefs_inside,
                   self.final_bq_inside))

        EPS = 1
        fixed_metrics = [
            Metric(self.steady__min_c.lo, EPS,
                   self.recursive[self.steady__min_c.lo], EPS, True),
            Metric(self.steady__max_c.hi, self.recursive[self.steady__max_c.hi],
                   10 * c.C, EPS, False)]
        metric_lists = [
            # [Metric(self.steady__rate.lo, EPS, c.C-EPS, EPS, True)],
            # [Metric(self.steady__rate.hi, c.C+EPS, 10 * c.C, EPS, False)],
            [Metric(self.steady__bottle_queue.hi, c.C * c.R, 10 * c.C * (c.R + c.D), EPS, False)]
        ]
        os = OptimizationStruct(link, self.vs, fixed_metrics,
                                metric_lists, recur_rate_queue, self.get_counter_example_str)
        logger.info("Lemma 3.1: initial rate/queue steady implies final steady")
        if (
            # self.steady__rate.lo not in self.recursive or
            # self.steady__rate.hi not in self.recursive or
            self.steady__bottle_queue.hi not in self.recursive
        ):
            model = find_optimum_bounds(self.solution, [os])
            return

        if(self.check_lemmas):
            ss_assignments = [
                self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
                self.steady__max_c.hi == self.recursive[self.steady__max_c.hi],
                # self.steady__rate.lo == self.recursive[self.steady__rate.lo],
                # self.steady__rate.hi == self.recursive[self.steady__rate.hi],
                self.steady__bottle_queue.hi == self.recursive[self.steady__bottle_queue.hi]
            ]
            model = self.debug_verifier(recur_rate_queue, ss_assignments)

    def lemma3__rate_steady(self):
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

        bq_reduces = self.steady__bottle_queue.final < self.steady__bottle_queue.initial - c.C * c.R / 2

        lemma3 = z3.Implies(
            z3.And(self.initial_beliefs_consistent,
                   self.initial_beliefs_inside,
                   z3.Not(self.initial_bq_inside)),
            z3.And(
                self.final_beliefs_valid,
                self.final_beliefs_consistent,
                self.final_beliefs_inside,
                z3.Or(self.final_bq_inside, bq_reduces)))

        EPS = 1
        fixed_metrics = [
            # Metric(self.steady__rate.lo,
            #        self.recursive[self.steady__rate.lo], c.C-EPS, EPS, True),
            # Metric(self.steady__rate.hi, c.C+EPS,
            #        self.recursive[self.steady__rate.hi], EPS, False),
            Metric(self.steady__bottle_queue.hi, c.C * c.R,
                   self.recursive[self.steady__bottle_queue.hi], EPS, False),
            Metric(self.steady__min_c.lo, EPS,
                   self.recursive[self.steady__min_c.lo], EPS, True),
            Metric(self.steady__max_c.hi,
                   self.recursive[self.steady__max_c.hi], 10 * c.C, EPS, False)
        ]
        # metric_lists = [
        #     [Metric(self.movement_mult__rate_queue, 1.5, 3, EPS, True)]
        # ]
        # os = OptimizationStruct(link, self.vs, fixed_metrics,
        #                         metric_lists, lemma3,
        #                         self.get_counter_example_str)
        # logger.info("Lemma 3: move quickly towards rate/queue")
        # if(self.movement_mult__rate_queue not in self.recursive):
        #     model = find_optimum_bounds(self.solution, [os])
        #     return

        if(self.check_lemmas):
            ss_assignments = [
                self.steady__min_c.lo == self.recursive[self.steady__min_c.lo],
                self.steady__max_c.hi == self.recursive[self.steady__max_c.hi],
                self.steady__rate.lo == self.recursive[self.steady__rate.lo],
                self.steady__rate.hi == self.recursive[self.steady__rate.hi],
                self.steady__bottle_queue.hi == self.recursive[self.steady__bottle_queue.hi],
                # self.movement_mult__rate_queue == self.recursive[self.movement_mult__rate_queue]
            ]
            model = self.debug_verifier(lemma3, ss_assignments)

    def lemma4__objectives(self):
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
                   self.initial_beliefs_inside,
                   self.initial_bq_inside),
            z3.And(self.final_beliefs_valid,
                   self.final_beliefs_consistent,
                   self.final_beliefs_inside,
                   self.final_bq_inside,
                   d.desired_in_ss))

        EPS = 1
        fixed_metrics = [
            # Metric(self.steady__rate.lo,
            #        self.recursive[self.steady__rate.lo], c.C-EPS, EPS, True),
            # Metric(self.steady__rate.hi, c.C+EPS,
            #        self.recursive[self.steady__rate.hi], EPS, False),
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
        self.setup_steady_variables()
        self.setup_functions()
        self.setup_conditions()
        self.setup_offline_cache()

        self.lemma1__beliefs_become_consistent()
        self.lemma21__beliefs_recursive()
        self.lemma2__beliefs_steady()
        # self.lemma3__objectives()
        self.lemma31__rate_recursive()
        self.lemma3__rate_steady()
        self.lemma4__objectives()