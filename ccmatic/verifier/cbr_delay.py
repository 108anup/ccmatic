from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import z3

from ccac.config import ModelConfig
from ccac.model import (cca_paced, cwnd_rate_arrival, initial, loss_oracle,
                        relate_tot)
from ccac.variables import VariableNames, Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten, get_name_for_list
from ccmatic.verifier import (BaseLink, calculate_qbound_defs, calculate_qdel_defs,
                              exceed_queue_defs, last_decrease_defs,
                              loss_deterministic, monotone_defs)
from cegis.util import get_model_value_list, z3_max, z3_min
from pyz3_utils.my_solver import MySolver


class CBRDelayLink(BaseLink):

    class LinkVariables(Variables):

        def bq(self, t: int):
            assert t >= 0 and t <= self.c.T-1
            return self.A[t] - self.L[t] - (self.C0 + self.c.C * t - self.W[t])

        def q(self, t: int):
            assert t >= 0 and t <= self.c.T-1
            return self.A[t] - self.L[t] - self.S[t]

        def derived_expressions(self):
            c = self.c
            assert isinstance(c, CBRDelayLink.LinkModelConfig)

            if(c.beliefs):
                buffer = c.buf_min
                if(c.buf_min is None):
                    # basicallly acts as infinity
                    buffer = 10 * c.C * (c.R + c.D)

                # Belief consistency
                # min_c_lambda (valid, consistent, improves)
                self.initial_minc_lambda_valid = z3.And([
                    self.min_c_lambda[n][0] >= self.alpha
                    for n in range(c.N)
                ])
                self.final_minc_lambda_valid = z3.And([
                    self.min_c_lambda[n][-1] >= self.alpha
                    for n in range(c.N)
                ])

                MI = c.minc_lambda_measurement_interval
                self.initial_minc_lambda_consistent = z3.And([z3.And(
                    c.C * MI + buffer >= self.min_c_lambda[n][0] * (MI+c.D+1),
                    self.min_c_lambda[n][0] <= c.C) for n in range(c.N)])
                self.final_minc_lambda_consistent = z3.And([z3.And(
                    c.C * MI + buffer >= self.min_c_lambda[n][-1] * (MI+c.D+1),
                    self.min_c_lambda[n][-1] <= c.C) for n in range(c.N)])

                self.stale_minc_lambda_improves = z3.And(
                    [self.min_c_lambda[n][-1] < self.min_c_lambda[n][0]
                     for n in range(c.N)])

                # bq_belief (valid, consistent, improves)
                bq_belief = self.bq_belief2
                self.initial_bq_valid = z3.And([
                    bq_belief[n][0] >= 0
                    for n in range(c.N)])
                self.final_bq_valid = z3.And([
                    bq_belief[n][-1] >= 0
                    for n in range(c.N)])

                self.initial_bq_consistent = z3.And([
                    bq_belief[n][0] >= self.bq(0)
                    for n in range(c.N)])
                self.final_bq_consistent = z3.And([
                    bq_belief[n][-1] >= self.bq(c.T-1)
                    for n in range(c.N)])

                self.stale_bq_belief_improves = z3.And(
                    [bq_belief[n][-1] > bq_belief[n][0]
                     for n in range(c.N)])

        def __init__(self, c: ModelConfig, s: MySolver,
                     name: Optional[str] = None):
            super().__init__(c, s, name)
            self.c = c
            pre = self.pre

            if c.calculate_qdel:
                self.first_qdel = np.array([[
                    s.Bool(f"{pre}first_qdel_{t},{dt}") for dt in range(c.T)]
                    for t in range(c.T)])

            if (c.beliefs):
                self.min_c_lambda = np.array([[
                    s.Real(f"{pre}min_c_lambda_{n},{t}")
                    for t in range(c.T)]
                    for n in range(c.N)])
                self.recomputed_min_c_lambda = np.array([[
                    s.Real(f"{pre}recomputed_min_c_lambda_{n},{t}")
                    for t in range(c.T)]
                    for n in range(c.N)])
                # self.under_utilized = np.array([[
                #     s.Bool(f"{pre}under_util_{n},{t}")
                #     for t in range(c.T)]
                #     for n in range(c.N)])

                # bytes in the bottleneck queue just after the latest byte
                # acked, assuming the link rate is min_c_lambda.
                # computed based on inflight
                self.bq_belief1 = np.array([[
                    s.Real(f"{pre}bq_belief1_{n},{t}")
                    for t in range(c.T)]
                    for n in range(c.N)])

                # computed based on net arrivals and expected departures.
                self.bq_belief2 = np.array([[
                    s.Real(f"{pre}bq_belief2_{n},{t}")
                    for t in range(c.T)]
                    for n in range(c.N)])

            self.derived_expressions()

    class LinkModelConfig(ModelConfig):
        minc_lambda_measurement_interval: float = 1
        fix_stale__min_c_lambda: bool = False
        fix_stale__bq_belief: bool = False

    @staticmethod
    def update_bq_belief(c: ModelConfig, s: MySolver, v: Variables):
        assert isinstance(v, CBRDelayLink.LinkVariables)
        assert isinstance(c, CBRDelayLink.LinkModelConfig)

        for n in range(c.N):
            for t in range(c.T):
                inflight = v.A_f[n][t] - v.Ld_f[n][t] - v.S_f[n][t]
                s.add(v.bq_belief1[n][t] == inflight)
                # s.add(v.bq_belief1[n][t] == z3_max(
                #     0, inflight - v.min_c_lambda[n][t] * c.D))

        for n in range(c.N):
            for t in range(1, c.T):
                delivery_rate = v.min_c_lambda[n][t]

                if(c.fix_stale__bq_belief and t == c.T-1):
                    delivery_rate = v.recomputed_min_c_lambda[n][t]

                # TODO: might want to use the new delivery rate
                # for the all the steps.

                # sent = ((v.A_f[n][t] - v.Ld_f[n][t]) -
                #         (v.A_f[n][t-1] - v.Ld_f[n][t-1]))
                sent = v.A_f[n][t] - v.A_f[n][t-1]
                delivered = delivery_rate * 1
                bq2 = z3_max(0, v.bq_belief2[n][t-1] + sent - delivered)
                # bq2 cannot be more than inflight

                if(c.fix_stale__bq_belief and t == c.T-1):
                    s.add(v.bq_belief2[n][t] == v.bq_belief1[n][t])
                else:
                    s.add(v.bq_belief2[n][t] == z3_min(v.bq_belief1[n][t], bq2))

        # for n in range(c.N):
        #     for t in range(1, c.T):
        #         delivery_rate = v.min_c_lambda[n][t]

        #         if(c.fix_stale__bq_belief and t == c.T-1):
        #             delivery_rate = v.recomputed_min_c_lambda[n][t]

        #         # Below is wrong as delivered will be upper bounded by sent +
        #         # queue in each cycle.
        #         sent = v.A_f[n][t] - v.A_f[n][0]
        #         delivered = delivery_rate * t
        #         bq2 = z3_max(0, v.bq_belief2[n][0] + sent - delivered)
        #         # bq2 cannot be more than inflight
        #         s.add(v.bq_belief2[n][t] == z3_min(v.bq_belief1[n][t], bq2))

    @staticmethod
    def update_min_c_lambda(c: ModelConfig, s: MySolver, v: Variables):
        assert isinstance(v, CBRDelayLink.LinkVariables)
        assert isinstance(c, CBRDelayLink.LinkModelConfig)

        # for n in range(c.N):
        #     for st in range(c.T-1):
        #         high_delay = z3.Not(z3.Or(*[v.first_qdel[st+1][dt]
        #                                     for dt in range(c.D+1+1)]))
        #         loss = v.Ld_f[n][st+1] - v.Ld_f[n][st] > 0
        #         if (c.D == 0):
        #             assert c.loss_oracle
        #             loss = v.L_f[n][st+1] - v.L_f[n][st] > 0
        #         recvd_new_pkts = v.S_f[n][st+1] - v.S_f[n][st] > 0
        #         sent_new_pkts = v.A_f[n][st+1] - v.A_f[n][st] > 0
        #         this_utilized = z3.Or(
        #             z3.And(high_delay, sent_new_pkts, recvd_new_pkts),
        #             loss)
        #         s.add(v.under_utilized[n][st+1] == z3.Not(this_utilized))

        # Update min_c_lambda, i.e., belief of min_c using sending rate instead
        # of ack rate.
        for n in range(c.N):
            s.add(v.recomputed_min_c_lambda[n][0] == v.alpha)

            for t in range(1, c.T):

                utilized_t = [None for _ in range(t)]
                # utilized_cummulative = [None for _ in range(t)]
                under_utilized_cummulative = [None for _ in range(t)]
                for st in range(t-1, -1, -1):
                    high_delay = z3.Not(z3.Or(*[v.first_qdel[st+1][dt]
                                                for dt in range(c.D+1+1)]))
                    loss = v.Ld_f[n][st+1] - v.Ld_f[n][st] > 0
                    if (c.D == 0):
                        assert c.loss_oracle
                        loss = v.L_f[n][st+1] - v.L_f[n][st] > 0
                    recvd_new_pkts = v.S_f[n][st+1] - v.S_f[n][st] > 0
                    sent_new_pkts = v.A_f[n][st+1] - v.A_f[n][st] > 0
                    this_utilized = z3.Or(
                        z3.And(high_delay, sent_new_pkts, recvd_new_pkts),
                        loss)
                    utilized_t[st] = this_utilized
                    if (st + 1 == t):
                        # utilized_cummulative[st] = this_utilized
                        under_utilized_cummulative[st] = z3.Not(this_utilized)
                    else:
                        # assert utilized_cummulative[st+1] is not None
                        # utilized_cummulative[st] =
                        # z3.And(utilized_cummulative[st+1], this_utilized)
                        assert under_utilized_cummulative[st+1] is not None
                        under_utilized_cummulative[st] = z3.And(
                            under_utilized_cummulative[st+1],
                            z3.Not(this_utilized))

                base_minc = v.min_c_lambda[n][t-1]
                overall_minc = base_minc
                recomputed_minc = v.recomputed_min_c_lambda[n][t-1]
                # OLD: RTT >= 1 at the very least, and so max et is t-1.
                # ^^ A[t] = S[t] corresponds to RTT of Rm. RTT >= Rm, so max et
                # is actually t.
                # min et is 1.
                for et in range(t, 0, -1):
                    # OLD: We only use this et if et = t-RTT(t). et = t-RTT(t)
                    # iff A[et]-L[et] <= S[t] and A[et+1]-L[et+1] > S[t].
                    # ^^ We use this as long as et <= t-RTT(t).

                    # Note, while CCA cannot see L_f, it can see sequence
                    # numbers that can be used to compute rtts.
                    # correct_et = z3.And(
                    #     v.A_f[n][et] - v.L_f[n][et] <= v.S_f[n][t],
                    #     v.A_f[n][et+1] - v.L_f[n][et] > v.S_f[n][t])
                    correct_et = z3.And(
                        v.A_f[n][et] - v.L_f[n][et] <= v.S_f[n][t])
                    # for st in range(et-1, -1, -1):
                    for st in [et-c.minc_lambda_measurement_interval]:
                        window = et - st
                        assert window > 0
                        assert et >= 0
                        assert st >= 0

                        # measured_c = (v.A_f[n][et] - v.A_f[n][st]) / window
                        # this_lower = measured_c * window / (window + c.D)

                        net_sent = v.A_f[n][et] - v.A_f[n][st]
                        this_lower = net_sent / (window + (c.D+1))
                        # We do c.D+1 as that is what we can measure for qdelay.

                        # Basically, min_c_lambda means, that if rate is below
                        # it, then bottleneck queue would have build above 2D
                        # and we would measure high util.

                        filtered_lower = z3.If(
                            z3.And(correct_et, under_utilized_cummulative[st]),
                            this_lower, 0)
                        overall_minc = z3_max(overall_minc, filtered_lower)
                        recomputed_minc = z3_max(recomputed_minc, filtered_lower)

                timeout_min_c_lambda = False
                # bq_went_low_rtt_ago = False
                # first = 0
                # minc_lambda_changed = overall_minc > v.min_c_lambda[n][first]
                if(c.fix_stale__min_c_lambda and t == c.T-1):
                    timeout_allowed = (t == c.T-1)

                    # Timeout if large loss happened
                    large_loss_list: List[z3.BoolRef] = []
                    for t in range(1, c.T):
                        large_loss_list.append(v.L[t] > v.L[t-1] + v.alpha * (c.T-1))
                    large_loss_count = z3.Sum(*large_loss_list)
                    large_loss_happened = large_loss_count > 0

                    # Deprecated
                    # bq_went_low_rtt_ago_list = []
                    # for et in range(t, 0, -1):
                    #     correct_et = z3.And(
                    #         v.A_f[n][et] - v.L_f[n][et] <= v.S_f[n][t])
                    #     bq_went_low = v.r_f[n][et] > v.alpha
                    #     bq_went_low_rtt_ago_list.append(z3.And(correct_et, bq_went_low))
                    # bq_went_low_rtt_ago = z3.Or(bq_went_low_rtt_ago_list)

                    # Timeout if probe happened and queue did not drain
                    probe_happened = z3.Or(*[v.r_f[n][t] > v.alpha for t in range(1, t+1)])
                    probe_did_not_happen_in_last_2_rm = z3.And(*[v.r_f[n][t] <= v.alpha for t in range(t-1, t+1)])
                    inflight_did_not_drain = v.bq_belief1[n][t] > v.alpha
                    probe_based_timeout = z3.And(probe_happened, probe_did_not_happen_in_last_2_rm, inflight_did_not_drain)

                    # Timeout if min_c from ack rate is consistent and lower
                    # than min_c_lambda
                    minc_increased_and_lower = z3.And(v.min_c[n][c.T-1] > v.min_c[n][0], v.min_c[n][c.T-1] < overall_minc)

                    # timeout_min_c_lambda = timeout_allowed
                    # timeout_min_c_lambda = z3.And(timeout_allowed, bq_went_low_rtt_ago)
                    # timeout_min_c_lambda = z3.And(
                    #     timeout_allowed, z3.Not(minc_lambda_changed))
                    timeout_min_c_lambda = z3.If(timeout_allowed,
                                                 z3.If(large_loss_happened, True,
                                                       z3.If(minc_increased_and_lower, True,
                                                             z3.If(probe_based_timeout, True, False))),
                                                 False)

                s.add(v.recomputed_min_c_lambda[n][t] == recomputed_minc)
                clamped_minc = z3_max(recomputed_minc, 1/2 * v.min_c_lambda[n][0])
                s.add(v.min_c_lambda[n][t] ==
                      z3.If(timeout_min_c_lambda, clamped_minc, overall_minc))
                # s.add(v.min_c_lambda[n][t] == z3_max(overall_minc, v.min_c[n][t]))

    @staticmethod
    def calculate_first_qdel_defs(c: ModelConfig, s: MySolver, v: Variables):
        """
        Queuing delay at time t: Non propagation delay experienced by the first
        byte that is acked between (t-1, t].

        first_qdel[t][dt] means that the qdelay of the first byte is in range [dt,
        dt+1).

        Note, ideally it should be (dt-1, dt+1) (due to discretization of time,
        error in time is 1 and so error in time interval is 2). We are in a
        sense restricting some continuous traces. But since we check all traces,
        the cts traces we restrict at t, might work at t-1. Unclear. TODO:
        Verify this...
        """

        """
        first_qdel[t][dt>=t] is non-deterministic
        first_qdel[t=0][dt] is non-deterministic
        first_qdel[t][dt<t] is deterministic

               dt
             0 1 2 3
           ----------
          0| n n n n
        t 1| d n n n
          2| d d n n
          3| d d d n
        """

        assert isinstance(v, CBRDelayLink.LinkVariables)

        for t in range(1, c.T):
            for dt in range(t):
                s.add(
                    v.first_qdel[t][dt] ==
                    z3.If(v.S[t] != v.S[t-1],
                          z3.And(v.A[t-dt] - v.L[t-dt] > v.S[t-1],
                                 v.A[t-dt-1] - v.L[t-dt-1] <= v.S[t-1]),
                          v.first_qdel[t-1][dt]))

    @staticmethod
    def calculate_first_qdel_env(c: ModelConfig, s: MySolver, v: Variables):
        """
        This funciton is exactly same constraints as regular qdelay.
        """
        assert isinstance(v, CBRDelayLink.LinkVariables)

        # The qdel dt values mark non-overlapping time intervals, so only one of
        # them can be true at a time t.
        for t in range(c.T):
            s.add(z3.Sum(*v.first_qdel[t]) <= 1)

        # qdelay is only measured when new pkts are received. Otherwise, by
        # convention we don't change the qdelay value.
        for t in range(1, c.T):
            for dt in range(t, c.T):
                s.add(z3.Implies(v.S[t] == v.S[t-1],
                                 v.first_qdel[t][dt] == v.first_qdel[t-1][dt]))

        # qdelay cannot increase too quickly. Say, I got pkts at time t1 and t2,
        # where t2 > t1, since pkts are serviced in FIFO, the send tstamp of pkt
        # arrived at t2 must be later than that of pkt recvd at t1. Thus the RTT
        # of pkt at t2 can be atmost t2-t1 more than pkt at t1.
        # If first_qdel[t1][dt1] is true then all of first_qdel[t2>t1][dt2>dt1+t2-t1] must
        # be False and for all t2>t1, at least one of first_qdel[t2][dt2<=dt1+t2-t1]
        # must be True.
        for t1 in range(c.T-1):
            for dt1 in range(c.T):
                t2 = t1+1
                # dt2 starts from dt1+1+1
                s.add(z3.Implies(
                    v.first_qdel[t1][dt1],
                    z3.And(*[z3.Not(v.first_qdel[t2][dt2])
                            for dt2 in range(dt1+1+1, c.T)])
                ))
                s.add(z3.Implies(
                    v.first_qdel[t1][dt1],
                    z3.Or(*[v.first_qdel[t2][dt2] for dt2 in range(min(c.T, dt1+1+1))])
                ))

        # qdelay cannot be more than max_delay = buffer/C + D.
        # This means that one of first_qdel[t][dt<=max_delay] must be true. This also
        # means that all of first_qdel[t][dt>max_delay] are false.
        if (c.buf_min is not None):
            max_delay = c.buf_min/c.C + c.D
            for t in range(c.T):
                some_qdel_is_true = (1 == z3.Sum(
                    *[z3.If(dt <= max_delay, v.first_qdel[t][dt], False)
                      for dt in range(c.T)]))
                # if max_delay is very high, then all qdel can be false.
                s.add(z3.Implies(max_delay < c.T, some_qdel_is_true))

    @staticmethod
    def waste_defs(c: ModelConfig, s: MySolver, v: Variables):
        """
        Set waste such that bottleneck queue is non-negative.
        Bottleneck queue is defined as:
            A[t] - L[t] - (C0 + C*t - W[t])

        Hence W[t] =
            If A[t] - L[t] - (C0 + C*t - W[t-1]) < 0
                C0 + C*t - (A[t] - L[t])
            Else
                W[t-1]
        """
        for t in range(1, c.T):
            s.add(v.W[t] == z3.If(
                v.A[t] - v.L[t] - (v.C0 + c.C * t - v.W[t-1]) < 0,
                v.C0 + c.C * t - (v.A[t] - v.L[t]),
                v.W[t-1]))

        # This is an environment constraint. Since this only involves verifier
        # only variables, it is fine to keep it in definitions. Ideally put it
        # in environment constraints.
        s.add(v.W[0] >= v.C0 + c.C * 0 - (v.A[0] - v.L[0]))

        # Not removing waste environment constraint. Above should always satisfy those.
        # TODO: verify this.

    @staticmethod
    def initial_beliefs(c: ModelConfig, s: MySolver, v: Variables):
        assert isinstance(v, CBRDelayLink.LinkVariables)
        assert isinstance(c, CBRDelayLink.LinkModelConfig)

        s.add(v.initial_minc_lambda_valid)
        if (not c.fix_stale__min_c_lambda):
            s.add(v.initial_minc_lambda_consistent)

        s.add(v.initial_bq_valid)
        if (not c.fix_stale__bq_belief):
            s.add(v.initial_bq_consistent)

    def setup_definitions(self, c: ModelConfig, v: Variables):
        s = super().setup_definitions(c, v)
        if(c.calculate_qdel):
            self.calculate_first_qdel_defs(c, s, v)
        if(c.beliefs):
            self.update_min_c_lambda(c, s, v)
            self.update_bq_belief(c, s, v)
        self.waste_defs(c, s, v)

        return s

    def setup_environment(self, c: ModelConfig, v: Variables):
        s = super().setup_environment(c, v)
        assert isinstance(c, CBRDelayLink.LinkModelConfig)

        if(c.calculate_qdel):
            self.calculate_first_qdel_env(c, s, v)
        if(c.beliefs):
            self.initial_beliefs(c, s, v)

        if (c.fix_stale__min_c_lambda):
            # WLOG realign the finite trace window so that if there is a probe,
            # the probe is moved to 2 Rm periods before the end of the trace.
            # TODO: see how history would need to be maintained in the kernel.
            assert c.N == 1
            probe_happened = z3.Or(
                *[v.r_f[0][t] > v.alpha for t in range(1, c.T)])
            probe_happened_2_rm_ago = z3.Or(
                *[v.r_f[0][t] > v.alpha for t in range(1, c.T-2)])
            s.add(z3.Implies(probe_happened, probe_happened_2_rm_ago))

        return s

    def get_base_cegis_vars(
        self,
        cc: CegisConfig, c: ModelConfig, v: Variables
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        history = cc.history
        verifier_vars = flatten(
            [v.A_f[:, :history], v.c_f[:, :history], v.r_f[:, :history], v.W[:1],
             v.alpha, v.C0])
        definition_vars = flatten(
            [v.A_f[:, history:], v.A, v.c_f[:, history:],
             v.r_f[:, history:], v.S, v.L, v.W[1:]])

        if(cc.feasible_response):
            verifier_vars.extend(flatten(v.S_choice))
            definition_vars.extend(flatten(v.S_f))
        else:
            verifier_vars.extend(flatten(v.S_f))

        return verifier_vars, definition_vars

    def get_cegis_vars(
        self,
        cc: CegisConfig, c: ModelConfig, v: Variables
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        verifier_vars, definition_vars = super().get_cegis_vars(cc, c, v)
        assert isinstance(v, self.LinkVariables)
        if (c.calculate_qdel):
            # first_qdel[t][dt<t] is deterministic
            # first_qdel[t][dt>=t] is non-deterministic
            definition_vars.extend(flatten(
                [v.first_qdel[t][dt]
                 for t in range(c.T)
                 for dt in range(t)]))
            verifier_vars.extend(flatten(
                [v.first_qdel[t][dt]
                 for t in range(c.T)
                 for dt in range(t, c.T)]))
        if (c.beliefs):
            definition_vars.extend(flatten([
                v.min_c_lambda[:, 1:],
                v.bq_belief1, v.bq_belief2[:, 1:],
                v.recomputed_min_c_lambda
            ]))
            verifier_vars.extend(flatten([
                v.min_c_lambda[:, :1],
                v.bq_belief2[:, :1]
            ]))
        return verifier_vars, definition_vars

    @staticmethod
    def get_cex_df(
        counter_example: z3.ModelRef, v: Variables,
        vn: VariableNames, c: ModelConfig
    ) -> pd.DataFrame:
        assert isinstance(v, CBRDelayLink.LinkVariables)
        assert isinstance(c, CBRDelayLink.LinkModelConfig)

        df = BaseLink.get_cex_df(counter_example, v, vn, c)

        def _get_model_value(l):
            return get_model_value_list(counter_example, l)

        if (c.beliefs):
            for n in range(c.N):
                df[get_name_for_list(vn.min_c_lambda[n])] = _get_model_value(v.min_c_lambda[n])
                df[get_name_for_list(vn.bq_belief1[n])] = _get_model_value(v.bq_belief1[n])
                df[get_name_for_list(vn.bq_belief2[n])] = _get_model_value(v.bq_belief2[n])
                # df[get_name_for_list(vn.recomputed_min_c_lambda[n])] = _get_model_value(v.recomputed_min_c_lambda[n])
                # df[get_name_for_list(vn.under_utilized[n])] = _get_model_value(v.under_utilized[n])

        if(c.calculate_qdel):
            qdelay = []
            for t in range(c.T):
                this_value = c.T
                for dt in range(c.T):
                    value = counter_example.eval(v.first_qdel[t][dt])
                    bool_value = bool(value)
                    if(bool_value):
                        this_value = min(this_value, dt)
                        break
                qdelay.append(this_value)
            assert len(qdelay) == c.T
            df["first_qdel_t"] = np.array(qdelay).astype(float)

        return df.astype(float)