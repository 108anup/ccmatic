import z3

from ccac.config import ModelConfig
from ccac.model import cca_paced, cwnd_rate_arrival, initial, loss_oracle, relate_tot
from ccac.variables import Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten
from ccmatic.verifier import BaseLink, calculate_qbound_defs, calculate_qdel_defs, exceed_queue_defs, last_decrease_defs, loss_deterministic, monotone_defs, update_beliefs
from cegis.util import z3_max
from pyz3_utils.my_solver import MySolver

from typing import Tuple, List


def update_min_c_lambda(c: ModelConfig, s: MySolver, v: Variables):
    # Update min_c_lambda, i.e., belief of min_c using sending rate instead
    # of ack rate.
    for n in range(c.N):
        for t in range(1, c.T):

            utilized_t = [None for _ in range(t)]
            # utilized_cummulative = [None for _ in range(t)]
            under_utilized_cummulative = [None for _ in range(t)]
            for st in range(t-1, -1, -1):
                high_delay = z3.Not(z3.Or(*[v.qdel[st+1][dt]
                                            for dt in range(c.D+1)]))
                loss = v.Ld_f[n][st+1] - v.Ld_f[n][st] > 0
                if(c.D == 0):
                    assert c.loss_oracle
                    loss = v.L_f[n][st+1] - v.L_f[n][st] > 0
                recvd_new_pkts = True
                sent_new_pkts = v.A_f[n][st+1] - v.A_f[n][st] > 0
                this_utilized = z3.Or(
                    z3.And(high_delay, sent_new_pkts, recvd_new_pkts),
                    loss)
                utilized_t[st] = this_utilized
                if(st + 1 == t):
                    # utilized_cummulative[st] = this_utilized
                    under_utilized_cummulative[st] = z3.Not(this_utilized)
                else:
                    # assert utilized_cummulative[st+1] is not None
                    # utilized_cummulative[st] = z3.And(utilized_cummulative[st+1], this_utilized)
                    assert under_utilized_cummulative[st+1] is not None
                    under_utilized_cummulative[st] = z3.And(under_utilized_cummulative[st+1], z3.Not(this_utilized))

            base_minc = v.min_c_lambda[n][t-1]
            overall_minc = base_minc
            # RTT >= 1 at the very least, and so max et is t-1.
            for et in range(t-1, 0, -1):
                # We only use this et if et = t-RTT(t)
                # et = t-RTT(t) iff A[et] <= S[t] and A[et+1] > S[t]
                correct_et = z3.And(
                    v.A_f[n][et] <= v.S_f[n][t],
                    v.A_f[n][et+1] > v.S_f[n][t])
                for st in range(et-1, -1, -1):
                    window = et - st
                    assert window > 0

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

            s.add(v.min_c_lambda[n][t] == overall_minc)
            # s.add(v.min_c_lambda[n][t] == z3_max(overall_minc, v.min_c[n][t]))


class CBRDelayLink(BaseLink):

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
    def update_beliefs(c: ModelConfig, s: MySolver, v: Variables):
        update_beliefs(c, s, v)
        update_min_c_lambda(c, s, v)

    def setup_definitions(self, c: ModelConfig, v: Variables):
        s = MySolver()
        s.warn_undeclared = False

        monotone_defs(c, s, v)  # Def only. Constraint not needed.
        initial(c, s, v)  # Either definition vars or verifier vars.
        # Keep as definitions for convenience.
        relate_tot(c, s, v)  # Definitions required to compute tot variables.
        if(c.deterministic_loss):
            loss_deterministic(c, s, v)  # Def to compute loss.
        if(c.loss_oracle):
            loss_oracle(c, s, v)  # Defs to compute loss detected.
        if(c.calculate_qdel):
            calculate_qdel_defs(c, s, v)  # Defs to compute qdel.
        if(c.calculate_qbound):
            calculate_qbound_defs(c, s, v)  # Defs to compute qbound.
            last_decrease_defs(c, s, v)
            exceed_queue_defs(c, s, v)
        cwnd_rate_arrival(c, s, v)  # Defs to compute arrival.
        if(c.cca == "paced"):
            cca_paced(c, s, v)  # Defs to compute rate.
        assert not c.feasible_response
        if(c.beliefs):
            self.update_beliefs(c, s, v)
        self.waste_defs(c, s, v)

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
