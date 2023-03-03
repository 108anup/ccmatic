import z3

from ccac.config import ModelConfig
from ccac.model import cca_paced, cwnd_rate_arrival, initial, loss_oracle, relate_tot
from ccac.variables import Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten
from ccmatic.verifier import BaseLink, calculate_qbound_defs, calculate_qdel_defs, exceed_queue_defs, last_decrease_defs, loss_deterministic, monotone_defs, update_beliefs
from pyz3_utils.my_solver import MySolver

from typing import Tuple, List


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
        # raise NotImplementedError

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
