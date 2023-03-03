from typing import List, Tuple

import z3

from ccac.model import (ModelConfig, cca_paced, cwnd_rate_arrival,
                        epsilon_alpha, initial, loss_oracle, multi_flows,
                        relate_tot)
from ccac.variables import Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import flatten
from ccmatic.verifier import (BaseLink, app_limits_env, calculate_qbound_defs,
                              calculate_qbound_env, calculate_qdel_defs,
                              calculate_qdel_env, exceed_queue_defs,
                              fifo_service, initial_beliefs,
                              last_decrease_defs, monotone_defs, monotone_env,
                              update_beliefs)
from pyz3_utils.my_solver import MySolver


class IdealLink(BaseLink):

    @staticmethod
    def loss_deterministic(c: ModelConfig, s: MySolver, v: Variables):
        """
        If A[t] - L[t-1] - S[t] > buffer
            A[t] - L[t] - S[t] == buffer
        Else
            L[t] = L[t-1]
        """
        assert c.deterministic_loss
        assert c.buf_max == c.buf_min

        if c.buf_min is not None:
            s.add(v.A[0] - v.L[0] <= v.S[0] + c.buf_min)
            s.add(z3.Implies(z3.Or([v.L_f[n][0] > v.Ld_f[n][0] for n in range(c.N)]),
                             v.A[0] - v.L[0] == v.S[0] + c.buf_min))
        for t in range(1, c.T):
            if c.buf_min is None:  # no loss case
                s.add(v.L[t] == v.L[0])
            else:
                s.add(v.L[t] == z3.If(
                    v.A[t] - v.L[t-1] > v.S[t] + c.buf_min,
                    v.A[t] - v.S[t] - c.buf_min,
                    v.L[t-1]
                ))
                # s.add(z3.Implies(
                #     v.A[t] - v.L[t-1] > v.S[t] + c.buf_min,
                #     v.A[t] - v.L[t] == v.S[t] + c.buf_min))
                # s.add(z3.Implies(
                #     v.A[t] - v.L[t-1] <= v.S[t] + c.buf_min,
                #     v.L[t] == v.L[t-1]))
        # L[0] is still chosen non-deterministically in an unconstrained fashion.

        if(c.buf_min is None):
            # For multiflow, L_f is ideally non-deterministic
            # When there is inf buffer however, L_f should be deterministic
            for n in range(c.N):
                for t in range(1, c.T):
                    s.add(v.L_f[n][t] == v.L_f[n][0])
                # There shouldn't be any detected losses as well.
                s.add(v.Ld_f[n][0] == v.L_f[n][0])

    @staticmethod
    def service_defs(c: ModelConfig, s: MySolver, v: Variables):
        """
        S[t] = Min(S[t-1] + C, A[t]-L[t-1])
        """
        for t in range(c.T):
            for n in range(c.N):
                s.add(v.S_f[n][t] <= v.A_f[n][t] - v.L_f[n][t])

            if(t >= 1):
                r1 = v.S[t-1] + c.C
                r2 = v.A[t] - v.L[t-1]
                s.add(v.S[t] == z3.If(r1 < r2, r1, r2))

    def get_cegis_vars(
        self,
        cc: CegisConfig, c: ModelConfig, v: Variables
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        assert c.N == 1
        # When c.N > 1, the division of service between multiple flows would be in
        # control of verifier. This case is not considered, i.e., S_f is made def
        # var.

        # In the ideal link, the verifier only chooses initial conditions.
        history = cc.history

        verifier_vars = flatten(
            [v.A_f[:, :history], v.c_f[:, :history], v.r_f[:, :history],
             v.alpha, v.S_f[:, :history]])

        # No need for tokens and waste
        if(not cc.assumption_verifier):
            del v.C0
            del v.W

        assert c.loss_oracle

        definition_vars = flatten(
            [v.A_f[:, history:], v.A, v.c_f[:, history:],
             v.r_f[:, history:], v.S, v.L, v.S_f[:, history:]])

        if(c.calculate_qdel):
            # qdel[t][dt<t] is deterministic
            # qdel[t][dt>=t] is non-deterministic
            verifier_vars.extend(flatten(
                [v.qdel[t][dt] for t in range(c.T) for dt in range(t, c.T)]))
        if(c.calculate_qbound):
            # qbound[t][dt<=t] is deterministic
            # qbound[t][dt>t] is non deterministic
            verifier_vars.extend(flatten(
                [v.qbound[t][dt] for t in range(c.T) for dt in range(t+1, c.T)]))
            definition_vars.extend(flatten(v.exceed_queue_f))
            definition_vars.extend(flatten(v.last_decrease_f))

        if(c.calculate_qdel):
            definition_vars.extend(flatten(
                [v.qdel[t][dt] for t in range(c.T) for dt in range(t)]))
        if(c.calculate_qbound):
            definition_vars.extend(flatten(
                [v.qbound[t][dt] for t in range(c.T) for dt in range(t+1)]))

        if(c.buf_min is None):
            # No loss
            verifier_vars.extend(flatten([v.L_f, v.Ld_f]))
        elif(c.deterministic_loss):
            # Determinisitic loss
            assert c.loss_oracle
            verifier_vars.extend(flatten([v.L_f[:, :1], v.Ld_f[:, :c.R]]))
            definition_vars.extend(flatten([v.L_f[:, 1:], v.Ld_f[:, c.R:]]))
        else:
            assert False

        if(isinstance(c.buf_min, z3.ExprRef)):
            verifier_vars.append(c.buf_min)

        if(c.mode_switch):
            definition_vars.extend(flatten(v.mode_f[:, 1:]))
            verifier_vars.extend(flatten(v.mode_f[:, :1]))

        if(c.beliefs):
            definition_vars.extend(flatten(
                [v.min_c[:, 1:], v.max_c[:, 1:],
                 v.min_qdel, v.max_qdel]))
            verifier_vars.extend(flatten(
                [v.min_c[:, :1], v.max_c[:, :1]]))
            if(c.buf_min is not None and c.beliefs_use_buffer):
                definition_vars.extend(flatten(
                    v.min_buffer[:, 1:]))
                verifier_vars.extend(flatten(
                    v.min_buffer[:, :1]))
                if(c.beliefs_use_max_buffer):
                    definition_vars.extend(flatten(
                        v.max_buffer[:, 1:]))
                    verifier_vars.extend(flatten(
                        v.max_buffer[:, :1]))
            verifier_vars.extend(flatten(v.start_state_f))

        if(c.app_limited):
            verifier_vars.extend(flatten(v.app_limits))
            verifier_vars.append(v.app_rate)
            if(c.app_fixed_avg_rate and c.beliefs):
                verifier_vars.extend(flatten(
                    [v.max_app_rate[:, :1], v.min_app_rate[:, :1]]))
                definition_vars.extend(flatten(
                    [v.max_app_rate[:, 1:], v.min_app_rate[:, 1:]]))

        return verifier_vars, definition_vars

    def setup_definitions(self,
                          c: ModelConfig, v: Variables):
        s = MySolver()
        s.warn_undeclared = False

        assert c.loss_oracle
        assert c.deterministic_loss

        monotone_defs(c, s, v)  # Def only. Constraint not needed.
        initial(c, s, v)  # Either definition vars or verifier vars.
        # Keep as definitions for convenience.
        relate_tot(c, s, v)  # Definitions required to compute tot variables.
        IdealLink.loss_deterministic(c, s, v)  # Def to compute loss.
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
        if(c.beliefs):
            update_beliefs(c, s, v)
        IdealLink.service_defs(c, s, v)
        return s

    def setup_environment(self, c: ModelConfig, v: Variables):
        s = MySolver()
        s.warn_undeclared = False

        assert c.loss_oracle
        assert c.deterministic_loss

        monotone_env(c, s, v)
        if(c.calculate_qdel):
            calculate_qdel_env(c, s, v)  # How to pick initial qdels
            # non-deterministically.
        if(c.calculate_qbound):
            calculate_qbound_env(c, s, v)  # How to pick initial qbounds
            # non-deterministically.
        if(c.N > 1):
            raise NotImplementedError
            multi_flows(c, s, v)  # Flows should be serviced fairly
            fifo_service(c, s, v)
        epsilon_alpha(c, s, v)  # Verifier only

        # Shouldn't be any loss at t0 otherwise cwnd is high and q is still 0.
        # s.add(v.L_f[0][0] == 0)

        # Remove periodicity, as generator overfits and produces monotonic CCAs.
        # make_periodic(c, s, v, c.R + c.D)

        # Avoid weird cases where single packet is larger than BDP.
        s.add(v.alpha < max(0.2, (c.C * c.R)/5))  # Verifier only

        # Buffer should at least have one packet
        if(isinstance(c.buf_min, z3.ExprRef)):
            s.add(c.buf_min > v.alpha)

            # Buffer taken from discrete choices
            # s.add(z3.Or(c.buf_min == c.C * (c.R + c.D),
            #             c.buf_min == 0.1 * c.C * (c.R + c.D)))
        if(c.beliefs):
            initial_beliefs(c, s, v)
        if(c.app_limited):
            app_limits_env(c, s, v)
        return s
