import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import z3

from ccac.model import (ModelConfig, cca_paced, cwnd_rate_arrival,
                        epsilon_alpha, initial, loss_detected, loss_oracle,
                        make_solver, multi_flows, relate_tot)
from ccac.variables import VariableNames, Variables
from ccmatic.cegis import CegisConfig
from ccmatic.common import (flatten, get_name_for_list, get_renamed_vars,
                            get_val_list)
from cegis import NAME_TEMPLATE
from cegis.util import get_model_value_list, get_raw_value, z3_max, z3_max_list, z3_min, z3_min_list
from pyz3_utils.binary_search import BinarySearch
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver
from pyz3_utils.small_denom import find_small_denom_soln

logger = logging.getLogger('verifier')
GlobalConfig().default_logger_setup(logger)


# * First write concrete $\exists\forall$ formula.
# * Decide classification of variables:
# 1. If variable has $\exists$ quantifier then it is a generator variable. If a
#    variable has $\forall$ quantifier then it can be a verifier variable or a
#    definition variable.
# 2. Technically, we can keep all $\forall$ variables as verifier variable.
#    Though, it is helpful to convert verifier variables to definition variables
#    (reduces CEGIS iterations, provides more information to generator for why
#    or why not a particular candidate works). This can be done iff the variable
#    can be computed as a deterministic function of other generator and verifier
#    variables. Ideally convert as many verifier variables to definition
#    variables as possible (if a variable needs non-deterministic choice, then
#    must keep it as verifier variable).
# * Decide classification of assertions:
# 1. If assertion only contains generator variables, then keep as search
#    constraint.
# 2. If assertion only contains verifier variables, then can keep as
#    specification or definition. Can choose based on convenience, e.g.,
#    depending on if other similar constraints need to be in definition or
#    specification.
# 3. Assertions that only involve definition variables are meant for sanity
#    only, they are not needed / should not be needed.
# 4. If assertion contains a definition variable and at least one other
#    generator or verifier variable, then this should be put in definitions iff
#    this is needed to compute the value of the definition variable in the
#    constraint. If we don't add a required constraint to definition then the
#    candidate solution can repeat, but this is very easy to spot/debug.
# 4. Err on the side of keeping assertions in specification, if we erroneously
#    add assertion into definitions, then a solution maybe erroneously pruned.
#    This is hard to debug.
# * Time per iteration concerns
# 1. For definition variables that solely depend on verifier variables. Can keep
#    those in verifier variables, and corresponding assertions in specification.


@dataclass
class SteadyStateVariable:
    name: str
    initial: z3.ArithRef
    final: z3.ArithRef
    lo: Optional[z3.ArithRef]
    hi: Optional[z3.ArithRef]

    def init_outside(self) -> z3.BoolRef:
        ret = z3.Not(self.init_inside())
        assert isinstance(ret, z3.BoolRef)
        return ret

    def final_outside(self) -> z3.BoolRef:
        ret = z3.Not(self.final_inside())
        assert isinstance(ret, z3.BoolRef)
        return ret

    def init_inside(self) -> z3.BoolRef:
        assert self.lo is not None or self.hi is not None
        init_inside_list = []
        if(self.lo is not None):
            init_inside_list.append(self.lo <= self.initial)
        if(self.hi is not None):
            init_inside_list.append(self.initial <= self.hi)
        ret = z3.And(init_inside_list)
        assert isinstance(ret, z3.BoolRef)
        return ret

    def final_inside(self) -> z3.BoolRef:
        assert self.lo is not None or self.hi is not None
        final_inside_list = []
        if(self.lo is not None):
            final_inside_list.append(self.lo <= self.final)
        if(self.hi is not None):
            final_inside_list.append(self.final <= self.hi)
        ret = z3.And(final_inside_list)
        assert isinstance(ret, z3.BoolRef)
        return ret

    def does_not_degrage(self) -> z3.BoolRef:
        assert self.lo is not None or self.hi is not None
        does_not_degrade_list = [
            z3.Implies(self.init_inside(), self.final_inside())]
        if(self.hi is None):
            does_not_degrade_list.append(
                z3.Implies(self.initial < self.lo,
                           self.final >= self.initial))
        elif(self.lo is None):
            does_not_degrade_list.append(
                z3.Implies(self.initial > self.hi,
                           self.final <= self.initial))
        else:
            does_not_degrade_list.append(
                z3.Implies(self.initial < self.lo,
                           z3.And(self.final >= self.initial,
                                  self.final <= self.hi)))
            does_not_degrade_list.append(
                z3.Implies(self.initial > self.hi,
                           z3.And(self.final <= self.initial,
                                  self.final >= self.lo)))
        ret = z3.And(does_not_degrade_list)
        assert isinstance(ret, z3.BoolRef)
        return ret

    def strictly_improves(
        self, movement_mult: Optional[Union[float, z3.ExprRef]] = None
    ) -> z3.BoolRef:
        assert self.lo is not None or self.hi is not None
        if(movement_mult is None):
            movement_mult = 1
        strictly_improves_list = []
        if(self.hi is None):
            strictly_improves_list.append(
                z3.Implies(self.initial < self.lo,
                           self.final > movement_mult * self.initial))
        elif(self.lo is None):
            strictly_improves_list.append(
                z3.Implies(self.initial > self.hi,
                           self.final * movement_mult < self.initial))
        else:
            strictly_improves_list.append(
                z3.Implies(self.initial < self.lo,
                           z3.And(self.final > movement_mult * self.initial,
                                  self.final <= self.hi)))
            strictly_improves_list.append(
                z3.Implies(self.initial > self.hi,
                           z3.And(self.final * movement_mult < self.initial,
                                  self.final >= self.lo)))
        ret = z3.And(strictly_improves_list)
        assert isinstance(ret, z3.BoolRef)
        return ret

    def strictly_improves_add(
        self, movement_add: Optional[Union[float, z3.ExprRef]] = None
    ) -> z3.BoolRef:
        assert self.lo is not None or self.hi is not None
        if(movement_add is None):
            movement_add = 0
        strictly_improves_list = []
        if(self.hi is None):
            strictly_improves_list.append(
                z3.Implies(self.initial < self.lo,
                           self.final > movement_add + self.initial))
        elif(self.lo is None):
            strictly_improves_list.append(
                z3.Implies(self.initial > self.hi,
                           self.final + movement_add < self.initial))
        else:
            strictly_improves_list.append(
                z3.Implies(self.initial < self.lo,
                           z3.And(self.final > movement_add + self.initial,
                                  self.final <= self.hi)))
            strictly_improves_list.append(
                z3.Implies(self.initial > self.hi,
                           z3.And(self.final + movement_add < self.initial,
                                  self.final >= self.lo)))
        ret = z3.And(strictly_improves_list)
        assert isinstance(ret, z3.BoolRef)
        return ret


def monotone_env(c, s, v):
    for t in range(1, c.T):
        for n in range(c.N):
            # Definition only (assertion really not needed). Keeping in defs.
            # In reality since arrival is max of prev and new value, this
            # constriant should never be broken anyway.
            # s.add(v.A_f[n][t] >= v.A_f[n][t - 1])

            # Verifier only. Keep part of specification.
            s.add(v.S_f[n][t] >= v.S_f[n][t - 1])

            # For loss detected and loss. If non-deterministic then, part of
            # specification, otherwise not needed in defs so keep in
            # specification always.
            s.add(v.Ld_f[n][t] >= v.Ld_f[n][t - 1])
            s.add(v.L_f[n][t] >= v.L_f[n][t - 1])

            # For non-deterministic loss, loss may not be feasible for new
            # arrival. For deterministic loss, not needed in defs.
            s.add(
                v.A_f[n][t] - v.L_f[n][t] >= v.A_f[n][t - 1] - v.L_f[n][t - 1])

        # Verifier only. Keep part of specification.
        if(hasattr(v, 'W')):
            s.add(v.W[t] >= v.W[t - 1])
            if(hasattr(v, 'C0')):
                s.add(v.C0 + c.C * t - v.W[t] >= v.C0 + c.C * (t-1) - v.W[t-1])


def monotone_defs(c, s, v):
    for t in range(1, c.T):
        for n in range(c.N):
            s.add(v.A_f[n][t] >= v.A_f[n][t - 1])
            # s.add(v.Ld_f[n][t] >= v.Ld_f[n][t - 1])
            # s.add(v.S_f[n][t] >= v.S_f[n][t - 1])
            # s.add(v.L_f[n][t] >= v.L_f[n][t - 1])

            # s.add(
            #     v.A_f[n][t] - v.L_f[n][t] >= v.A_f[n][t - 1] - v.L_f[n][t - 1])
        # s.add(v.W[t] >= v.W[t - 1])
        # s.add(v.C0 + c.C * t - v.W[t] >= v.C0 + c.C * (t-1) - v.W[t-1])


def service_waste(c: ModelConfig, s: MySolver, v: Variables):
    for t in range(c.T):
        for n in range(c.N):
            s.add(v.S_f[n][t] <= v.A_f[n][t] - v.L_f[n][t])

        s.add(v.S[t] <= v.C0 + c.C * t - v.W[t])
        if t >= c.D:
            s.add(v.C0 + c.C * (t - c.D) - v.W[t - c.D] <= v.S[t])
        else:
            # The constraint is the most slack when black line is steepest. So
            # we'll say there was no wastage when t < 0
            s.add(v.C0 + c.C * (t - c.D) - v.W[0] <= v.S[t])

        if c.compose:
            if t > 0:
                s.add(
                    z3.Implies(v.W[t] > v.W[t - 1],
                               v.A[t] - v.L[t] <= v.C0 + c.C * t - v.W[t]))
        else:
            if t > 0:
                s.add(
                    z3.Implies(v.W[t] > v.W[t - 1],
                               v.A[t] - v.L[t] <= v.S[t] + v.epsilon))


def service_choice(c: ModelConfig, s: MySolver, v: Variables):
    assert c.N == 1
    for t in range(c.T):
        S_lower = v.C0 + c.C * (t - c.D) - v.W[0]
        if t >= c.D:
            S_lower = v.C0 + c.C * (t - c.D) - v.W[t - c.D]
        feasible_choice = z3.And(
            v.S_choice[t] <= v.A[t] - v.L[t],
            v.S_choice[t] <= v.C0 + c.C * t - v.W[t],
            v.S_choice[t] >= S_lower)
        s.add(z3.Implies(feasible_choice, v.S[t] == v.S_choice[t]))
        s.add(z3.Implies(z3.Not(feasible_choice), v.S[t] == S_lower))


def app_limits_env(c: ModelConfig, s: MySolver, v: Variables):
    assert c.app_limited

    for n in range(c.N):
        s.add(v.A_f[n][0] <= v.app_limits[n][0])
        for t in range(1, c.T):
            s.add(v.app_limits[n][t] >= v.app_limits[n][t-1])

    if(c.app_fixed_avg_rate):
        # Set app rate
        if(c.app_rate):
            s.add(v.app_rate == c.app_rate)
        else:
            s.add(v.app_rate >= 0.1 * c.C)

        # Enfore app rate and burst
        burst = c.app_burst_factor * v.app_rate * 1
        # here one is the time quanta            ^^^

        def alpha(t):
            return burst + v.app_rate * t

        # From https://leboudec.github.io/netcal/
        for n in range(c.N):
            for t in range(c.T):
                for st in range(t+1):
                    s.add(v.app_limits[n][t] <= v.app_limits[n][st] + alpha(t-st))
                if(t >= 1):
                    s.add(v.app_limits[n][t] >= v.app_limits[n][t-1] + v.app_rate * 1)


def update_bandwidth_beliefs_with_timeout(
        c: ModelConfig, s: MySolver, v: Variables):
    assert(c.beliefs)

    # For a timeout, we need enough history. So we can't trigger timeouts at
    # t=1.
    for n in range(c.N):
        s.add(v.start_state_f[n] == 0)

    # Bandwidth beliefs (C)
    cycle = c.T-1  # as first time is chosen by verifier.
    reset_minc_time = int((cycle-1)/2)
    reset_maxc_time = int((cycle-1))
    reset_minc_time = c.T-1
    reset_maxc_time = c.T-1
    for n in range(c.N):
        for et in range(1, c.T):
            # Compute utilization stats. Useful for updating {min, max}_c

            # TODO: qdelay high or loss signals tell about network conditions at
            # earlier time than when the CCA observed them. This lag perhaps
            # needs to be incorporated.
            utilized_t = [None for _ in range(et)]
            utilized_cummulative = [None for _ in range(et)]
            for st in range(et-1, -1, -1):
                # Encodes that utilization must have been high if loss happened
                # or if queing delay was more than D

                # All of qdel might be False. So to check high delay we just
                # check Not of low delay.
                high_delay = z3.Not(z3.Or(*[v.qdel[st+1][dt]
                                            for dt in range(c.D+1)]))
                # Wrong:
                # high_delay = z3.Or(*[v.qdel[st+1][dt]
                #                      for dt in range(c.D+1, c.T)])
                loss = v.Ld_f[n][st+1] - v.Ld_f[n][st] > 0
                this_utilized = z3.Or(loss, high_delay)
                utilized_t[st] = this_utilized
                if(st + 1 == et):
                    utilized_cummulative[st] = this_utilized
                else:
                    assert utilized_cummulative[st+1] is not None
                    utilized_cummulative[st] = z3.And(utilized_cummulative[st+1], this_utilized)

            # If the belief recently changed then it must be consistent** This
            # is not true when maxc is small and we still see qdel measurements
            # due to initial conditions. But in this case beliefs become invalid
            # quickly.
            # minc_changed_list = []
            # maxc_changed_list = []
            rate_above_minc_list = []
            for st in range(1, et):
                # minc_changed_list.append(v.min_c[n][st] > v.min_c[n][st-1])
                # maxc_changed_list.append(v.max_c[n][st] < v.max_c[n][st-1])
                rate_above_minc_list.append(v.r_f[n][st] > v.min_c[n][st])
            # minc_changed = z3.Or(*minc_changed_list)
            # maxc_changed = z3.Or(*maxc_changed_list)
            minc_changed = v.min_c[n][et-1] > v.min_c[n][0]
            maxc_changed = v.max_c[n][et-1] < v.max_c[n][0]

            minc_changed_significantly = v.min_c[n][et-1] > c.maxc_minc_change_mult * v.min_c[n][0]
            maxc_changed_significantly = v.max_c[n][et-1] * c.maxc_minc_change_mult < v.max_c[n][0]
            # none_changed_significantly = z3.Not(z3.Or(minc_changed_significantly, maxc_changed_significantly))
            # rate_above_minc = z3.Or(*rate_above_minc_list)

            base_lower = v.min_c[n][et-1]
            base_upper = v.max_c[n][et-1]
            reset_upper = z3_max(v.min_c[n][et]*c.maxc_minc_change_mult, 3/2*v.max_c[n][et-1])
            beliefs_close_mult = base_upper <= base_lower * c.min_maxc_minc_gap_mult
            # decent_utilization = z3.Sum(utilized_t) >= 2

            """
            Time out the beliefs every cycle time.
            """
            if(c.fix_stale__min_c):
                # Timeout lower bound if we are in the appropriate cycle in the
                # state machine and we have seen utilization recently.
                # TODO: only need to look at utilized happening if we were sending at or below minc.
                # This again requires more nuanced book keeping of the sequence of events.

                # Do timeout if:
                # It is allowed by state
                # AND, if either none changed, or other changed and brought beliefs close.
                timeout_allowed_by_state = z3.Or(v.start_state_f[n] + et == reset_minc_time,
                                                 v.start_state_f[n] + et == reset_minc_time + cycle)
                other_came_close = z3.And(
                    beliefs_close_mult, maxc_changed)
                # timeout_lower = z3.And(timeout_allowed_by_state,
                #                        # z3.Or(utilized_t),
                #                        # decent_utilization,
                #                        # z3.Not(rate_above_minc)
                #                        z3.Or(none_changed_significantly, other_came_close),
                #                        z3.Not(minc_changed),
                #                        # z3.Not(maxc_changed),
                #                        )
                timeout_lower = z3.If(
                    z3.Or(minc_changed, z3.Not(timeout_allowed_by_state)), False,
                    z3.Or(other_came_close, z3.Not(maxc_changed_significantly))
                )
                base_lower = z3.If(timeout_lower, 0, base_lower)

            if(c.fix_stale__max_c):
                # Timeout upper bound if we are in the appropraite cycle in the
                # state machine and we haven't recently utilized the link.
                timeout_allowed_by_state = z3.Or(v.start_state_f[n] + et == reset_maxc_time,
                                                 v.start_state_f[n] + et == reset_maxc_time + cycle)
                other_came_close = z3.And(
                    beliefs_close_mult, minc_changed)
                # timeout_upper = z3.And(timeout_allowed_by_state,
                #                        # z3.Not(decent_utilization),
                #                        # z3.Not(z3.Or(utilized_t)),
                #                        # z3.Not(beliefs_close),
                #                        z3.Or(none_changed_significantly, other_came_close),
                #                        # z3.Not(minc_changed),
                #                        z3.Not(maxc_changed),
                #                        )
                timeout_upper = z3.If(
                    z3.Or(maxc_changed, z3.Not(timeout_allowed_by_state)), False,
                    z3.Or(other_came_close, z3.Not(minc_changed_significantly))
                )
                """
                Philosophy: we consider beliefs invalid when they are close
                enough, so that it does not take arbitrarily long to reset them.

                Only timing out out when no beleifs change is fine for
                invariant. But then we risk indefinitely not timing out maxc
                when minc only improves slightly: this happens when max is stale
                maxc < 2 * minc and cca is sending at or below minc. jitter
                causes ackrate to be above minc and slightly improves it. this
                can happen indefinitely.

                TODO: see if we can avoid timing out minc when maxc changes
                slightly. If maxc changes slightly, chances are that minc even
                using good measurements is a good bound, so unnecessarily timing
                it out might still be fine.

                We can timout upper when minc changed significantly. it may
                never come close anyway. We can timeout lower when maxc comes
                close, maxc changing slightly does not mean that the verifier is
                fooling us...
                """

                # timeout_upper_signal = z3.Or(v.start_state_f[n] + et == reset_maxc_time,
                #                              v.start_state_f[n] + et == reset_maxc_time + cycle)
                # beliefs_close = v.max_c[n][et-1] - v.min_c[n][et-1] <= 10
                # beliefs_did_not_change = z3.Not(
                #     z3.Or(minc_changed, maxc_changed))
                # timeout_upper = z3.And(
                #     timeout_upper_signal,
                #     z3.Or(z3.And(beliefs_close, z3.Not(z3.Or(utilized_t))),
                #           beliefs_did_not_change))
                base_upper = z3.If(timeout_upper, reset_upper, base_upper)

            """
            In summary C >= r * n/(n+1) always, if we additionally know that
            sending rate is higher than C then, C <= r * n/(n-1).
            """
            overall_lower = base_lower
            overall_upper = base_upper
            for st in range(et-1, -1, -1):
                window = et - st
                measured_c = (v.S_f[n][et] - v.S_f[n][st]) / (window)

                # LOWER
                this_lower = measured_c * window / (window + 1)
                overall_lower = z3_max(overall_lower, this_lower)

                # UPPER
                if(window - 1 > 0):
                    this_upper = z3.If(utilized_cummulative[st], measured_c * window / (window - 1), base_upper)
                    assert isinstance(this_upper, z3.ArithRef)
                    overall_upper = z3_min(overall_upper, this_upper)

            # max_c_next = overall_upper
            # if(c.fix_stale__max_c):
            #     TIME_BETWEEN_PROBES = 5
            #     if(et-TIME_BETWEEN_PROBES >= 0):
            #         t_start = et-TIME_BETWEEN_PROBES
            #         # max_c_updated_recently = z3.Or(*[v.max_c[n][t+1] < v.max_c[n][t]
            #         #                                  for t in range(t_start, et-1)])
            #         utilized_recently = z3.Or(*[utilized_t[st]
            #                                     for st in range(t_start, et)])
            #         # import ipdb; ipdb.set_trace()
            #         # overall_upper >= v.max_c[n][et-1]),
            #         max_c_next = z3.If(z3.Not(utilized_recently),
            #                            2 * v.max_c[n][et-1], overall_upper)

            s.add(v.min_c[n][et] == overall_lower)
            s.add(v.max_c[n][et] == overall_upper)


def update_bandwidth_beliefs_invalidation(
        c: ModelConfig, s: MySolver, v: Variables):
    assert(c.beliefs)

    # Bandwidth beliefs (C)
    for n in range(c.N):
        for et in range(1, c.T):
            # Compute utilization stats. Useful for updating {min, max}_c

            # TODO: qdelay high or loss signals tell about network conditions at
            # earlier time than when the CCA observed them. This lag perhaps
            # needs to be incorporated.
            utilized_t = [None for _ in range(et)]
            utilized_cummulative = [None for _ in range(et)]
            for st in range(et-1, -1, -1):
                # Encodes that utilization must have been high if loss happened
                # or if queing delay was more than D

                # All of qdel might be False. So to check high delay we just
                # check Not of low delay.
                high_delay = z3.Not(z3.Or(*[v.qdel[st+1][dt]
                                            for dt in range(c.D+1)]))
                # Wrong:
                # high_delay = z3.Or(*[v.qdel[st+1][dt]
                #                      for dt in range(c.D+1, c.T)])
                loss = v.Ld_f[n][st+1] - v.Ld_f[n][st] > 0
                this_utilized = z3.Or(loss, high_delay)
                utilized_t[st] = this_utilized
                if(st + 1 == et):
                    utilized_cummulative[st] = this_utilized
                else:
                    assert utilized_cummulative[st+1] is not None
                    utilized_cummulative[st] = z3.And(utilized_cummulative[st+1], this_utilized)

            """
            In summary C >= r * n/(n+1) always, if we additionally know that
            sending rate is higher than C then, C <= r * n/(n-1).
            """
            base_upper = v.max_c[n][et-1]
            overall_lower = v.min_c[n][et-1]
            overall_upper = v.max_c[n][et-1]
            recomputed_lower = 0
            for st in range(et-1, -1, -1):
                window = et - st
                measured_c = (v.S_f[n][et] - v.S_f[n][st]) / (window)

                # LOWER
                this_lower = measured_c * window / (window + 1)
                overall_lower = z3_max(overall_lower, this_lower)
                overall_lower = z3_max(overall_lower, this_lower)
                recomputed_lower = z3_max(recomputed_lower, this_lower)

                # UPPER
                if(window - 1 > 0):
                    this_upper = z3.If(
                        utilized_cummulative[st],
                        measured_c * window / (window - 1),
                        base_upper)
                    # We could have replaced base_upper with overall_upper, but
                    # that creates a complicated encoding.
                    assert isinstance(this_upper, z3.ArithRef)
                    overall_upper = z3_min(overall_upper, this_upper)

            # if maxc changed, then update minc
            # if minc changed, then update maxc
            # when things become invalid, both should not change.
            s.add(
                v.min_c[n][et] == z3.If(
                    z3.And(overall_upper < overall_lower,
                           overall_upper < v.max_c[n][et-1]),
                    recomputed_lower, overall_lower))
            reset_upper = z3_max(v.min_c[n][et], 3/2 * v.max_c[n][et-1])
            s.add(
                v.max_c[n][et] == z3.If(
                    z3.And(overall_upper < overall_lower,
                           overall_lower > v.min_c[n][et-1]),
                    reset_upper, overall_upper))


def update_bandwidth_beliefs_invalidation_and_timeout(
        c: ModelConfig, s: MySolver, v: Variables):
    assert(c.beliefs)

    # Bandwidth beliefs (C)
    for n in range(c.N):
        for et in range(1, c.T):
            # Compute utilization stats. Useful for updating {min, max}_c

            # TODO: qdelay high or loss signals tell about network conditions at
            # earlier time than when the CCA observed them. This lag perhaps
            # needs to be incorporated.
            utilized_t = [None for _ in range(et)]
            utilized_cummulative = [None for _ in range(et)]
            for st in range(et-1, -1, -1):
                # Encodes that utilization must have been high if loss happened
                # or if queing delay was more than D

                # All of qdel might be False. So to check high delay we just
                # check Not of low delay.
                high_delay = z3.Not(z3.Or(*[v.qdel[st+1][dt]
                                            for dt in range(c.D+1)]))
                # Wrong:
                # high_delay = z3.Or(*[v.qdel[st+1][dt]
                #                      for dt in range(c.D+1, c.T)])
                loss = v.Ld_f[n][st+1] - v.Ld_f[n][st] > 0
                if(c.D == 0):
                    assert c.loss_oracle
                    loss = v.L_f[n][st+1] - v.L_f[n][st] > 0

                # We can only count qdelay when packets actually arrived.
                # Otherwise the qdelay is stale.
                # recvd_new_pkts = v.S_f[n][st+1] > v.S_f[n][st]
                recvd_new_pkts = True

                # CCAC can somehow give high utilization signals even when cwnd is low.
                # This basically happens when no new pkts are sent due to low cwnd.
                # To avoid underestimating max_c, we update it only when we sent new pkts.

                # Updated reason: A delay of high can happen when there are last
                # few pkts left in the queue. Thus the ack rate will be low. If
                # the CCA sent pkts R + D time ago, this qdelay can only be high
                # if this quue is big, if the queue is small, then the recently
                # sent pkts will show low qdelay.

                # Based on this, ideally we should be checking A_f[n][st+1-D]
                # > A_f[n][st-D]. But this works out. For ideal link D = 0, so
                # this is fine. For jittery link, since we always look at
                # windows >=D+1, the constraint for previous batch would ensure
                # constraint for this batch (we either sent new packets in
                # previous batch or caused loss, loss can only happen if we sent
                # new pkts.). We only really care about the constraint for the
                # last batch.
                sent_new_pkts = v.A_f[n][st+1] - v.A_f[n][st] > 0

                # We assume the link is utilized if we did not recv any packets.
                # If we don't do this, the network sends 0, then 2 * CD, then 0 pkts and so on.
                # This causes us to never have long enough utilized window (hence we can't improve max_c).
                # this_utilized = z3.And(v.S_f[n][st+1] > v.S_f[n][st],
                #                        z3.Or(loss, high_delay))
                this_utilized = z3.Or(
                    z3.And(high_delay, sent_new_pkts, recvd_new_pkts),
                    loss)
                utilized_t[st] = this_utilized
                if(st + 1 == et):
                    utilized_cummulative[st] = this_utilized
                else:
                    assert utilized_cummulative[st+1] is not None
                    utilized_cummulative[st] = z3.And(utilized_cummulative[st+1], this_utilized)

            """
            In summary C >= r * n/(n+1) always, if we additionally know that
            sending rate is higher than C then, C <= r * n/(n-1).
            """
            base_maxc = v.max_c[n][et-1]
            overall_minc = v.min_c[n][et-1]
            overall_maxc = v.max_c[n][et-1]
            recomputed_minc = 0
            for st in range(et-1, -1, -1):
                window = et - st
                measured_c = (v.S_f[n][et] - v.S_f[n][st]) / (window)

                # LOWER
                this_lower = measured_c * window / (window + c.D)
                overall_minc = z3_max(overall_minc, this_lower)
                recomputed_minc = z3_max(recomputed_minc, this_lower)

                # UPPER
                """
                For ideal link, D = 0, but still we want ot measure ack rate
                over at least windows of length 2. This is because initial
                conditions can provide utilized signals for 1 timeunit while
                still giving low ack rate.

                # if(window - max(1, c.D) > 0):

                Above issue if fixed by ensuring loss detected happens then
                buffer must be full and also using L_f instead of Ld_f, i.e., if
                loss happened then we must have sent pkts and service must have
                been high. This needs to change when we do loss detection.
                """
                if(window - c.D > 0):
                    this_upper = z3.If(
                        utilized_cummulative[st],
                        measured_c * window / (window - c.D),
                        base_maxc)
                    # We could have replaced base_upper with overall_upper, but
                    # that creates a complicated encoding.
                    assert isinstance(this_upper, z3.ArithRef)
                    overall_maxc = z3_min(overall_maxc, this_upper)

            """
            If self changed, don't recompute
            If other came close do recompute.

            On timeout, if other did not change significantly, then recompute.
            """
            timeout_minc = False
            timeout_maxc = False
            beliefs_close_mult = overall_maxc < overall_minc * c.min_maxc_minc_gap_mult
            beliefs_far = overall_maxc > 2 * overall_minc  # False
            first = 0
            minc_changed = overall_minc > v.min_c[n][first]
            maxc_changed = overall_maxc < v.max_c[n][first]
            # TODO: replace other came close, with beliefs came close.
            maxc_came_close = z3.And(maxc_changed, beliefs_close_mult)
            minc_came_close = z3.And(minc_changed, beliefs_close_mult)
            minc_changed_significantly = overall_minc > c.maxc_minc_change_mult * \
                v.min_c[n][0]
            maxc_changed_significantly = overall_maxc * \
                c.maxc_minc_change_mult < v.max_c[n][0]
            timeout_allowed = et == c.T-1
            RESET_IMMEDIATELY = False
            if(RESET_IMMEDIATELY):
                timeout_allowed = et == c.T-2

            if(c.fix_stale__min_c):
                if(RESET_IMMEDIATELY):
                    # Reset immediately if invalid.
                    timeout_minc = z3.If(
                        minc_changed, False,
                        z3.If(maxc_came_close, True,
                              z3.And(timeout_allowed,
                                     z3.Not(maxc_changed_significantly))))
                else:
                    # All resets are infrequent
                    timeout_minc = z3.And(
                        timeout_allowed, z3.Not(minc_changed), z3.Not(beliefs_far),
                        z3.Or(maxc_came_close, z3.Not(maxc_changed_significantly)))

            if(c.fix_stale__max_c):
                if(RESET_IMMEDIATELY):
                    # Reset immediately if invalid.
                    timeout_maxc = z3.If(
                        maxc_changed, False,
                        z3.If(minc_came_close, True,
                              z3.And(timeout_allowed,
                                     z3.Not(minc_changed_significantly))))
                else:
                    # All resets are infrequent
                    timeout_maxc = z3.And(
                        timeout_allowed, z3.Not(maxc_changed), z3.Not(beliefs_far),
                        z3.Or(minc_came_close, z3.Not(minc_changed_significantly)))

            s.add(v.min_c[n][et] ==
                  z3.If(timeout_minc, recomputed_minc, overall_minc))
            reset_maxc = z3_max(
                v.min_c[n][et] * c.min_maxc_minc_gap_mult, 3/2 * v.max_c[n][et-1])
            s.add(v.max_c[n][et] ==
                  z3.If(timeout_maxc, reset_maxc, overall_maxc))


def update_bandwidth_beliefs_invalidation_and_timeout_use_more_vars(
        c: ModelConfig, s: MySolver, v: Variables):
    assert(c.beliefs)
    assert isinstance(v, BaseLink.LinkVariables)

    """
    utilized[st] means interval (st, st+1] is utilized
    utilized_cummulative[et][st] means interval (st, et] is utilized
    """

    # Compute utilized
    for n in range(c.N):
        for st in range(c.T-1):
            # Encodes that utilization must have been high if loss happened
            # or if queing delay was more than D

            # All of qdel might be False. So to check high delay we just
            # check Not of low delay.
            high_delay = z3.Not(z3.Or(*[v.qdel[st+1][dt]
                                        for dt in range(c.D+1)]))
            # Wrong:
            # high_delay = z3.Or(*[v.qdel[st+1][dt]
            #                      for dt in range(c.D+1, c.T)])
            loss = v.Ld_f[n][st+1] - v.Ld_f[n][st] > 0
            if(c.D == 0):
                assert c.loss_oracle
                loss = v.L_f[n][st+1] - v.L_f[n][st] > 0

            # We can only count qdelay when packets actually arrived.
            # Otherwise the qdelay is stale.
            # recvd_new_pkts = v.S_f[n][st+1] > v.S_f[n][st]
            recvd_new_pkts = True

            # CCAC can somehow give high utilization signals even when cwnd is low.
            # This basically happens when no new pkts are sent due to low cwnd.
            # To avoid underestimating max_c, we update it only when we sent new pkts.

            # Updated reason: A delay of high can happen when there are last
            # few pkts left in the queue. Thus the ack rate will be low. If
            # the CCA sent pkts R + D time ago, this qdelay can only be high
            # if this quue is big, if the queue is small, then the recently
            # sent pkts will show low qdelay.

            # Based on this, ideally we should be checking A_f[n][st+1-D]
            # > A_f[n][st-D]. But this works out. For ideal link D = 0, so
            # this is fine. For jittery link, since we always look at
            # windows >=D+1, the constraint for previous batch would ensure
            # constraint for this batch (we either sent new packets in
            # previous batch or caused loss, loss can only happen if we sent
            # new pkts.). We only really care about the constraint for the
            # last batch.
            sent_new_pkts = v.A_f[n][st+1] - v.A_f[n][st] > 0

            # We assume the link is utilized if we did not recv any packets.
            # If we don't do this, the network sends 0, then 2 * CD, then 0 pkts and so on.
            # This causes us to never have long enough utilized window (hence we can't improve max_c).
            # this_utilized = z3.And(v.S_f[n][st+1] > v.S_f[n][st],
            #                        z3.Or(loss, high_delay))
            this_utilized = z3.Or(
                z3.And(high_delay, sent_new_pkts, recvd_new_pkts),
                loss)

            s.add(v.utilized[n][st] == this_utilized)

    # Compute utilized_cummulative
    for n in range(c.N):
        for et in range(1, c.T):
            for st in range(et-1, -1, -1):
                if(st + 1 == et):
                    s.add(v.utilized_cummulative[n][et][st] == v.utilized[n][st])
                else:
                    s.add(v.utilized_cummulative[n][et][st] == z3.And(
                        v.utilized_cummulative[n][et][st+1], v.utilized[n][st]))

    # Bandwidth beliefs (C)
    """
    In summary C >= r * n/(n+1) always, if we additionally know that
    sending rate is higher than C then, C <= r * n/(n-1).
    """
    for n in range(c.N):
        # There is no measurement for min_measured_c when et=1, so we define it
        # to just be initial max_c.
        s.add(v.min_measured_c[n][1][0] == v.max_c[n][0])

        for et in range(1, c.T):
            base_maxc = v.max_c[n][et-1]
            for st in range(et-1, -1, -1):
                window = et - st
                measured_c = (v.S_f[n][et] - v.S_f[n][st]) / (window)

                # LOWER
                this_lower = measured_c * window / (window + c.D)
                if (st+1 == et):
                    s.add(v.max_measured_c[n][et][st] == this_lower)
                else:
                    s.add(v.max_measured_c[n][et][st] == z3_max(
                        v.max_measured_c[n][et][st+1], this_lower))

                # UPPER
                """
                For ideal link, D = 0, but still we want ot measure ack rate
                over at least windows of length 2. This is because initial
                conditions can provide utilized signals for 1 timeunit while
                still giving low ack rate.

                # if(window - max(1, c.D) > 0):

                Above issue if fixed by ensuring loss detected happens then
                buffer must be full and also using L_f instead of Ld_f, i.e., if
                loss happened then we must have sent pkts and service must have
                been high. This needs to change when we do loss detection.
                """
                if(window - c.D > 0):
                    this_upper = z3.If(
                        v.utilized_cummulative[n][et][st],
                        measured_c * window / (window - c.D),
                        base_maxc)
                    assert isinstance(this_upper, z3.ArithRef)
                    if(window - c.D == 1):
                        s.add(v.min_measured_c[n][et][st] == this_upper)
                    else:
                        s.add(v.min_measured_c[n][et][st] == z3_min(
                            v.min_measured_c[n][et][st+1], this_upper))

            recomputed_minc = v.max_measured_c[n][et][0]
            overall_minc = z3_max(v.min_c[n][et-1], v.max_measured_c[n][et][0])
            overall_maxc = z3_min(v.max_c[n][et-1], v.min_measured_c[n][et][0])

            """
            If self changed, don't recompute
            If other came close do recompute.

            On timeout, if other did not change significantly, then recompute.
            """
            timeout_minc = False
            timeout_maxc = False
            beliefs_close_mult = overall_maxc < overall_minc * c.min_maxc_minc_gap_mult
            beliefs_far = overall_maxc > 2 * overall_minc  # False
            first = 0
            minc_changed = overall_minc > v.min_c[n][first]
            maxc_changed = overall_maxc < v.max_c[n][first]
            # TODO: replace other came close, with beliefs came close.
            maxc_came_close = z3.And(maxc_changed, beliefs_close_mult)
            minc_came_close = z3.And(minc_changed, beliefs_close_mult)
            minc_changed_significantly = overall_minc > c.maxc_minc_change_mult * \
                v.min_c[n][0]
            maxc_changed_significantly = overall_maxc * \
                c.maxc_minc_change_mult < v.max_c[n][0]
            timeout_allowed = et == c.T-1
            RESET_IMMEDIATELY = False
            if(RESET_IMMEDIATELY):
                timeout_allowed = et == c.T-2

            if(c.fix_stale__min_c):
                if(RESET_IMMEDIATELY):
                    # Reset immediately if invalid.
                    timeout_minc = z3.If(
                        minc_changed, False,
                        z3.If(maxc_came_close, True,
                              z3.And(timeout_allowed,
                                     z3.Not(maxc_changed_significantly))))
                else:
                    # All resets are infrequent
                    timeout_minc = z3.And(
                        timeout_allowed, z3.Not(minc_changed), z3.Not(beliefs_far),
                        z3.Or(maxc_came_close, z3.Not(maxc_changed_significantly)))

            if(c.fix_stale__max_c):
                if(RESET_IMMEDIATELY):
                    # Reset immediately if invalid.
                    timeout_maxc = z3.If(
                        maxc_changed, False,
                        z3.If(minc_came_close, True,
                              z3.And(timeout_allowed,
                                     z3.Not(minc_changed_significantly))))
                else:
                    # All resets are infrequent
                    timeout_maxc = z3.And(
                        timeout_allowed, z3.Not(maxc_changed), z3.Not(beliefs_far),
                        z3.Or(minc_came_close, z3.Not(minc_changed_significantly)))

            s.add(v.min_c[n][et] ==
                  z3.If(timeout_minc, recomputed_minc, overall_minc))
            reset_maxc = z3_max(
                v.min_c[n][et] * c.min_maxc_minc_gap_mult, 3/2 * v.max_c[n][et-1])
            s.add(v.max_c[n][et] ==
                  z3.If(timeout_maxc, reset_maxc, overall_maxc))


def update_beliefs(c: ModelConfig, s: MySolver, v: Variables):
    assert(c.beliefs)
    # Note beleifs are updated after packets have been recvd at time t.
    # At time t, the CCA can only look at beliefs for time t-1.

    # Qdel
    for n in range(c.N):
        for et in range(c.T):
            for dt in range(c.T):
                s.add(z3.Implies(
                    v.qdel[et][dt], v.min_qdel[n][et] == max(0, dt-c.D)))
            s.add(z3.Implies(
                z3.And(*[z3.Not(v.qdel[et][dt]) for dt in range(c.T)]),
                v.min_qdel[n][et] == c.T-c.D))

            for dt in range(c.T):
                s.add(z3.Implies(
                    v.qdel[et][dt], v.max_qdel[n][et] == dt+1))
            s.add(z3.Implies(
                z3.And(*[z3.Not(v.qdel[et][dt]) for dt in range(c.T)]),
                v.max_qdel[n][et] == 1000 * c.T))

    # Buffer
    if (c.buf_min is not None and c.beliefs_use_buffer):
        for n in range(c.N):
            for et in range(1, c.T):
                for dt in range(c.T):
                    s.add(z3.Implies(
                        v.qdel[et][dt],
                        v.min_buffer[n][et] ==
                        z3_max(v.min_buffer[n][et-1], max(0, dt-c.D))))
                # TODO: is this the best way to handle the case when qdel > et?
                s.add(z3.Implies(
                    z3.And(*[z3.Not(v.qdel[et][dt]) for dt in range(c.T)]),
                    v.min_buffer[n][et] == z3_max(
                        c.T-c.D,
                        v.min_buffer[n][et-1])))

                if (c.beliefs_use_max_buffer):
                    for dt in range(et+1):
                        s.add(z3.Implies(
                            z3.And(v.Ld_f[n][et] > v.Ld_f[n]
                                   [et-1], v.qdel[et][dt]),
                            v.max_buffer[n][et] ==
                            z3_min(v.max_buffer[n][et-1], dt+1)))
                    # TODO: is this the best way to handle the case when qdel > et?
                    s.add(z3.Implies(
                        z3.Or(z3.Not(v.Ld_f[n][et] > v.Ld_f[n][et-1]),
                              z3.And(*[z3.Not(v.qdel[et][dt]) for dt in range(et+1)])),
                        v.max_buffer[n][et] ==
                        v.max_buffer[n][et-1]))

    # update_bandwidth_beliefs_invalidation(c, s, v)
    # update_bandwidth_beliefs_with_timeout(c, s, v)
    # update_bandwidth_beliefs_invalidation_and_timeout(c, s, v)
    update_bandwidth_beliefs_invalidation_and_timeout_use_more_vars(c, s, v)

    if (c.app_limited and c.app_fixed_avg_rate):
        for n in range(c.N):
            for et in range(1, c.T):
                lb_list = [v.min_app_rate[n][et-1]]
                ub_list = [v.max_app_rate[n][et-1]]
                for st in range(et):
                    app_pkts = v.app_limits[n][et] - v.app_limits[n][st]
                    window = et - st
                    measured_app_rate = app_pkts / window

                    lb_list.append(measured_app_rate * window /
                                   (window + c.app_burst_factor * 1))
                    ub_list.append(measured_app_rate)

                s.add(v.min_app_rate[n][et] == z3_max_list(lb_list))
                s.add(v.max_app_rate[n][et] == z3_min_list(ub_list))


def initial_beliefs(c: ModelConfig, s: MySolver, v: Variables):
    assert(c.beliefs)

    # TODO: Use this only if we are requiring CCA to not reset beliefs when it thinks
    #  network changed.
    for n in range(c.N):
        s.add(v.min_c[n][0] >= v.alpha)
        s.add(v.min_c[n][0] * c.min_maxc_minc_gap_mult <= v.max_c[n][0])
        # s.add(v.min_c[n][0] >= 0)
        # s.add(v.max_c[n][0] > 0)
        # s.add(v.min_c[n][0] <= v.max_c[n][0])
        """
        Anytime that min_c > max_c happens, we want to see it happen within the
        trace, not b4.
        """
        if(not c.fix_stale__min_c):
            s.add(v.min_c[n][0] <= c.C)
        if(not c.fix_stale__max_c):
            s.add(v.max_c[n][0] >= c.C)

    if(c.buf_min is not None and c.beliefs_use_buffer):
        for n in range(c.N):
            s.add(v.min_buffer[n][0] >= 0)
            s.add(v.min_buffer[n][0] <= c.buf_min / c.C)
            if(c.beliefs_use_max_buffer):
                s.add(v.min_buffer[n][0] <= v.max_buffer[n][0])
                s.add(v.max_buffer[n][0] >= c.buf_min / c.C)

            for dt in range(c.T):
                s.add(z3.Implies(
                    v.qdel[0][dt],
                    v.min_buffer[n][0] >= max(0, dt-c.D)))

    if(c.app_limited and c.app_fixed_avg_rate):
        for n in range(c.N):
            s.add(v.min_app_rate[n][0] >= 0)
            s.add(v.min_app_rate[n][0] <= v.max_app_rate[n][0])
            s.add(v.min_app_rate[n][0] <= v.app_rate)
            s.add(v.max_app_rate[n][0] >= v.app_rate)

    # Part of def vars now.
    # for n in range(c.N):
    #     s.add(v.min_qdel[n][0] >= 0)
    #     s.add(v.min_qdel[n][0] <= v.max_qdel[n][0])
    #     if(c.buf_min is not None):
    #         assert c.buf_min == c.buf_max
    #         s.add(v.min_qdel[n][0] <= c.buf_min / c.C)
    #         s.add(v.max_qdel[n][0] >= c.buf_min / c.C)


# Deprecated
# def get_beliefs_reset(cc: CegisConfig, c: ModelConfig, v: Variables):
#     assert(c.beliefs)

#     # TODO: currently only doing this for link rate. Synthesized CCAs don't seem
#     #  to be using buffer anyway.

#     first = cc.history
#     first = 0

#     WRONGNESS_FACTOR = 10
#     beliefs_bad_list: List[z3.BoolRef] = []
#     for n in range(c.N):
#         beliefs_bad_list.append(
#             v.min_c[n][0] > WRONGNESS_FACTOR * c.C)
#         beliefs_bad_list.append(
#             v.max_c[n][0] < c.C / WRONGNESS_FACTOR)
#     beliefs_bad = z3.Or(*beliefs_bad_list)

#     # TODO: Will eventually reset actually ever cause reset?
#     beliefs_eventually_reset_list: List[z3.BoolRef] = []
#     for n in range(c.N):
#         # beliefs_eventually_reset_list.append(
#         #     v.max_c[n][-1] < v.min_c[n][-1])
#         beliefs_eventually_reset_list.append(
#             z3.And(
#                 z3.Implies(v.min_c[n][0] > WRONGNESS_FACTOR * c.C,
#                            v.max_c[n][-1] < v.max_c[n][0]),
#                 z3.Implies(v.max_c[n][0] < c.C / WRONGNESS_FACTOR,
#                            z3.Or(v.min_c[n][-1] > v.min_c[n][0],
#                                  v.max_c[n][-1] > v.max_c[n][0])
#                            )
#             ))
#         # beliefs_eventually_reset_list.append(
#         #     v.max_c[n][-1] - v.min_c[n][-1] < v.max_c[n][0] - v.min_c[n][0]
#         # )
#     # TODO: should this be AND or OR? Should all flows reset or any one?
#     beliefs_reset = z3.And(*beliefs_eventually_reset_list)

#     return beliefs_bad, beliefs_reset


def loss_deterministic(c: ModelConfig, s: MySolver, v: Variables):
    """
    L[t] - L[t-1] <= Max(0, r - c)
    """
    assert c.deterministic_loss
    assert c.buf_max == c.buf_min

    if c.buf_min is not None:
        s.add(v.A[0] - v.L[0] <= v.C0 + c.C * 0 - v.W[0] + c.buf_min)
        s.add(z3.Implies(z3.Or([v.L_f[n][0] > v.Ld_f[n][0] for n in range(c.N)]),
                         v.A[0] - v.L[0] == v.C0 + c.C * 0 - v.W[0] + c.buf_min))
    for t in range(1, c.T):
        if c.buf_min is None:  # no loss case
            s.add(v.L[t] == v.L[0])
        else:
            s.add(z3.Implies(
                v.A[t] - v.L[t-1] > v.C0 + c.C * t - v.W[t] + c.buf_min,
                v.A[t] - v.L[t] == v.C0 + c.C * t - v.W[t] + c.buf_min))
            s.add(z3.Implies(
                v.A[t] - v.L[t-1] <= v.C0 + c.C * t - v.W[t] + c.buf_min,
                v.L[t] == v.L[t-1]))
    # L[0] is still chosen non-deterministically in an unconstrained fashion.

    if(c.buf_min is None):
        # For multiflow, L_f is ideally non-deterministic
        # When there is inf buffer however, L_f should be deterministic
        for n in range(c.N):
            for t in range(1, c.T):
                s.add(v.L_f[n][t] == v.L_f[n][0])
            # There shouldn't be any detected losses as well.
            s.add(v.Ld_f[n][0] == v.L_f[n][0])


def loss_non_deterministic(c: ModelConfig, s: MySolver, v: Variables):
    assert not c.deterministic_loss
    for t in range(c.T):
        if c.buf_min is not None:
            if t > 0:
                s.add(
                    z3.Implies(
                        v.L[t] > v.L[t - 1], v.A[t] - v.L[t] >= v.C0 + c.C *
                        (t - 1) - v.W[t - 1] + c.buf_min
                    ))
            # L[0] is chosen non-deterministically without any constraints.
        else:
            s.add(v.L[t] == v.L[0])

        # Enforce buf_max if given
        if c.buf_max is not None:
            s.add(v.A[t] - v.L[t] <= v.C0 + c.C * t - v.W[t] + c.buf_max)


def calculate_qbound_defs(c: ModelConfig, s: MySolver, v: Variables):
    # qbound[t][dt<=t] is deterministic
    # qbound[t][dt>t] is non deterministic (including qbound[0][dt>0])
    """
           dt
         0 1 2 3
       ----------
      0| d n n n
    t 1| d d n n
      2| d d d n
      3| d d d d
    """

    # Let solver choose non-deterministically what happens for
    # t = 0, dt > 0, i.e., no constraint on qdel[0][dt>0].

    # By definition queuing delay >= 0
    for t in range(c.T):
        s.add(v.qbound[t][0])

    for t in range(1, c.T):
        for dt in range(1, t+1):
            s.add(
                z3.Implies(v.S[t] == v.S[t-1],
                           v.qbound[t][dt] == v.qbound[t-1][dt]))
            s.add(
                z3.Implies(v.S[t] != v.S[t-1],
                           v.qbound[t][dt] ==
                           (v.S[t] <= v.A[t-dt] - v.L[t-dt])))


def calculate_qbound_env(c: ModelConfig, s: MySolver, v: Variables):
    # These have to be part of environment as vars at t+1 and dt+1 might be
    # deterministic (def var), and those at t, dt might be non determinisitic
    # (verifier var), a new value to def var may make the non-deterministic
    # choice infeasible. This could violate below constraint. Such violation
    # needs to be allowed.

    # Needed only for non-deterministic choices, mostly a sanity constraint for
    # deterministic variables.
    for t in range(c.T):
        for dt in range(c.T-1):
            # If queuing delay at t is greater than dt+1 then
            # it is also greater than dt.
            s.add(z3.Implies(v.qbound[t][dt+1], v.qbound[t][dt]))

    for t in range(1, c.T):
        for dt in range(t+1, c.T):
            s.add(z3.Implies(v.S[t] == v.S[t-1],
                             v.qbound[t][dt] == v.qbound[t-1][dt]))
            # Let solver choose non-deterministically what happens when
            # S[t] != S[t-1] for t-dt < 0, i.e.,
            # no constraint on qbound[t][dt>t]

    for t in range(c.T-1):
        for dt in range(c.T-1):
            # Queuing delay at time (t+1) cannot be
            # greater than (queuing delay at time t) + 1.

            # Why? -> Say queuing delay at time t1 is qd1 and at time t2>t1 is
            # qd2, so the packet recvd at time t1 was sent at t1-qd1 and at that
            # recvd t2 was sent at t2-qd2. Since packets are sent in FIFO order
            # t2-qd2 >= t1-qd1. Put t2=t1+1, we get qd2 <= qd1+1, i.e., (qd at
            # time t1+1) is less than or equal to qd1+1.

            # We encode if queueing delay at time t is less than dt, then
            # queuing delay at time t+1 is less than dt+1. i.e., qd(t+1) <=
            # qd(t) + 1 < dt + 1. Recall qbound[t][dt] encodes qdelay at time t
            # is greater than or equal to dt. Hence, not(qbound[t][dt]) encodes
            # qdelay is less than dt.
            s.add(z3.Implies(
                z3.Not(v.qbound[t][dt]),
                z3.Not(v.qbound[t+1][dt+1])))

    # # Queueing delay cannot be more than buffer/C + D.
    # if(c.buf_min is not None):
    #     max_delay_float = c.buf_min/c.C + c.D
    #     max_delay_ceil = math.ceil(max_delay_float)
    #     lowest_unallowed_delay = max_delay_ceil
    #     if(max_delay_float == max_delay_ceil):
    #         lowest_unallowed_delay = max_delay_ceil + 1
    #     if(lowest_unallowed_delay <= c.T - 1):
    #         for t in range(c.T):
    #             s.add(z3.Not(v.qbound[t][lowest_unallowed_delay]))

    # Delay cannot be more than max_delay. This means that one of
    # qbound[t][dt>max_delay] are False
    if(c.buf_min is not None):
        max_delay = c.buf_min/c.C + c.D
        for t in range(c.T):
            for dt in range(c.T):
                s.add(z3.Implies(dt > max_delay, z3.Not(v.qbound[t][dt])))


def last_decrease_defs(c: ModelConfig, s: MySolver, v: Variables):

    for n in range(c.N):
        # s.add(v.last_decrease_f[0][0] == v.A_f[0][0] - v.L_f[0][0])
        s.add(v.last_decrease_f[n][0] == v.S_f[n][0])

        for t in range(1, c.T):
            # Const last_decrease in history
            # definition_constrs.append(
            #     last_decrease_f[0][t] == v.S_f[0][t])

            s.add(
                z3.Implies(v.c_f[n][t] < v.c_f[n][t-1],
                           v.last_decrease_f[n][t] == v.A_f[n][t] - v.L_f[n][t]))
            s.add(
                z3.Implies(v.c_f[n][t] >= v.c_f[n][t-1],
                           v.last_decrease_f[n][t] == v.last_decrease_f[n][t-1]))


def exceed_queue_defs(c: ModelConfig, s: MySolver, v: Variables):
    for n in range(c.N):
        for t in range(c.R, c.T):
            for dt in range(c.T):
                s.add(z3.Implies(
                    z3.And(dt == v.qsize_thresh, v.qbound[t-c.R][dt]),
                    v.exceed_queue_f[n][t]))
                s.add(z3.Implies(
                    z3.And(dt == v.qsize_thresh, z3.Not(v.qbound[t-c.R][dt])),
                    z3.Not(v.exceed_queue_f[n][t])))


def calculate_qdel_defs(c: ModelConfig, s: MySolver, v: Variables):
    # qdel[t][dt>=t] is non-deterministic (including,
    #                qdel[0][dt], qdel[t][dt>t-1])
    # qdel[t][dt<t] is deterministic
    """
           dt
         0 1 2 3
       ----------
      0| n n n n
    t 1| d n n n
      2| d d n n
      3| d d d n
    """

    for t in range(1, c.T):
        for dt in range(t):
            s.add(z3.Implies(v.S[t] != v.S[t - 1],
                             v.qdel[t][dt] == z3.And(
                v.A[t - dt - 1] - v.L[t - dt - 1] < v.S[t],
                v.A[t - dt] - v.L[t - dt] >= v.S[t])))
            s.add(z3.Implies(v.S[t] == v.S[t - 1],
                             v.qdel[t][dt] == v.qdel[t-1][dt]))


def calculate_qdel_env(c: ModelConfig, s: MySolver, v: Variables):
    # There can be only one value for queuing delay at a given time.
    # Needed only for non-deterministic choices, mostly a sanity constraint for
    # deterministic variables.
    for t in range(c.T):
        s.add(z3.Sum(*v.qdel[t]) <= 1)

    # Let solver choose non-deterministically what happens for
    # t = 0, i.e., no constraint on qdel[0][dt].
    for t in range(1, c.T):
        for dt in range(t, c.T):
            s.add(z3.Implies(v.S[t] == v.S[t - 1],
                             v.qdel[t][dt] == v.qdel[t-1][dt]))
            # We let solver choose non-deterministically what happens when
            # S[t] != S[t-1] for dt > t-1, i.e.,
            # no constraint on qdel[t][dt>t-1]

    # qdel[t][dt] is True iff queueing delay is >=dt but <dt+1
    # If queuing delay at time t1 is dt1, then queuing delay at time t2=t1+1,
    # cannot be more than dt1+1. I.e., qdel[t2][dt1+1+1] has to be false.
    # Note qdel[t2][dt1+1] can be true as queueing delay at t1+1 can be dt1+1.
    for t1 in range(c.T-1):
        for dt1 in range(c.T):
            t2 = t1+1
            # dt2 starts from dt1+1+1
            s.add(z3.Implies(
                v.qdel[t1][dt1],
                z3.And(*[z3.Not(v.qdel[t2][dt2])
                         for dt2 in range(dt1+1+1, c.T)])
            ))
            s.add(z3.Implies(
                v.qdel[t1][dt1],
                z3.Or(*[v.qdel[t2][dt2] for dt2 in range(min(c.T, dt1+1+1))])
            ))

    # Queueing delay cannot be more than buffer/C + D.
    # # TODO: Can just put dt > max_delay implies not qdel t dt
    # if(c.buf_min is not None and not isinstance(c.buf_min, z3.ExprRef)):
    #     max_delay_float = c.buf_min/c.C + c.D
    #     max_delay_ceil = math.ceil(max_delay_float)
    #     lowest_unallowed_delay = max_delay_ceil
    #     if(max_delay_float == max_delay_ceil):
    #         lowest_unallowed_delay = max_delay_ceil + 1
    #     for t in range(c.T):
    #         for dt in range(lowest_unallowed_delay, c.T):
    #             s.add(z3.Not(v.qdel[t][dt]))
    #         s.add(1 == z3.Sum(
    #             *[v.qdel[t][dt]
    #               for dt in range(min(c.T, lowest_unallowed_delay))]))

    # Delay cannot be more than max_delay. This means that one of
    # qdel[t][dt<=max_delay] must be true. This also means that all of
    # qdel[t][dt>max_delay] are false.
    if(c.buf_min is not None):
        max_delay = c.buf_min/c.C + c.D
        for t in range(c.T):
            some_qdel_is_true = (1 == z3.Sum(
                *[z3.If(dt <= max_delay, v.qdel[t][dt], False)
                  for dt in range(c.T)]))
            # if max_delay is very high, then all qdel can be false.
            s.add(z3.Implies(max_delay < c.T, some_qdel_is_true))


def fifo_service(c: ModelConfig, s: MySolver, v: Variables):
    # If a flow sees service of more bytes than in queue at a particular time,
    # then each flow should have seen service of all their bytes in the queue
    # at that particular time.
    for n1 in range(c.N):
        for t1 in range(c.T):
            for t2 in range(t1+1, c.T):
                s.add(
                    z3.Implies(
                        v.S_f[n1][t2] - v.S_f[n1][t1] >
                        v.A_f[n1][t1] - v.L_f[n1][t1] - v.S_f[n1][t1],
                        z3.And(
                            *[v.S_f[n2][t2] - v.S_f[n2][t1] >=
                              v.A_f[n2][t1] - v.L_f[n2][t1] - v.S_f[n2][t1]
                              for n2 in range(c.N)])))


def setup_ccac():
    c = ModelConfig.default()
    c.compose = True
    c.cca = "paced"
    c.simplify = False
    c.calculate_qdel = False
    c.C = 100
    c.T = 9

    s = MySolver()
    s.warn_undeclared = False
    v = Variables(c, s)

    # s.add(z3.And(v.S[0] <= 1000, v.S[0] >= -1000))
    return c, s, v


def not_too_adversarial_init_cwnd(
        cc: CegisConfig, c: ModelConfig, s: MySolver, v: Variables):
    for n in range(c.N):
        for t in range(1, cc.history):
            s.add(v.c_f[n][t] <= v.c_f[n][t-1] * 3/2)


def setup_ccac_full(cca="copa"):
    c = ModelConfig.default()
    c.compose = True
    c.cca = cca
    c.simplify = False
    c.calculate_qdel = True
    # c.S0 = 0
    # c.L0 = 0
    # c.W0 = 1000
    c.C = 100
    c.T = 7
    s, v = make_solver(c)
    # Consider the no loss case for simplicity
    s.add(v.L[0] == v.L[-1])
    # make_periodic(c, s, v, c.R + c.D)
    # Avoid weird cases where single packet is larger than BDP.
    s.add(v.alpha < 1/5)
    return c, s, v


def get_periodic_constraints(cc: CegisConfig, c: ModelConfig, v: Variables):
    periodic = []

    # # In beginning and in end,
    # # For each flow, packet queue and cwnd should be same
    # # Token queue should be same
    # last = cc.T-1 - (cc.history-1)
    # for h in range(cc.history):
    #     for n in range(cc.N):
    #         periodic.append(v.c_f[n][h] == v.c_f[n][last+h])
    #         periodic.append(
    #             v.A_f[n][h] - v.L_f[n][h] - v.S_f[n][h] ==
    #             v.A_f[n][last+h] - v.L_f[n][last+h] - v.S_f[n][last+h])
    #     periodic.append(v.C0 + c.C * h - v.W[h] - v.S[h] ==
    #                     v.C0 + c.C * (last+h) - v.W[last+h] - v.S[last+h])

    # First decided by CCA is equal to last decided by CCA
    for n in range(cc.N):
        periodic.append(v.c_f[n][cc.history] == v.c_f[n][c.T-1])
        periodic.append(
            v.A_f[n][cc.history] - v.L_f[n][cc.history] - v.S_f[n][cc.history] ==
            v.A_f[n][c.T-1] - v.L_f[n][c.T-1] - v.S_f[n][c.T-1])
    periodic.append(v.C0 + c.C * cc.history - v.W[cc.history] - v.S[cc.history] ==
                    v.C0 + c.C * (c.T-1) - v.W[c.T-1] - v.S[c.T-1])

    return z3.And(*periodic)


def maximize_gap(
    c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    """
    Adds constraints as a side effect to the verifier formulation
    that try to maximize the gap between
    service curve and the upper bound of service curve (based on waste choices).
    """
    s = verifier

    orig_sat = s.check()
    if str(orig_sat) != "sat":
        return orig_sat, None

    orig_model = s.model()
    cur_min = np.inf
    for t in range(1, c.T):
        if(orig_model.eval(v.W[t] > v.W[t-1])):
            this_gap = get_raw_value(orig_model.eval(
                v.C0 + c.C * t - v.W[t] - v.S[t]))
            cur_min = min(cur_min, this_gap)

    if(cur_min == np.inf):
        return orig_sat, orig_model

    if(cur_min < 0):
        # This is basically in case of multi-link. Where this link may not be
        # satisfying the environment constraints.
        return orig_sat, orig_model

    eps = max(1, c.C * c.D)/64
    binary_search = BinarySearch(0, c.C * c.D, eps)
    min_gap = z3.Real('min_gap', ctx=ctx)

    for t in range(1, c.T):
        s.add(z3.Implies(
            v.W[t] > v.W[t-1],
            min_gap <= v.C0 + c.C * t - v.W[t] - v.S[t]))

    while True:
        pt = binary_search.next_pt()
        if pt is None:
            break

        s.push()
        s.add(min_gap >= pt)
        try:
            sat = s.check()
        except z3.z3types.Z3Exception as e:
            logger.error(f"During binary search for max gap,"
                         f" verifier threw error: {e}")
            logger.info("Defaulting to using unoptimized model")
            s.pop()
            return orig_sat, orig_model
        s.pop()

        if(str(sat) == 'sat'):
            binary_search.register_pt(pt, 1)
        elif str(sat) == "unknown":
            binary_search.register_pt(pt, 2)
        else:
            assert str(sat) == "unsat", f"Unknown value: {str(sat)}"
            binary_search.register_pt(pt, 3)

    best_gap, _, _ = binary_search.get_bounds()
    s.add(min_gap >= best_gap)

    sat = s.check()
    assert sat == orig_sat
    model = s.model()

    logger.info("Improved gap from {} to {}".format(float(cur_min), best_gap))
    return sat, model


def worst_counter_example(
    first: int, c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    """
    Adds constraints as a side effect to the verifier formulation
    to get the counter example that causes generator to move the most.
    """
    s = verifier

    orig_sat = s.check()
    if str(orig_sat) != "sat":
        return orig_sat, None

    objective_rhs = 0
    for t in range(c.T):
        objective_rhs += v.A[t] - v.L[t] - v.S[t]
        objective_rhs += v.C0 + c.C * t - v.W[t] - v.S[t]
        if(t >= c.D):
            objective_rhs += v.S[t] - (v.C0 + c.C * (t-c.D) - v.W[t-c.D])
    objective_rhs += v.A[-1] - v.L[-1] - v.S[-1] - \
        (v.A[first] - v.L[first] - v.S[first])
    objective_rhs += -v.S[-1] + v.S[first]

    orig_model = s.model()
    cur_obj = get_raw_value(orig_model.eval(objective_rhs))

    binary_search = BinarySearch(0, 1e5, c.C * c.D/64)
    objective = z3.Real('objective', ctx=ctx)

    s.add(objective <= objective_rhs)

    while True:
        pt = binary_search.next_pt()
        if pt is None:
            break

        s.push()
        s.add(objective >= pt)
        sat = s.check()
        s.pop()

        if(str(sat) == 'sat'):
            binary_search.register_pt(pt, 1)
        elif str(sat) == "unknown":
            binary_search.register_pt(pt, 2)
        else:
            assert str(sat) == "unsat", f"Unknown value: {str(sat)}"
            binary_search.register_pt(pt, 3)

    best_obj, _, _ = binary_search.get_bounds()
    s.add(objective >= best_obj)

    sat = s.check()
    assert sat == orig_sat
    model = s.model()

    logger.info("Improved obj from {} to {}".format(float(cur_obj), best_obj))
    return sat, model


def run_verifier_incomplete(
    c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    # This is meant to create a partial function by
    # getting c, v, ctx from closure.
    if(hasattr(v, 'W') and hasattr(v, 'C0')):
        maximize_gap(c, v, ctx, verifier)
    sat, _, model = find_small_denom_soln(verifier, 4096)
    return sat, model


def run_verifier_incomplete_wce(
    first: int, c: ModelConfig, v: Variables, ctx: z3.Context, verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    # This is meant to create a partial function by
    # getting c, v, ctx from closure.
    _, _ = maximize_gap(c, v, ctx, verifier)
    _, _ = worst_counter_example(first, c, v, ctx, verifier)
    sat, _, model = find_small_denom_soln(verifier, 4096)
    return sat, model


def get_cex_df(
    counter_example: z3.ModelRef, v: Variables,
    vn: VariableNames, c: ModelConfig
) -> pd.DataFrame:
    def _get_model_value(l):
        return get_model_value_list(counter_example, l)

    cex_dict = {
        # get_name_for_list(vn.A): _get_model_value(v.A),
        # get_name_for_list(vn.S): _get_model_value(v.S),
        # get_name_for_list(vn.L): _get_model_value(v.L),
    }
    for n in range(c.N):
        cex_dict.update({
            get_name_for_list(vn.A_f[n]): _get_model_value(v.A_f[n]),
            get_name_for_list(vn.c_f[n]): _get_model_value(v.c_f[n]),
            get_name_for_list(vn.r_f[n]): _get_model_value(v.r_f[n]),
            get_name_for_list(vn.S_f[n]): _get_model_value(v.S_f[n]),
            get_name_for_list(vn.L_f[n]): _get_model_value(v.L_f[n]),
            get_name_for_list(vn.Ld_f[n]): _get_model_value(v.Ld_f[n]),
            # get_name_for_list(vn.timeout_f[n]): _get_model_value(v.timeout_f[n]),
        })
    if(c.feasible_response):
        cex_dict.update({
            get_name_for_list(vn.S_choice): _get_model_value(v.S_choice)})
    if(c.beliefs):
        for n in range(c.N):
            cex_dict.update({
                get_name_for_list(vn.min_c[n]): _get_model_value(v.min_c[n]),
                get_name_for_list(vn.max_c[n]): _get_model_value(v.max_c[n]),
                get_name_for_list(vn.min_qdel[n]): _get_model_value(v.min_qdel[n]),
                # get_name_for_list(vn.max_qdel[n]): _get_model_value(v.max_qdel[n])
                })
        if(c.buf_min is not None and c.beliefs_use_buffer):
            for n in range(c.N):
                cex_dict.update({
                    get_name_for_list(vn.min_buffer[n]): _get_model_value(v.min_buffer[n])})
            if(c.beliefs_use_max_buffer):
                for n in range(c.N):
                    cex_dict.update({
                        get_name_for_list(vn.max_buffer[n]): _get_model_value(v.max_buffer[n])})
    # if(hasattr(v, 'W')):
    #     cex_dict.update({
    #         get_name_for_list(vn.W): _get_model_value(v.W)})
    #     # cex_dict.update({
    #     #     'non-wasted-tokens': _get_model_value(
    #     #         [v.C0 + c.C * t - v.W[t] for t in range(c.T)])})
    if(c.app_limited):
        for n in range(c.N):
            cex_dict.update({
                get_name_for_list(vn.app_limits[n]): _get_model_value(v.app_limits[n])
            })

    # ----------------------------------------------------------------
    # BUILD DF
    # ----------------------------------------------------------------
    df = pd.DataFrame(cex_dict).astype(float)

    # if(c.beliefs and c.buf_min is not None):
    #     for n in range(c.N):
    #         df[f'min_buffer_bytes_{n},t'] = df[get_name_for_list(vn.min_buffer[n])] * df[get_name_for_list(vn.min_c[n])]
    #         df[f'max_buffer_bytes_{n},t'] = df[get_name_for_list(vn.max_buffer[n])] * df[get_name_for_list(vn.max_c[n])]

    # Can remove this by adding queue_t as a definition variable...
    # This would also allow easily quiering this from generator
    queue_t = []
    for t in range(c.T):
        queue_t.append(
            get_raw_value(counter_example.eval(
                v.A[t] - v.L[t] - v.S[t])))
    df["queue_t"] = queue_t

    if (c.calculate_qdel and c.beliefs):
        for n in range(c.N):
            utilized_t = [-1]
            for t in range(1, c.T):
                high_delay = z3.Not(z3.Or(*[v.qdel[t][dt]
                                            for dt in range(c.D+1)]))
                loss = v.Ld_f[n][t] - v.Ld_f[n][t-1] > 0
                if(c.D == 0):
                    assert c.loss_oracle
                    loss = v.L_f[n][t] - v.L_f[n][t-1] > 0
                sent_new_pkts = v.A_f[n][t] - v.A_f[n][t-1] > 0
                # recvd_new_pkts = v.S_f[n][t] - v.S_f[n][t-1] > 0
                recvd_new_pkts = True
                this_utilized = z3.Or(
                    z3.And(high_delay, sent_new_pkts, recvd_new_pkts),
                    loss)
                utilized_t.append(get_raw_value(
                    counter_example.eval(this_utilized)))
            df[f"utilized_{n},t"] = utilized_t

    if(hasattr(v, 'C0') and hasattr(v, 'W')):
        bottle_queue_t = []
        token_queue_t = []
        for t in range(c.T):
            bottle_queue_t.append(
                get_raw_value(counter_example.eval(
                    v.A[t] - v.L[t] - (v.C0 + c.C * t - v.W[t])
                )))
            token_queue_t.append(
                get_raw_value(counter_example.eval(
                    v.C0 + c.C * t - v.W[t] - v.S[t]
                )))
        df["bqueue_t"] = bottle_queue_t
        df["tqueue_t"] = token_queue_t

        # upper_S = []
        # for t in range(c.T):
        #     upper_S.append(
        #         get_raw_value(counter_example.eval(
        #             (v.C0 + c.C * t - v.W[t])
        #         )))
        #     # if(t > c.D):
        #     #     lower_S.append(
        #     #         get_raw_value(counter_example.eval(
        #     #             v.C0 + c.C * (t - c.D) - v.W[t - c.D]
        #     #         ))
        #     #     )
        # df["upper_S_t"] = upper_S

    if(c.N == 1):
        df["del_A_f"] = df[get_name_for_list(vn.A_f[0])] - df[get_name_for_list(vn.A_f[0])].shift(1)
        df["del_S_f"] = df[get_name_for_list(vn.S_f[0])] - df[get_name_for_list(vn.S_f[0])].shift(1)
        df["del_L_f"] = df[get_name_for_list(vn.L_f[0])] - df[get_name_for_list(vn.L_f[0])].shift(1)

        # max_minc = [0]
        # for t in range(1, c.T):
        #     this_minc = 0
        #     for st in range(0, t):
        #         this_minc = max(this_minc, np.sum(df[f"del_A_f"][st+1:t+1])/(t-st + c.D + 1))
        #     max_minc.append(this_minc)
        # df['max_minc'] = max_minc

    if(c.calculate_qbound):
        qdelay = []
        for t in range(c.T):
            assert bool(counter_example.eval(v.qbound[t][0]))
            for dt in range(c.T-1, -1, -1):
                value = counter_example.eval(v.qbound[t][dt])
                try:
                    bool_value = bool(value)
                except z3.z3types.Z3Exception:
                    bool_value = True
                if(bool_value):
                    qdelay.append(dt+1)
                    break
        assert len(qdelay) == c.T
        df["qbound_t"] = np.array(qdelay).astype(float)
        for n in range(c.N):
            df[f"last_decrease_f_{n},t"] = np.array(
                [counter_example.eval(x).as_fraction()
                 for x in v.last_decrease_f[n]]).astype(float)
            df[f"exceed_queue_f_{n},t"] = [-1] + \
                get_val_list(counter_example, v.exceed_queue_f[n][1:])

        # qbound_vals = []
        # for qbound_list in v.qbound:
        #     qbound_val_list = get_val_list(counter_example, qbound_list)
        #     qbound_vals.append(qbound_val_list)
        # ret += "\n{}".format(np.array(qbound_vals))
        # ret += "\n{}".format(counter_example.eval(qsize_thresh))

    # if(c.calculate_qdel):
    #     qdelay = []
    #     for t in range(c.T):
    #         this_value = c.T
    #         for dt in range(c.T):
    #             value = counter_example.eval(v.qdel[t][dt])
    #             bool_value = bool(value)
    #             if(bool_value):
    #                 this_value = min(this_value, dt)
    #                 break
    #         qdelay.append(this_value)
    #     assert len(qdelay) == c.T
    #     df["qdelay_t"] = np.array(qdelay).astype(float)

    return df.astype(float)


def get_gen_cex_df(
    solution: z3.ModelRef, v: Variables, vn: VariableNames,
    n_cex: int, c: ModelConfig
) -> pd.DataFrame:
    if(n_cex == 0):
        return pd.DataFrame()
    name_template = NAME_TEMPLATE + str(n_cex)
    ctx = solution.ctx

    def _get_model_value(l):
        ret = []
        for vvar in l:
            cex_vvar_name = name_template.format(vvar)
            cex_var = z3.Const(cex_vvar_name, vvar.sort())
            ret.append(get_raw_value(solution.eval(cex_var)))
        return ret
    cex_dict = {
        get_name_for_list(vn.A): _get_model_value(v.A),
        get_name_for_list(vn.S): _get_model_value(v.S),
        # get_name_for_list(vn.L): _get_model_value(v.L),
    }
    for n in range(c.N):
        cex_dict.update({
            get_name_for_list(vn.A_f[n]): _get_model_value(v.A_f[n]),
            get_name_for_list(vn.c_f[n]): _get_model_value(v.c_f[n]),
            get_name_for_list(vn.r_f[n]): _get_model_value(v.r_f[n]),
            get_name_for_list(vn.S_f[n]): _get_model_value(v.S_f[n]),
            get_name_for_list(vn.L_f[n]): _get_model_value(v.L_f[n]),
            # get_name_for_list(vn.Ld_f[n]): _get_model_value(v.Ld_f[n]),
            # get_name_for_list(vn.timeout_f[n]): _get_model_value(v.timeout_f[n]),
        })
    if(hasattr(v, 'W')):
        cex_dict.update({
            get_name_for_list(vn.W): _get_model_value(v.W)})

    df = pd.DataFrame(cex_dict).astype(float)

    if(c.calculate_qbound):
        qdelay = []
        for t in range(c.T):
            g_qbound_t = get_renamed_vars(v.qbound[t], n_cex)
            assert bool(solution.eval(g_qbound_t[0]))
            for dt in range(c.T-1, -1, -1):
                value = solution.eval(g_qbound_t[dt])
                try:
                    bool_value = bool(value)
                except z3.z3types.Z3Exception:
                    bool_value = True
                if(bool_value):
                    qdelay.append(dt+1)
                    break
        assert len(qdelay) == c.T
        df["qdelay"] = np.array(qdelay).astype(float)

        g_last_decrease_f = get_renamed_vars(v.last_decrease_f[0], n_cex)
        df["last_decrease_f"] = np.array(
            [solution.eval(x).as_fraction()
             for x in g_last_decrease_f]).astype(float)

        g_exceed_queue_f = get_renamed_vars(v.exceed_queue_f[0], n_cex)
        df["exceed_queue_f"] = [-1] + \
            get_val_list(solution, g_exceed_queue_f[1:])

        # qbound_vals = []
        # for qbound_list in v.qbound:
        #     qbound_val_list = get_val_list(solution, _get_renamed(qbound_list))
        #     qbound_vals.append(qbound_val_list)
        # ret += "\n{}".format(np.array(qbound_vals))

    # df['service_delta'] = df[get_name_for_list(vn.S_f[0])] - df[get_name_for_list(vn.S_f[0])].shift()
    return df


def plot_cex(m: z3.ModelRef, df: pd.DataFrame, c: ModelConfig, v: Variables, fpath: str):
    vn = VariableNames(v)
    fig, ax = plt.subplots()
    xx = list(range(c.T))
    ax.plot(xx, df[get_name_for_list(vn.A_f[0])], color='blue', label='arrival')
    ax.plot(xx, df[get_name_for_list(vn.S_f[0])], color='red', label='service')
    if(hasattr(v, 'W')):
        ubl = []
        lbl = []
        for t in range(c.T):
            if(t >= c.D):
                lbl.append(get_raw_value(m.eval(v.C0 + c.C * (t-c.D) - v.W[t-c.D])))
            else:
                lbl.append(get_raw_value(m.eval(v.C0 + c.C * (t-c.D) - v.W[t])))
            ubl.append(get_raw_value(m.eval(v.C0 + c.C * t - v.W[t])))
        ax.plot(xx, lbl, color='black', alpha=0.5, ls='--')
        ax.plot(xx, ubl, color='black', alpha=0.5, ls='--')
    ax.legend()
    ax.set_ylim(bottom=float(m.eval(v.S[0]).as_fraction()))
    ax.grid(True)
    fig.set_tight_layout(True)
    fig.savefig(fpath, bbox_inches='tight', pad_inches=0.01)


class BaseLink:

    class LinkVariables(Variables):
        def bq(self, t: int):
            assert t >= 0 and t <= self.c.T-1
            return self.A[t] - self.L[t] - (self.C0 + self.c.C * t - self.W[t])

        def q(self, t: int):
            assert t >= 0 and t <= self.c.T-1
            return self.A[t] - self.L[t] - self.S[t]

        def __init__(self, c: ModelConfig, s: MySolver,
                     name: Optional[str] = None):
            super().__init__(c, s, name)
            self.c = c
            pre = self.pre

            # self.derived_expressions()

            if (c.beliefs):
                self.utilized = np.array([[
                    s.Bool(f"{pre}utilized_{n},{t}")
                    for t in range(c.T)]
                    for n in range(c.N)])
                self.utilized_cummulative = np.array([[[
                    s.Bool(f"{pre}utilized_cummulative_{n},{et},{st}")
                    for et in range(c.T)]
                    for st in range(c.T)]
                    for n in range(c.N)])
                self.max_measured_c = np.array([[[
                    s.Real(f"{pre}max_measured_c_{n},{et},{st}")
                    for et in range(c.T)]
                    for st in range(c.T)]
                    for n in range(c.N)])
                self.min_measured_c = np.array([[[
                    s.Real(f"{pre}min_measured_c_{n},{et},{st}")
                    for et in range(c.T)]
                    for st in range(c.T)]
                    for n in range(c.N)])

        # def derived_expressions(self):
        #     c = self.c

        #     if (c.beliefs):
        #         self._initial_minc_consistent = z3.And([self.min_c[n][0] <= c.C
        #                                                 for n in range(c.N)])
        #         self._initial_maxc_consistent = z3.And([self.max_c[n][0] >= c.C
        #                                                 for n in range(c.N)])
        #         self.initial_beliefs_consistent = z3.And(
        #             self._initial_minc_consistent, self._initial_maxc_consistent)
        #         self._final_minc_consistent = z3.And([self.min_c[n][-1] <= c.C
        #                                               for n in range(c.N)])
        #         self._final_maxc_consistent = z3.And([self.max_c[n][-1] >= c.C
        #                                               for n in range(c.N)])
        #         self.final_beliefs_consistent = z3.And(
        #             self._final_minc_consistent, self._final_maxc_consistent)


    class LinkModelConfig(ModelConfig):
        pass

    @staticmethod
    def check_config(cc: CegisConfig):
        # Don't use both of them together as they will produce different values for
        # non-deterministically chosen queing delays.
        assert not (cc.template_queue_bound and cc.template_qdel)

    def setup_model_config(self, cc: CegisConfig):
        self.check_config(cc)
        c = self.LinkModelConfig.default()
        c.compose = True
        c.cca = cc.cca
        c.simplify = False
        c.C = cc.C
        c.T = cc.T
        c.R = cc.R
        c.D = cc.D
        c.compose = cc.compose

        # Add a prefix to all names so we can have multiple Variables instances
        # in one solver
        if cc.name is None:
            pre = ""
        else:
            pre = cc.name + "__"

        # Signals
        c.loss_oracle = cc.template_loss_oracle

        # environment
        c.deterministic_loss = cc.deterministic_loss
        if(cc.infinite_buffer):
            c.buf_max = None
        elif(cc.dynamic_buffer):
            c.buf_max = z3.Real(f'{pre}buf_size')
        else:
            c.buf_max = cc.buffer_size_multiplier * c.C * (c.R + c.D)
        c.buf_min = c.buf_max
        c.N = cc.N

        if(cc.template_queue_bound):
            c.calculate_qbound = True
        if(c.N > 1 or cc.template_qdel):
            c.calculate_qdel = True
        c.mode_switch = cc.template_mode_switching
        c.feasible_response = cc.feasible_response

        c.app_limited = cc.app_limited
        c.app_fixed_avg_rate = cc.app_fixed_avg_rate
        c.app_rate = cc.app_rate
        c.app_burst_factor = cc.app_burst_factor

        c.beliefs = cc.template_beliefs
        c.beliefs_use_buffer = cc.template_beliefs_use_buffer
        c.beliefs_use_max_buffer = cc.template_beliefs_use_max_buffer
        c.fix_stale__min_c = cc.fix_stale__min_c
        c.fix_stale__max_c = cc.fix_stale__max_c
        c.min_maxc_minc_gap_mult = cc.min_maxc_minc_gap_mult
        c.maxc_minc_change_mult = cc.maxc_minc_change_mult

        c.send_min_alpha = cc.send_min_alpha

        return c

    def setup_environment(self, c: ModelConfig, v: Variables):
        s = MySolver()
        s.warn_undeclared = False

        monotone_env(c, s, v)
        service_waste(c, s, v)
        if(not c.deterministic_loss):
            loss_non_deterministic(c, s, v)
        # else:
        #     if(c.N == 1):
        #         # If loss happens at t=0, then bottleneck queue must be full at t=0.
        #         s.add(z3.Implies(
        #             v.Ld_f[0][0] != v.L_f[0][0],
        #             v.A[0] - v.L[0] - (v.C0 - v.W[0]) == c.buf_min))
        if(not c.loss_oracle):
            loss_detected(c, s, v)
        if(c.calculate_qdel):
            calculate_qdel_env(c, s, v)  # How to pick initial qdels
            # non-deterministically.
        if(c.calculate_qbound):
            calculate_qbound_env(c, s, v)  # How to pick initial qbounds
            # non-deterministically.
        if(c.N > 1):
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
        if(c.feasible_response):
            service_choice(c, s, v)
        if(c.beliefs):
            update_beliefs(c, s, v)

        return s

    def get_base_cegis_vars(self,
        cc: CegisConfig, c: ModelConfig, v: Variables
    ) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
        history = cc.history
        verifier_vars = flatten(
                [v.A_f[:, :history], v.c_f[:, :history], v.r_f[:, :history], v.W,
                v.alpha, v.C0])
        definition_vars = flatten(
                [v.A_f[:, history:], v.A, v.c_f[:, history:],
                v.r_f[:, history:], v.S, v.L])

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
        history = cc.history
        verifier_vars, definition_vars = self.get_base_cegis_vars(cc, c, v)
        assert isinstance(v, BaseLink.LinkVariables)

        if(not c.loss_oracle):
            verifier_vars.append(v.dupacks)
            definition_vars.extend(flatten(v.timeout_f))

        if(not c.compose):
            verifier_vars.append(v.epsilon)
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
            # Non-determinisitic loss
            assert c.loss_oracle
            verifier_vars.extend(flatten([v.L_f, v.Ld_f[:, :c.R]]))
            definition_vars.extend(flatten([v.Ld_f[:, c.R:]]))

        if(isinstance(c.buf_min, z3.ExprRef)):
            verifier_vars.append(c.buf_min)

        if(c.mode_switch):
            definition_vars.extend(flatten(v.mode_f[:, 1:]))
            verifier_vars.extend(flatten(v.mode_f[:, :1]))

        if (c.beliefs):
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
            definition_vars.extend(flatten([
                v.utilized, v.utilized_cummulative,
                v.max_measured_c, v.min_measured_c
            ]))

        if(c.app_limited):
            verifier_vars.extend(flatten(v.app_limits))
            verifier_vars.append(v.app_rate)
            if(c.app_fixed_avg_rate and c.beliefs):
                verifier_vars.extend(flatten(
                    [v.max_app_rate[:, :1], v.min_app_rate[:, :1]]))
                definition_vars.extend(flatten(
                    [v.max_app_rate[:, 1:], v.min_app_rate[:, 1:]]))

        return verifier_vars, definition_vars

    def setup_cegis_input(self, cc: CegisConfig, name=None):
        if(name is not None):
            cc.name = name
        c = self.setup_model_config(cc)
        s = MySolver()
        s.warn_undeclared = False
        v = self.LinkVariables(c, s, cc.name)

        ccac_domain = z3.And(*s.assertion_list)
        assert isinstance(ccac_domain, z3.BoolRef)
        se = self.setup_environment(c, v)
        sd = self.setup_definitions(c, v)
        ccac_definitions = z3.And(*sd.assertion_list)
        assert isinstance(ccac_definitions, z3.BoolRef)
        # not_too_adversarial_init_cwnd(cc, c, se, v)
        environment = z3.And(*se.assertion_list)
        assert isinstance(environment, z3.BoolRef)

        verifier_vars, definition_vars = self.get_cegis_vars(cc, c, v)

        return (c, s, v,
                ccac_domain, ccac_definitions, environment,
                verifier_vars, definition_vars)

    @staticmethod
    def get_cex_df(
        counter_example: z3.ModelRef, v: Variables,
        vn: VariableNames, c: ModelConfig
    ) -> pd.DataFrame:
        return get_cex_df(counter_example, v, vn, c)
