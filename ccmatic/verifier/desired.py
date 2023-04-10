import z3
from dataclasses import dataclass
from typing import Optional, List

from ccac.config import ModelConfig
from ccac.variables import Variables
from ccmatic.cegis import CegisConfig
from ccmatic.verifier import logger, SteadyStateVariable
from ccmatic.verifier.cbr_delay import CBRDelayLink
from cegis import rename_vars
from cegis.util import z3_min


@dataclass
class DesiredContainer:
    desired_necessary: Optional[z3.BoolRef] = None
    desired_in_ss: Optional[z3.BoolRef] = None
    desired_invariant: Optional[z3.BoolRef] = None
    desired_belief_invariant: Optional[z3.BoolRef] = None

    fefficient: Optional[z3.BoolRef] = None
    bounded_queue: Optional[z3.BoolRef] = None
    bounded_loss_count: Optional[z3.BoolRef] = None
    bounded_large_loss_count: Optional[z3.BoolRef] = None
    bounded_loss_amount: Optional[z3.BoolRef] = None

    ramp_up_cwnd: Optional[z3.BoolRef] = None
    ramp_down_cwnd: Optional[z3.BoolRef] = None

    ramp_down_queue: Optional[z3.BoolRef] = None
    ramp_up_queue: Optional[z3.BoolRef] = None

    ramp_down_bq: Optional[z3.BoolRef] = None
    ramp_up_bq: Optional[z3.BoolRef] = None

    loss_count: Optional[z3.ArithRef] = None
    large_loss_count: Optional[z3.ArithRef] = None
    loss_amount: Optional[z3.ArithRef] = None

    steady_state_variables: Optional[List[SteadyStateVariable]] = None
    steady_state_exists: Optional[z3.BoolRef] = None

    atleast_one_outside: Optional[z3.BoolRef] = None
    none_degrade: Optional[z3.BoolRef] = None
    atleast_one_moves_inside: Optional[z3.BoolRef] = None

    init_inside: Optional[z3.BoolRef] = None
    final_inside: Optional[z3.BoolRef] = None

    fast_decrease: Optional[z3.BoolRef] = None
    fast_increase: Optional[z3.BoolRef] = None

    beliefs_remain_consistent: Optional[z3.BoolRef] = None
    beliefs_improve: Optional[z3.BoolRef] = None
    stale_minc_improves: Optional[z3.BoolRef] = None
    stale_maxc_improves: Optional[z3.BoolRef] = None
    initial_bw_consistent: Optional[z3.BoolRef] = None
    final_bw_consistent: Optional[z3.BoolRef] = None
    final_bw_moves_consistent: Optional[z3.BoolRef] = None

    def rename_vars(self, var_list: List[z3.ExprRef], template: str):
        conds = {
            "fefficient": self.fefficient,
            "bounded_queue": self.bounded_queue,
            "bounded_loss_count": self.bounded_loss_count,
            "bounded_loss_amount": self.bounded_loss_amount,
            "ramp_up_cwnd": self.ramp_up_cwnd,
            "ramp_down_cwnd": self.ramp_down_cwnd,
            "ramp_up_bq": self.ramp_up_bq,
            "ramp_down_bq": self.ramp_down_bq,
            "ramp_up_queue": self.ramp_up_queue,
            "ramp_down_queue": self.ramp_down_queue,
            "loss_count": self.loss_count,
            "loss_amount": self.loss_amount,
        }
        for attr_name, cond in conds.items():
            if(isinstance(cond, bool)):
                continue
            new_cond = rename_vars(cond, var_list, template)
            setattr(self, attr_name, new_cond)

    def to_string(self, cc: CegisConfig,
                  c: ModelConfig, model: z3.ModelRef) -> str:
        conds = {
            "fefficient": self.fefficient,
            "bounded_queue": self.bounded_queue,
            "bounded_loss_count": self.bounded_loss_count,
            "bounded_large_loss_count": self.bounded_large_loss_count,
            "bounded_loss_amount": self.bounded_loss_amount,
            "ramp_up_cwnd": self.ramp_up_cwnd,
            "ramp_down_cwnd": self.ramp_down_cwnd,
            "ramp_up_bq": self.ramp_up_bq,
            "ramp_down_bq": self.ramp_down_bq,
            "ramp_up_queue": self.ramp_up_queue,
            "ramp_down_queue": self.ramp_down_queue,
            "loss_count": self.loss_count,
            "large_loss_count": self.large_loss_count,
            "loss_amount": self.loss_amount,

            "atleast_one_outside": self.atleast_one_outside,
            "none_degrade": self.none_degrade,
            "atleast_one_moves_inside": self.atleast_one_moves_inside,

            "init_inside": self.init_inside,
            "final_inside": self.final_inside,

            "fast_decrease": self.fast_decrease,
            "fast_increase": self.fast_increase,

            "beliefs_remain_consistent": self.beliefs_remain_consistent,
            "beliefs_improve": self.beliefs_improve,
            "stale_minc_improves": self.stale_minc_improves,
            "stale_maxc_improves": self.stale_maxc_improves,
            "initial_bw_consistent": self.initial_bw_consistent,
            "final_bw_consistent": self.final_bw_consistent,
            "final_bw_moves_consistent": self.final_bw_moves_consistent,
        }

        def get_val(cond):
            if(isinstance(cond, bool)):
                return cond
            else:
                return model.eval(cond)

        cond_list = []
        term_count = 0
        for cond_name, cond in conds.items():
            if(cond is not None):
                cond_list.append(
                    "{}={}".format(cond_name, get_val(cond)))
                term_count += 1
                if(term_count % 6 == 0):
                    cond_list.append("\n")
        ret = ", ".join(cond_list)
        if(self.steady_state_variables):
            for sv in self.steady_state_variables:
                ret += "\n{}: [{}, {}]".format(
                    sv.name, model.eval(sv.lo), model.eval(sv.hi))
        return ret


def get_stale_minc_improves(cc: CegisConfig, c: ModelConfig, v: Variables):
    """
    TODO: currently we are assumignt hat when beleifs become invalid,
    we correct them outside CCAC. We can encode this within CCAC.
    """
    beliefs_bad_list: List[z3.BoolRef] = []
    beliefs_eventually_reset_list: List[z3.BoolRef] = []
    beliefs_become_invalid_list: List[z3.BoolRef] = []
    for n in range(c.N):
        beliefs_bad_list.append(
            v.min_c[n][0] > c.C)
        beliefs_eventually_reset_list.append(
            v.min_c[n][-1] < v.min_c[n][0])
        beliefs_become_invalid_list.append(
            v.min_c[n][-1] > v.max_c[n][-1])

    beliefs_bad = z3.Or(*beliefs_bad_list)
    # TODO: should this be AND or OR? Should all flows reset or any one?
    beliefs_eventually_reset = z3.And(*beliefs_eventually_reset_list)
    beliefs_become_invalid = z3.And(*beliefs_become_invalid_list)
    # stale_minc_improves = z3.Implies(beliefs_bad, beliefs_eventually_reset)
    return beliefs_bad, z3.Or(beliefs_eventually_reset, beliefs_become_invalid)


def get_stale_maxc_improves(cc: CegisConfig, c: ModelConfig, v: Variables):
    beliefs_bad_list: List[z3.BoolRef] = []
    beliefs_eventually_reset_list: List[z3.BoolRef] = []
    beliefs_become_invalid_list: List[z3.BoolRef] = []
    for n in range(c.N):
        beliefs_bad_list.append(
            v.max_c[n][0] < c.C)
        beliefs_eventually_reset_list.append(
            v.max_c[n][-1] > v.max_c[n][0])
        beliefs_become_invalid_list.append(
            v.min_c[n][-1] > v.max_c[n][-1])

    beliefs_bad = z3.Or(*beliefs_bad_list)
    # TODO: should this be AND or OR? Should all flows reset or any one?
    beliefs_eventually_reset = z3.And(*beliefs_eventually_reset_list)
    beliefs_become_invalid = z3.And(*beliefs_become_invalid_list)
    # stale_maxc_improves = z3.Implies(beliefs_bad, beliefs_eventually_reset)
    return beliefs_bad, z3.Or(beliefs_eventually_reset, beliefs_become_invalid)


def get_bw_beliefs_move_consistent(cc: CegisConfig, c: ModelConfig, v: Variables):
    first = 0  # cc.history
    speed = 1
    # For multi-flow, would need to think if both flows need to improve or just one.
    final_moves_consistent = z3.Or([
        z3.Or(
            z3.And(
                v.max_c[n][first] < c.C,
                z3.Or(v.max_c[n][-1] > speed * v.max_c[n][first],
                      v.min_c[n][-1] > speed * v.min_c[n][first])),
            z3.And(
                v.min_c[n][first] > c.C,
                z3.Or(v.min_c[n][-1] * speed < v.min_c[n][first],
                      v.max_c[n][-1] * speed < v.max_c[n][first]))
        ) for n in range(c.N)
    ])
    initial_consistent = z3.And(
        [v.max_c[n][first] >= c.C for n in range(c.N)] +
        [v.min_c[n][first] <= c.C for n in range(c.N)]
    )
    final_consistent = z3.And(
        [v.max_c[n][-1] >= c.C for n in range(c.N)] +
        [v.min_c[n][-1] <= c.C for n in range(c.N)]
    )
    assert isinstance(initial_consistent, z3.BoolRef)
    assert isinstance(final_consistent, z3.BoolRef)
    assert isinstance(final_moves_consistent, z3.BoolRef)
    return initial_consistent, final_consistent, final_moves_consistent


# Deprecated
def get_beliefs_improve_old(cc: CegisConfig, c: ModelConfig, v: Variables):
    assert(c.beliefs)

    # Since initial beliefs are non adversarial. We can afford to just compare
    # against the verifier chosen beliefs.

    # Correctness wise, verifier could just
    # check for init conditions at t = first, so we are not losing traces. The
    # only benefit is that if no improvement happens after t = first, then we
    # must deliver on objective or improve within T. Checking from t = 0 allows
    # us to improve within T + first (basically allowing us to keep a smaller T).
    first = 0  # cc.history
    none_degrade_list = []
    atleast_one_improves_list = []

    for n in range(c.N):
        atleast_one_improves_list.extend([
            v.max_c[n][c.T-1] < v.max_c[n][first],
            v.min_c[n][c.T-1] > v.min_c[n][first],
        ])
        if(c.buf_min is not None and c.beliefs_use_buffer):
            atleast_one_improves_list.append(
                v.min_buffer[n][c.T-1] > v.min_buffer[n][first])
            if(c.beliefs_use_max_buffer):
                atleast_one_improves_list.append(
                    v.max_buffer[n][c.T-1] < v.max_buffer[n][first])

        none_degrade_list.extend([
            v.max_c[n][c.T-1] <= v.max_c[n][first],
            v.min_c[n][c.T-1] >= v.min_c[n][first],
        ])
        if(c.buf_min is not None and c.beliefs_use_buffer):
            none_degrade_list.append(
                v.min_buffer[n][c.T-1] >= v.min_buffer[n][first])
            if(c.beliefs_use_max_buffer):
                none_degrade_list.append(
                    v.max_buffer[n][c.T-1] <= v.max_buffer[n][first])

    none_degrade = z3.And(*none_degrade_list)
    atleast_one_improves = z3.Or(*atleast_one_improves_list)

    return atleast_one_improves, none_degrade


def get_beliefs_remain_consistent(cc: CegisConfig, c: ModelConfig, v: Variables):
    assert(c.beliefs)

    final_consistent_list = []

    for n in range(c.N):
        final_consistent_list.extend([
            v.max_c[n][c.T-1] >= c.C,
            v.min_c[n][c.T-1] <= c.C
        ])
        if (c.buf_min is not None and c.beliefs_use_buffer):
            final_consistent_list.append(
                v.min_buffer[n][c.T-1] <= c.buf_min / c.C)
            if (c.beliefs_use_max_buffer):
                final_consistent_list.append(
                    v.max_buffer[n][c.T-1] >= c.buf_min / c.C)
        if (c.app_limited and c.app_fixed_avg_rate):
            final_consistent_list.extend([
                v.max_app_rate[n][c.T-1] >= v.app_rate,
                v.min_app_rate[n][c.T-1] <= v.app_rate
            ])

        if (isinstance(v, CBRDelayLink.LinkVariables)):
            assert isinstance(c, CBRDelayLink.LinkModelConfig)
            final_consistent_list.append(v.final_minc_lambda_consistent)
            final_consistent_list.append(v.final_bq_consistent)

    final_consistent = z3.And(*final_consistent_list)
    assert(isinstance(final_consistent, z3.BoolRef))
    return final_consistent


def get_beliefs_improve(cc: CegisConfig, c: ModelConfig, v: Variables):
    assert(c.beliefs)

    # This constraint is only used when beliefs are consistent to begin with.
    # Since initial beliefs are non adversarial. We can afford to just compare
    # against the verifier chosen beliefs.

    # Correctness wise, verifier could just
    # check for init conditions at t = first, so we are not losing traces. The
    # only benefit is that if no improvement happens after t = first, then we
    # must deliver on objective or improve within T. Checking from t = 0 allows
    # us to improve within T + first (basically allowing us to keep a smaller T).

    # An individual beleif might degrade but the range can still shrink, and
    # this is fine for overall proof to work.
    first = 0  # cc.history
    none_expand_list = []
    atleast_one_shrinks_list = []

    for n in range(c.N):
        atleast_one_shrinks_list.extend([
            v.max_c[n][c.T-1] - v.min_c[n][c.T-1] <
            v.max_c[n][first] - v.min_c[n][first],
        ])
        if(c.buf_min is not None and c.beliefs_use_buffer):
            if(c.beliefs_use_max_buffer):
                atleast_one_shrinks_list.extend([
                    v.max_buffer[n][c.T-1] - v.min_buffer[n][c.T-1] <
                    v.max_buffer[n][first] - v.min_buffer[n][first]
                ])
            else:
                atleast_one_shrinks_list.append(
                    v.min_buffer[n][c.T-1] > v.min_buffer[n][first])
        if(c.app_limited and c.app_fixed_avg_rate):
            atleast_one_shrinks_list.extend([
                v.max_app_rate[n][c.T-1] - v.min_app_rate[n][c.T-1] <
                v.max_app_rate[n][first] - v.min_app_rate[n][first]
            ])
        if(isinstance(v, CBRDelayLink.LinkVariables)):
            atleast_one_shrinks_list.append(
                v.min_c_lambda[n][c.T-1] > v.min_c_lambda[n][first])

        none_expand_list.extend([
            v.max_c[n][c.T-1] - v.min_c[n][c.T-1] <=
            v.max_c[n][first] - v.min_c[n][first]
        ])
        if(c.buf_min is not None and c.beliefs_use_buffer):
            if(c.beliefs_use_max_buffer):
                none_expand_list.extend([
                    v.max_buffer[n][c.T-1] - v.min_buffer[n][c.T-1] <=
                    v.max_buffer[n][first] - v.min_buffer[n][first]
                ])
            else:
                none_expand_list.append(
                    v.min_buffer[n][c.T-1] >= v.min_buffer[n][first])
        if(c.app_limited and c.app_fixed_avg_rate):
            none_expand_list.extend([
                v.max_app_rate[n][c.T-1] - v.min_app_rate[n][c.T-1] <=
                v.max_app_rate[n][first] - v.min_app_rate[n][first]
            ])
        if(isinstance(v, CBRDelayLink.LinkVariables)):
            none_expand_list.append(
                v.min_c_lambda[n][c.T-1] >= v.min_c_lambda[n][first])

    none_expand = z3.And(*none_expand_list)
    atleast_one_shrinks = z3.Or(*atleast_one_shrinks_list)

    return none_expand, atleast_one_shrinks


def get_belief_invariant(cc: CegisConfig, c: ModelConfig, v: Variables):
    # d = get_desired_in_ss(cc, c, v)
    d = get_desired_necessary(cc, c, v)
    final_consistent = get_beliefs_remain_consistent(cc, c, v)
    none_expand, atleast_one_shrinks = \
        get_beliefs_improve(cc, c, v)

    d.beliefs_remain_consistent = final_consistent
    d.beliefs_improve = z3.And(
        none_expand, atleast_one_shrinks)
    invariant = z3.And(d.beliefs_remain_consistent,
                       z3.Or(d.beliefs_improve, d.desired_necessary))
    assert isinstance(invariant, z3.BoolRef)
    d.desired_belief_invariant = invariant

    if (cc.desired_no_large_loss):
        # We don't want large loss even when potentially probing (improving beliefs)
        invariant = z3.And(invariant,
                           z3.Or(d.bounded_large_loss_count, d.ramp_down_cwnd,
                                 d.ramp_down_queue, d.ramp_down_bq))
        assert isinstance(invariant, z3.BoolRef)
        d.desired_belief_invariant = invariant

    # if(c.fix_stale__max_c):
    #     # TODO: Can’t remove non degrade. CCA might improve X and degrade Y, and
    #     # then improve Y and degrade X. This loop may keep repeating and desired
    #     # properties are violated all the time.
    #     # invariant = atleast_one_improves

    #     d.beliefs_improve = atleast_one_improves
    #     invariant = z3.Or(d.desired_necessary, d.beliefs_improve)
    #     assert isinstance(invariant, z3.BoolRef)

    #     beliefs_bad, beliefs_reset = get_beliefs_reset(cc, c, v)
    #     d.desired_belief_invariant = z3.If(beliefs_bad, beliefs_reset, invariant)

    # if(c.fix_stale__max_c or c.fix_stale__min_c):
    #     logger.info("Probe invariant")
    #     initial_consistent, final_consistent, final_moves_consistent = \
    #         get_bw_beliefs_move_consistent(cc, c, v)
    #     d.initial_bw_consistent = initial_consistent
    #     d.final_bw_consistent = final_consistent
    #     d.final_bw_moves_consistent = final_moves_consistent
    #     final_invalid = z3.Or([v.max_c[n][-1] < v.min_c[n][-1]
    #                           for n in range(c.N)])
    #     invariant = z3.And(
    #         z3.Not(final_invalid),
    #         z3.Implies(z3.Not(d.initial_bw_consistent), z3.Or(
    #             d.final_bw_consistent, d.final_bw_moves_consistent,
    #             d.desired_in_ss))
    #     )
    #     assert isinstance(invariant, z3.BoolRef)
    #     d.desired_belief_invariant = invariant

    if(c.fix_stale__min_c):
        logger.info("Old invariant")
        minc_is_stale, stale_minc_improves = get_stale_minc_improves(cc, c, v)
        d.stale_minc_improves = stale_minc_improves
        invariant = z3.If(minc_is_stale,
                          z3.Or(stale_minc_improves,
                                d.desired_in_ss, d.beliefs_improve),
                          invariant)
        # invariant = z3.Or(invariant, stale_minc_improves)
        assert isinstance(invariant, z3.BoolRef)
        d.desired_belief_invariant = invariant

    if(c.fix_stale__max_c):
        logger.info("Old invariant")
        maxc_is_stale, stale_maxc_improves = get_stale_maxc_improves(cc, c, v)
        d.stale_maxc_improves = stale_maxc_improves
        invariant = z3.If(maxc_is_stale,
                          z3.Or(stale_maxc_improves,
                                d.desired_in_ss, d.beliefs_improve),
                          invariant)
        assert isinstance(invariant, z3.BoolRef)
        d.desired_belief_invariant = invariant

    if(isinstance(v, CBRDelayLink.LinkVariables)):
        assert isinstance(c, CBRDelayLink.LinkModelConfig)

        if(c.fix_stale__min_c_lambda):
            invariant = z3.If(
                z3.Not(v.initial_minc_lambda_consistent),
                z3.Or(v.stale_minc_lambda_improves,
                      v.final_minc_lambda_consistent, d.desired_necessary),
                invariant)
            assert isinstance(invariant, z3.BoolRef)
            d.desired_belief_invariant = invariant

        if(c.fix_stale__bq_belief):
            invariant = z3.If(
                z3.Not(v.initial_bq_consistent),
                z3.Or(v.stale_bq_belief_improves,
                      v.final_bq_consistent, d.desired_necessary),
                invariant)
            assert isinstance(invariant, z3.BoolRef)
            d.desired_belief_invariant = invariant

        invariant = z3.And(invariant, v.final_minc_lambda_valid, v.final_bq_valid)
        assert isinstance(invariant, z3.BoolRef)
        d.desired_belief_invariant = invariant

    return d


def ramp_up_when_cwnd_reset_fi(cc: CegisConfig, c: ModelConfig, v: Variables):
    first = cc.history

    total_first_cwnd = 0
    total_second_cwnd = 0
    total_second_last_cwnd = 0
    total_last_cwnd = 0
    assert c.T-2 > first+1
    for n in range(c.N):
        total_first_cwnd += v.c_f[n][first]
        total_second_cwnd += v.c_f[n][first+1]
        total_second_last_cwnd += v.c_f[n][c.T-2]
        total_last_cwnd += v.c_f[n][c.T-1]
    assert isinstance(total_first_cwnd, z3.ArithRef)
    assert isinstance(total_second_cwnd, z3.ArithRef)
    assert isinstance(total_second_last_cwnd, z3.ArithRef)
    assert isinstance(total_last_cwnd, z3.ArithRef)

    min_initial_cwnd = z3_min(total_first_cwnd, total_second_cwnd)
    min_final_cwnd = z3_min(total_second_last_cwnd, total_last_cwnd)
    assert isinstance(min_initial_cwnd, z3.ArithRef)
    assert isinstance(min_final_cwnd, z3.ArithRef)

    ramp_up_cwnd = min_final_cwnd > min_initial_cwnd
    ramp_down_cwnd = min_final_cwnd < min_initial_cwnd
    return ramp_up_cwnd, ramp_down_cwnd


def get_desired_necessary(
        cc: CegisConfig, c: ModelConfig, v: Variables):
    first = cc.history

    d = get_desired_in_ss(cc, c, v)

    # Induction invariants
    total_final_cwnd = 0
    total_initial_cwnd = 0
    for n in range(c.N):
        total_initial_cwnd += v.c_f[n][first]
        total_final_cwnd += v.c_f[n][-1]

    total_final_rate = 0
    total_initial_rate = 0
    for n in range(c.N):
        total_initial_rate += v.r_f[n][first]
        total_final_rate += v.r_f[n][-1]

    # TODO: check if this is the right invariant for rate based CCAs.
    #  For window based CCAs, pacing is const, so this should be fine.
    if(cc.rate_or_window != 'rate'):
        d.ramp_up_cwnd = z3.And(
            total_final_cwnd > total_initial_cwnd,
            total_final_rate > total_initial_rate)
        if(cc.template_fi_reset):
            ru, rd = ramp_up_when_cwnd_reset_fi(cc, c, v)
            d.ramp_up_cwnd = ru
            d.ramp_down_cwnd = rd
        d.ramp_down_cwnd = z3.And(
            total_final_cwnd < total_initial_cwnd,
            total_final_rate < total_initial_rate)
    else:
        d.ramp_up_cwnd = z3.And(total_final_rate > total_initial_rate)
        d.ramp_down_cwnd = z3.And(total_final_rate < total_initial_rate)

    d.ramp_down_queue = (v.A[-1] - v.L[-1] - v.S[-1] <
                         v.A[first] - v.L[first] - v.S[first])
    d.ramp_up_queue = (v.A[-1] - v.L[-1] - v.S[-1] >
                       v.A[first] - v.L[first] - v.S[first])

    if(hasattr(v, 'C0') and hasattr(v, 'W')):
        d.ramp_down_bq = (
            (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
            < (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))
        d.ramp_up_bq = (
            (v.A[-1] - v.L[-1] - (v.C0 + c.C * (c.T-1) - v.W[-1]))
            > (v.A[first] - v.L[first] - (v.C0 + c.C * first - v.W[first])))
    else:
        # For ideal link
        d.ramp_down_bq = d.ramp_down_queue
        d.ramp_up_bq = d.ramp_up_queue

    d.desired_necessary = z3.And(
        z3.Or(d.fefficient, d.ramp_up_cwnd,
              d.ramp_up_queue, d.ramp_up_bq),
        z3.Or(d.bounded_queue, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq),
        z3.Or(d.bounded_loss_count, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq),
        z3.Or(d.bounded_large_loss_count, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq),
        z3.Or(d.bounded_loss_amount, d.ramp_down_cwnd,
              d.ramp_down_queue, d.ramp_down_bq))

    # # Simpler desired necessary
    # d.desired_necessary = z3.And(
    #     z3.Or(d.fefficient, d.ramp_up_cwnd),
    #     z3.Or(d.bounded_queue, d.ramp_down_bq),
    #     z3.Or(d.bounded_loss_count, d.ramp_down_cwnd),
    #     z3.Or(d.bounded_large_loss_count, d.ramp_down_cwnd),
    #     z3.Or(d.bounded_loss_amount, d.ramp_down_cwnd))

    # Above, instead of d.ramp_down_bq, we could use d.ramp_down_cwnd or
    # d.ramp_down_queue.

    return d


def get_desired_in_ss(cc: CegisConfig, c: ModelConfig, v: Variables):
    d = DesiredContainer()

    first = cc.history
    cond_list = []
    for t in range(first, c.T):
        # Queue seen by a new packet should not be
        # more than desired_queue_bound
        cond_list.append(
            v.A[t] - v.L[t] - v.S[t] <=
            cc.desired_queue_bound_multiplier * c.C * (c.R + c.D)
            + cc.desired_queue_bound_alpha * v.alpha)
    d.bounded_queue = z3.And(*cond_list)

    assert first >= 1
    d.fefficient = (
        v.S[-1] - v.S[first-1] >=
        cc.desired_util_f * c.C * (c.T-1-(first-1)-c.D))

    if(c.app_limited):
        # all_app_limited = z3.And(*[
        #     v.A_f[n][c.T-1] == v.app_limits[n][c.T-1]
        #     for n in range(c.N)])
        # d.fefficient = z3.Or(d.fefficient, all_app_limited)

        # I can't expect 50% link utilization if app sends at 30% link rate. App
        # can blast packets on the last second, so I can't expect app saturation
        # at the last second.
        d.fefficient = (v.S[-1] - v.S[first-1] >=
                        (c.T-1-(first-1)-c.D)
                        * z3_min(v.app_rate, cc.desired_util_f * c.C))

    loss_list: List[z3.BoolRef] = []
    for t in range(first, c.T):
        loss_list.append(v.L[t] > v.L[t-1])
    d.loss_count = z3.Sum(*loss_list)
    d.bounded_loss_count = d.loss_count <= cc.desired_loss_count_bound

    large_loss_list: List[z3.BoolRef] = []
    for t in range(first, c.T):
        large_loss_list.append(v.L[t] > v.L[t-1] + v.alpha * cc.desired_loss_amount_bound_alpha)
    d.large_loss_count = z3.Sum(*large_loss_list)
    d.bounded_large_loss_count = d.large_loss_count <= cc.desired_large_loss_count_bound

    d.loss_amount = v.L[-1] - v.L[first]
    d.bounded_loss_amount = (
        d.loss_amount <=
        cc.desired_loss_amount_bound_multiplier * (c.C * (c.R + c.D))
        + cc.desired_loss_amount_bound_alpha * v.alpha)


    d.desired_in_ss = z3.And(d.fefficient, d.bounded_queue,
                             d.bounded_loss_count, d.bounded_loss_amount)
    return d


def get_desired_ss_invariant(cc: CegisConfig, c: ModelConfig, v: Variables):
    d = get_desired_in_ss(cc, c, v)

    total_final_cwnd = 0
    total_initial_cwnd = 0
    for n in range(c.N):
        total_initial_cwnd += v.c_f[n][cc.history]
        total_final_cwnd += v.c_f[n][-1]

    assert(isinstance(total_final_cwnd, z3.ArithRef))
    assert(isinstance(total_initial_cwnd, z3.ArithRef))
    d.steady_state_variables = [
        SteadyStateVariable(
            'cwnd',
            total_initial_cwnd,
            total_final_cwnd,
            z3.Int('SSThresh_cwnd_lo'),
            z3.Int('SSThresh_cwnd_hi')),

        # Queue
        SteadyStateVariable(
            'queue',
            v.A[cc.history] - v.L[cc.history] - v.S[cc.history],
            v.A[c.T-1] - v.L[c.T-1] - v.S[c.T-1],
            z3.Int('SSThresh_queue_lo'),
            z3.Int('SSThresh_queue_hi'))

        # # Bottleneck Queue
        # SteadyStateVariable(
        #     'queue',
        #     v.A[cc.history] - v.L[cc.history]
        #     - (v.C0 + c.C * (cc.history) - v.W[cc.history]),
        #     v.A[c.T-1] - v.L[c.T-1]
        #     - (v.C0 + c.C * (c.T-1) - v.W[c.T-1]),
        #     z3.Int('SSThresh_queue_lo'),
        #     z3.Int('SSThresh_queue_hi'))
    ]
    d.steady_state_exists = get_steady_state_definitions(
        cc, c, v, d)
    inside_ss = z3.And(*[sv.init_inside()
                         for sv in d.steady_state_variables])

    d.desired_invariant = z3.And(d.steady_state_exists,
                                 z3.Implies(inside_ss, d.desired_in_ss))
    return d


def get_steady_state_definitions(
        cc: CegisConfig, c: ModelConfig, v: Variables,
        d: DesiredContainer):
    assert d.steady_state_variables
    assertions = []

    # # Initial in SS implies Final in SS
    # for sv in steady_state_variables:
    #     assertions.append(z3.Implies(
    #         z3.And(sv.initial >= sv.lo, sv.initial <= sv.hi),
    #         z3.And(sv.final >= sv.lo, sv.final <= sv.hi)))

    # At least one outside
    #     IMPLIES
    #         none should degrade AND
    #         atleast one that is outside must move towards inside
    d.atleast_one_outside = z3.Or(
        *[sv.init_outside() for sv in d.steady_state_variables])
    d.none_degrade = z3.And(*[sv.does_not_degrage()
                              for sv in d.steady_state_variables])
    d.atleast_one_moves_inside = \
        z3.Or(*[z3.And(sv.init_outside(), sv.strictly_improves())
                for sv in d.steady_state_variables])
    assertions.append(z3.Implies(
        d.atleast_one_outside,
        z3.And(
            d.none_degrade,
            d.atleast_one_moves_inside)))

    # All inside
    #    IMPLIES
    #         All remain inside
    d.init_inside = z3.And(*[sv.init_inside() for sv in d.steady_state_variables])
    d.final_inside = z3.And(*[sv.final_inside() for sv in d.steady_state_variables])
    assertions.append(z3.Implies(
        d.init_inside,
        d.final_inside
    ))

    ret = z3.And(*assertions)
    assert isinstance(ret, z3.BoolRef)
    return ret
