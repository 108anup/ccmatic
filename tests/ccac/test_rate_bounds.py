import matplotlib.pyplot as plt
import fractions
import logging
import math

import pandas as pd
import z3

from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from cegis.util import Metric, optimize_var
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

LINK_RATE = 100
R = 1
D = R


def test_rate_bounds(
        pacing_gain=2, tsteps=1, upper_bound=True, cwnd_gain: float=100):
    cc = CegisConfig()
    cc.C = LINK_RATE
    cc.R = R
    cc.D = D
    cc.T = 10
    cc.history = cc.R
    cc.infinite_buffer = False
    cc.dynamic_buffer = True
    cc.buffer_size_multiplier = 1
    cc.template_queue_bound = False
    cc.template_mode_switching = False
    cc.template_qdel = False
    cc.ideal_link = False

    # These are just placed holders, these are not used.
    cc.desired_util_f = 1
    cc.desired_queue_bound_multiplier = 1/2
    cc.desired_queue_bound_alpha = 3
    cc.desired_loss_count_bound = 3
    cc.desired_loss_amount_bound_multiplier = 0
    cc.desired_loss_amount_bound_alpha = 10

    cc.compose = True
    cc.cca = "none"
    cc.feasible_response = False

    link = CCmatic(cc)
    link.setup_config_vars()
    c, _, v, = link.c, link.s, link.v
    link.setup_cegis_loop(True, [], [], lambda x, y: "")

    sending_rate_constraint = z3.And(*[
        v.r_f[n][t] == cc.C * pacing_gain
        for t in range(cc.T) for n in range(cc.N)])
    cwnd_constraint = z3.And(*[
        v.c_f[n][t] == cwnd_gain * c.C * (c.R + c.D)
        for t in range(cc.T) for n in range(cc.N)])
    # Ensure we are only rate limited, never cwnd limited.

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.ccac_domain)
    verifier.add(link.ccac_definitions)
    verifier.add(link.environment)
    verifier.add(sending_rate_constraint)
    verifier.add(cwnd_constraint)
    # For convenience of looking at counter examples
    verifier.add(v.S[0] == 0)
    assert cc.R == cc.D
    # Assume link rate has been stagnant throughout.
    # So max queuing is sending_rate * R
    verifier.add(v.A[0] - v.L[0] - v.S[0] == 0)
    # verifier.add(v.A[0] - v.L[0] - v.S[0] <= cc.C * cc.R * pacing_gain)
    # verifier.add(v.A[0] - v.L[0] - v.S[0] <= cc.C * (cc.R + cc.D) * cwnd_gain)
    # verifier.add(v.A[0] - v.L[0] - v.S[0] <= cc.C * (cc.R + cc.D))
    # verifier.add(z3.And(*[v.A[t] - v.L[t] - v.S[t] == cc.C * sending_rate_multiplier * cc.R
    #                       for t in range(2)]))

    assert cc.T-1-tsteps >= cc.history
    assert cc.R == cc.D
    rate_estimate = z3.Real("rate_estimate")
    verifier.add(rate_estimate == (v.S[cc.T-1] - v.S[cc.T-1-tsteps])/(cc.D * tsteps))
    default_lo = 0
    default_hi = 100 * cc.C * (cc.R + cc.D)

    # verifier.add(rate_estimate == 20)
    # sat = verifier.check()
    # print(sat)
    # if(str(sat) == "sat"):
    #     print(link.get_counter_example_str(verifier.model(), []))
    # return
    # verifier.add(rate_estimate == 79)
    # print(verifier.check())
    # return

    def get_feasible_rate():
        sat = verifier.check()
        assert str(sat) == "sat"
        model = verifier.model()
        return float(model.eval(rate_estimate).as_fraction())

    if(upper_bound):
        default_lo = get_feasible_rate()
    else:
        default_hi = get_feasible_rate()

    optimization_list = [
        Metric(
            rate_estimate,
            default_lo, default_hi, 1e-6, upper_bound)
    ]

    GlobalConfig().logging_levels['cegis'] = logging.INFO
    logger = logging.getLogger('cegis')
    GlobalConfig().default_logger_setup(logger)

    metric = optimization_list[0]
    # Note reversed polarity of metric here as we want max value such that formula is sat.
    ret = optimize_var(verifier, metric.z3ExprRef, metric.lo, metric.hi, metric.eps, not metric.maximize)
    # ret = optimize_multi_var(verifier, optimization_list)
    # df = pd.DataFrame(ret)
    # sort_columns = [x.name() for x in optimization_list]
    # sort_order = [x.maximize for x in optimization_list]
    # df = df.sort_values(by=sort_columns, ascending=sort_order)

    print(pacing_gain, cwnd_gain, tsteps, ret)
    if(upper_bound):
        return ret[0]
    else:
        return ret[2]


def vary_sending_rate():
    all_records = []
    # sweep_range = [0.1, 0.25, fractions.Fraction(1, 3), 0.5, fractions.Fraction(2, 3), 0.75, 1, 2]
    sweep_range = sorted(list(set(
        [x/10 for x in range(1, 21)] + [fractions.Fraction(x/3) for x in range(1, 6)] +
        [fractions.Fraction(x/7) for x in range(1, 14)]
    )))
    sweep_range = sorted(list(set(
        [fractions.Fraction(x/3) for x in range(1, 6)]
    )))
    for pacing_gain in sweep_range:
        records = []
        n_tsteps = 1
        for tsteps in range(n_tsteps, n_tsteps+1):
            lower = test_rate_bounds(
                pacing_gain, tsteps, upper_bound=False)
            upper = test_rate_bounds(pacing_gain, tsteps)
            print(pacing_gain, tsteps, lower, upper)
            min_rate = min(LINK_RATE, LINK_RATE * pacing_gain)
            den_limit = 32
            if(min_rate == LINK_RATE):
                den_limit = tsteps
            record = {
                'sending_rate': pacing_gain,
                'tsteps': tsteps,
                # 'lower': fractions.Fraction(lower/min_rate).limit_denominator(den_limit),
                # 'upper': fractions.Fraction(upper/min_rate).limit_denominator(den_limit)
                # 'lower': round(lower/min_rate, 6),
                # 'upper': round(upper/min_rate, 6),
                'lower_ack_rate': lower,
                'upper_ack_rate': upper
            }
            records.append(record)
            all_records.append(record)
        df = pd.DataFrame(records).astype(float)
        print(df)

    df = pd.DataFrame(all_records).astype(float)
    print(df)
    return df


def vary_inflight():
    all_records = []
    # [0.1, 0.25, fractions.Fraction(1, 3), 0.5, fractions.Fraction(2, 3), 0.75, 1, 2]
    sweep = sorted(list(set([x/10 for x in range(1, 21, 1)] + [x/20 for x in range(1, 21, 1)])))
    for cwnd_gain in sweep:
        records = []
        n_tsteps = 3
        for tsteps in range(n_tsteps, n_tsteps+1):
            lower = test_rate_bounds(
                100, tsteps, False, cwnd_gain)
            upper = test_rate_bounds(100, tsteps, cwnd_gain=cwnd_gain)
            print(cwnd_gain, tsteps, lower, upper)
            record = {
                'inflight': cwnd_gain,
                'tsteps': tsteps,
                'lower_ack_rate': lower,
                'upper_ack_rate': upper
            }
            records.append(record)
            all_records.append(record)
        df = pd.DataFrame(records).astype(float)
        print(df)

    df = pd.DataFrame(all_records).astype(float)
    print(df)
    return df


def plot_rate_bounds(df, vary='inflight'):

    # Assumes df has one a single tstep for all rows.
    tsteps = df['tsteps'].iloc[0]
    xx = df[vary]
    ylower = df['lower']
    yupper = df['upper']

    fig, ax = plt.subplots()
    ax.fill_between(xx, ylower, yupper, alpha=0.5)
    ax.set_xlabel(vary)
    ax.set_ylabel(f'Delivery rate (over {tsteps} Rm)')
    if(vary == "inflight"):
        ax.set_xlabel("Inflight (max BDP)")
        ax.axvline(x=0.5, label='min BDP', color='black')
        ax.axvline(x=1, label='max BDP', color='black')
        ax.axhline(y=LINK_RATE, label='max BDP', color='black')

    fig.savefig(f"tmp/{vary}-{tsteps}.pdf")


if (__name__ == "__main__"):

    # test_rate_bounds(0.1, 4)
    # import sys
    # sys.exit(0)

    # # When rate is higher
    # records = []
    # n_tsteps = 5
    # for tsteps in range(1, n_tsteps+1):
    #     lower = test_rate_bounds(2, tsteps, upper_bound=False)
    #     upper = test_rate_bounds(2, tsteps)
    #     print(tsteps, lower, upper)
    #     record = {
    #         'tsteps': tsteps,
    #         'lower': fractions.Fraction(lower/LINK_RATE).limit_denominator(tsteps),
    #         'upper': fractions.Fraction(upper/LINK_RATE).limit_denominator(tsteps)
    #     }
    #     records.append(record)
    # df = pd.DataFrame(records)
    # print(df)

    df = vary_inflight()
    # plot_rate_bounds(df)
    # df = vary_sending_rate()
    # plot_rate_bounds(df)


"""
Findings from this excercise:
C: link rate
r: avg recv rate over n time steps.
n: time steps

C >= r * n/(n+1) always, if we additionally know that sending rate is
higher than C then, C <= r * n/(n-1).
"""
