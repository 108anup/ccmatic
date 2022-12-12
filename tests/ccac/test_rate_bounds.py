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


def test_rate_bounds(
        sending_rate_multiplier=2, tsteps=1, upper_bound=True):
    cc = CegisConfig()
    cc.C = LINK_RATE
    cc.T = 10
    cc.history = cc.R
    cc.infinite_buffer = False
    cc.dynamic_buffer = True
    cc.buffer_size_multiplier = 1
    cc.template_queue_bound = False
    cc.template_mode_switching = False
    cc.template_qdel = False

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
        v.r_f[n][t] == cc.C * sending_rate_multiplier
        for t in range(cc.T) for n in range(cc.N)])
    cwnd_constraint = z3.And(*[
        v.c_f[n][t] == 100 * cc.C * (c.R + c.D)
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
    verifier.add(v.A[0] - v.L[0] - v.S[0] <= cc.C * cc.R * sending_rate_multiplier)
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

    print(sending_rate_multiplier, tsteps, ret)
    if(upper_bound):
        return ret[0]
    else:
        return ret[2]


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

    all_records = []
    for sending_rate_multiplier in \
            [0.1, 0.25, fractions.Fraction(1, 3), 0.5, fractions.Fraction(2, 3), 0.75, 1, 2]:
        # When rate is higher
        records = []
        n_tsteps = 5
        for tsteps in range(1, n_tsteps+1):
            lower = test_rate_bounds(
                sending_rate_multiplier, tsteps, upper_bound=False)
            upper = test_rate_bounds(sending_rate_multiplier, tsteps)
            print(sending_rate_multiplier, tsteps, lower, upper)
            min_rate = min(LINK_RATE, LINK_RATE * sending_rate_multiplier)
            den_limit = 32
            if(min_rate == LINK_RATE):
                den_limit = tsteps
            record = {
                'sending_rate': sending_rate_multiplier,
                'tsteps': tsteps,
                # 'lower': fractions.Fraction(lower/min_rate).limit_denominator(den_limit),
                # 'upper': fractions.Fraction(upper/min_rate).limit_denominator(den_limit)
                'lower': round(lower/min_rate, 6),
                'upper': round(upper/min_rate, 6)
            }
            records.append(record)
            all_records.append(record)
        df = pd.DataFrame(records).astype(float)
        print(df)

    df = pd.DataFrame(all_records).astype(float)
    print(df)


"""
Findings from this excercise:
C: link rate
r: avg recv rate over n time steps.
n: time steps

C >= r * n/(n+1) always, if we additionally know that sending rate is
higher than C then, C <= r * n/(n-1).
"""
