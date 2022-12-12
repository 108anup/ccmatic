import sys
import pandas as pd
import logging
import z3
from ccac.variables import VariableNames
from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig
from cegis import get_unsat_core
from cegis.util import Metric, fix_metrics, get_raw_value, optimize_multi_var, optimize_var, unroll_assertions
from ccmatic.verifier import get_cex_df, setup_cegis_basic
from ccmatic.verifier.assumptions import (get_cca_definition, get_cca_vvars,
                                          get_periodic_constraints_ccac)
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver


def test_sending_rate_higher(
        sending_rate_multiplier=2, timesteps=1, upper_bound=True):
    cc = CegisConfig()
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
    cc.cca = "paced"
    cc.feasible_response = False

    link = CCmatic(cc)
    link.setup_config_vars()
    c, _, v, = link.c, link.s, link.v
    link.setup_cegis_loop(True, [], [], lambda x, y: "")

    sending_rate_constraint = z3.And(*[
        v.r_f[n][t] == cc.C * sending_rate_multiplier
        for t in range(cc.T) for n in range(cc.N)])

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.ccac_domain)
    verifier.add(link.ccac_definitions)
    verifier.add(link.environment)
    verifier.add(sending_rate_constraint)

    assert cc.T-1-timesteps >= cc.history
    assert cc.R == cc.D
    rate_estimate = z3.Real("rate_estimate")
    verifier.add(rate_estimate == (v.S[cc.T-1] - v.S[cc.T-1-timesteps])/(cc.D * timesteps))
    default_lo = 0
    default_hi = 100 * cc.C * (cc.R + cc.D)

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
            default_lo, default_hi, 0.001, upper_bound)
    ]

    # GlobalConfig().logging_levels['cegis'] = logging.DEBUG
    # logger = logging.getLogger('cegis')
    # GlobalConfig().default_logger_setup(logger)

    metric = optimization_list[0]
    # Note reversed polarity of metric here as we want max value such that formula is sat.
    ret = optimize_var(verifier, metric.z3ExprRef, metric.lo, metric.hi, metric.eps, not metric.maximize)
    # ret = optimize_multi_var(verifier, optimization_list)
    # df = pd.DataFrame(ret)
    # sort_columns = [x.name() for x in optimization_list]
    # sort_order = [x.maximize for x in optimization_list]
    # df = df.sort_values(by=sort_columns, ascending=sort_order)
    print(ret)


if (__name__ == "__main__"):
    test_sending_rate_higher()
    test_sending_rate_higher(upper_bound=False)
    test_sending_rate_higher(timesteps=2)
    test_sending_rate_higher(timesteps=2, upper_bound=False)
