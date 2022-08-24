import logging

import pandas as pd
import z3
from ccmatic.cegis import CegisConfig
from ccmatic.verifier import (get_all_desired,
                              get_cex_df,
                              get_desired_property_string,
                              setup_cegis_basic)
from cegis.util import (Metric, optimize_multi_var)
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames

cc = CegisConfig()
cc.infinite_buffer = False
cc.dynamic_buffer = True
cc.buffer_size_multiplier = 0.1
cc.template_queue_bound = True
cc.template_mode_switching = True

cc.desired_util_f = z3.Real('desired_util_f')
cc.desired_queue_bound_multiplier = z3.Real('desired_queue_bound_multiplier')
cc.desired_loss_bound = z3.Real('desired_loss_bound')
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)

(desired, fefficient, bounded_queue, bounded_loss,
 ramp_up_cwnd, ramp_down_cwnd, ramp_down_q, ramp_down_bq,
 total_losses) = get_all_desired(cc, c, v)

vn = VariableNames(v)
first = cc.history  # First cwnd idx decided by synthesized cca
template_definitions = []
if(cc.template_queue_bound):
    template_definitions.append(v.qsize_thresh == 8)

if(cc.template_mode_switching):
    for t in range(1, c.T):
        # mode selection
        loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
        template_definitions.append(
            z3.If(loss_detected, v.mode_f[0][t],  # else
                  z3.If(v.exceed_queue_f[0][t], z3.Not(v.mode_f[0][t]),
                        v.mode_f[0][t] == v.mode_f[0][t-1])))
        # True means mode0 otherwise mode1
        # Check if we want this_decrease instead of exceed_queue_f

for t in range(first, c.T):
    loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    acked_bytes = v.S_f[0][t-c.R] - v.S_f[0][t-cc.history]
    if(cc.template_queue_bound):
        delay_detected = v.exceed_queue_f[0][t]
        this_decrease = z3.And(delay_detected,
                               v.S_f[0][t-c.R] > v.S_f[0][t-c.R-1],
                               v.S_f[0][t-1-c.R] >= v.last_decrease_f[0][t-1])

    if(cc.template_mode_switching):
        rhs_mode0_if = 0
        rhs_mode0_else = 1 * (acked_bytes)
        rhs_mode1_if = 1 * (acked_bytes)

        rhs = z3.If(v.mode_f[0][t], z3.If(
            loss_detected, rhs_mode0_if, rhs_mode0_else), rhs_mode1_if)
    elif(cc.template_queue_bound):
        rhs_loss = 0
        rhs_delay = 1/2 * (acked_bytes)
        rhs_noloss = acked_bytes

        rhs = z3.If(loss_detected, rhs_loss, z3.If(
            delay_detected, rhs_delay, rhs_noloss))
    else:
        # AIMD
        # rhs_loss = 1/2 * v.c_f[0][t-c.R]
        # rhs_noloss = v.c_f[0][t-c.R] + 1

        rhs_loss = 0
        rhs_noloss = 1/2 * v.c_f[0][t-c.R] + 1/2 * (acked_bytes)

        rhs_loss = 1/2 * (acked_bytes)
        rhs_noloss = 3/2 * v.c_f[0][t-c.R]

        rhs_loss = 0
        rhs_noloss = acked_bytes

        rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)

    assert isinstance(rhs, z3.ArithRef)
    template_definitions.append(
        v.c_f[0][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                             rhs, cc.template_cca_lower_bound))

    # template_definitions.append(v.c_f[0][t] == 4096)
    # template_definitions.append(v.c_f[0][t] == 0.01)


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    desired_string = get_desired_property_string(
        cc, c, fefficient, bounded_queue, bounded_loss,
        ramp_up_cwnd, ramp_down_bq, ramp_down_q, ramp_down_cwnd,
        total_losses, counter_example)
    ret = "{}\n{}.".format(df, desired_string)
    return ret


optimization_list = [
    Metric(cc.desired_util_f, 0.33, 1, 0.001, True),
    Metric(cc.desired_queue_bound_multiplier, 1, 2, 0.001, False),
    Metric(cc.desired_loss_bound, 0, 3, 0.001, False),
]

verifier = MySolver()
verifier.warn_undeclared = False
verifier.add(ccac_domain)
verifier.add(ccac_definitions)
verifier.add(environment)
verifier.add(z3.And(*template_definitions))
verifier.add(z3.Not(desired))

verifier.push()
for metric in optimization_list:
    if(metric.maximize):
        verifier.add(metric.z3ExprRef == metric.lo)
    else:
        verifier.add(metric.z3ExprRef == metric.hi)

sat = verifier.check()
if(str(sat) == "sat"):
    model = verifier.model()
    print(get_counter_example_str(model))

# import ipdb; ipdb.set_trace()

# else:
#     # Unsat core
#     dummy = MySolver()
#     dummy.warn_undeclared = False
#     dummy.set(unsat_core=True)

#     assertion_list = verifier.assertion_list
#     for assertion in assertion_list:
#         for expr in unroll_assertions(assertion):
#             dummy.add(expr)
#     assert(str(dummy.check()) == "unsat")
#     unsat_core = dummy.unsat_core()
#     print(len(unsat_core))
#     import ipdb; ipdb.set_trace()

verifier.pop()

GlobalConfig().logging_levels['cegis'] = logging.INFO
logger = logging.getLogger('cegis')
GlobalConfig().default_logger_setup(logger)


ret = optimize_multi_var(verifier, optimization_list)
df = pd.DataFrame(ret)
sort_columns = [x.name() for x in optimization_list]
sort_order = [x.maximize for x in optimization_list]
df = df.sort_values(by=sort_columns, ascending=sort_order)
print(df)