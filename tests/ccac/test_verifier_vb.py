import logging

import pandas as pd
import z3
from ccmatic.cegis import CegisConfig
from ccmatic.verifier import (get_cex_df, get_desired_necessary,
                              setup_cegis_basic)
from ccmatic.verifier.ideal import IdealLink
from cegis.util import Metric, optimize_multi_var, write_solver
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames

cc = CegisConfig()
cc.infinite_buffer = False
cc.dynamic_buffer = True
cc.buffer_size_multiplier = 1
cc.template_queue_bound = False
cc.template_mode_switching = False

cc.history = 4
cc.D = 1
cc.C = 100

# cc.loss_alpha = True
# cc.ideal_link = True

cc.desired_util_f = z3.Real('desired_util_f')
cc.desired_queue_bound_multiplier = z3.Real('desired_queue_bound_multiplier')
cc.desired_queue_bound_alpha = z3.Real('desired_queue_bound_alpha')
cc.desired_loss_count_bound = z3.Real('desired_loss_count_bound')
cc.desired_loss_amount_bound_multiplier = z3.Real('desired_loss_amount_bound')
cc.desired_loss_amount_bound_alpha = z3.Real('desired_loss_amount_alpha')

if(cc.ideal_link):
    (c, s, v,
     ccac_domain, ccac_definitions, environment,
     verifier_vars, definition_vars) = IdealLink.setup_cegis_basic(cc)
else:
    cc.feasible_response = True
    (c, s, v,
     ccac_domain, ccac_definitions, environment,
     verifier_vars, definition_vars) = setup_cegis_basic(cc)

d = get_desired_necessary(cc, c, v)
desired = d.desired_in_ss
desired = d.desired_necessary

vn = VariableNames(v)
first = cc.history  # First cwnd idx decided by synthesized cca
template_definitions = []
if(cc.template_queue_bound):
    template_definitions.append(v.qsize_thresh == 2)

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

        rhs_loss = 1/2 * v.c_f[0][t-c.R]
        rhs_delay = (acked_bytes)
        rhs_noloss = 1/2 * acked_bytes + 1/2 * v.c_f[0][t-c.R]

        rhs = z3.If(loss_detected, rhs_loss, z3.If(
            delay_detected, rhs_delay, rhs_noloss))
    else:
        # rhs_loss = 1/2 * (acked_bytes)
        # rhs_noloss = 3/2 * v.c_f[0][t-c.R]

        # # Hybrid
        # rhs_loss = 0
        # rhs_noloss = acked_bytes

        # MIMD
        rhs_loss = 1/2 * v.c_f[0][t-c.R]
        rhs_noloss = 3/2 * v.c_f[0][t-c.R]

        # AIAD
        rhs_loss = v.c_f[0][t-c.R] - 1
        rhs_noloss = v.c_f[0][t-c.R] + 1

        # Hybrid half_acked_bytes
        rhs_loss = 1/2 * acked_bytes
        rhs_noloss = 1/2 * acked_bytes + 1/2 * v.c_f[0][t-c.R]

        # Hybrid to 0
        rhs_loss = 0
        rhs_noloss = 1/2 * acked_bytes + 1/2 * v.c_f[0][t-c.R]

        # # Hybrid MD no combination
        # rhs_loss = 1/2 * v.c_f[0][t-c.R]
        # rhs_noloss = acked_bytes

        # 0 jitter network
        rhs_loss = 1/2 * v.c_f[0][t-c.R]
        rhs_noloss = 1/2 * acked_bytes

        # AIMD
        rhs_loss = 1/2 * v.c_f[0][t-c.R]
        rhs_noloss = v.c_f[0][t-c.R] + 1

        # RoCC + MD
        rhs_loss = 1/2 * v.c_f[0][t-c.R] - v.alpha
        rhs_noloss = acked_bytes + v.alpha

        # Hybrid MD
        rhs_loss = 1/2 * v.c_f[0][t-c.R] - v.alpha
        rhs_noloss = 1/2 * acked_bytes + 1/2 * v.c_f[0][t-c.R] + v.alpha

        rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)

    assert isinstance(rhs, z3.ArithRef)
    # target_cwnd = z3.If(rhs >= cc.template_cca_lower_bound+v.alpha,
    #                     rhs, cc.template_cca_lower_bound+v.alpha)
    target_cwnd = rhs
    # next_cwnd = z3.If(v.c_f[0][t-1] < target_cwnd,
    #                   v.c_f[0][t-1] + v.alpha,
    #                   v.c_f[0][t-1] - v.alpha)
    next_cwnd = target_cwnd
    template_definitions.append(
        v.c_f[0][t] == z3.If(next_cwnd >= v.alpha, next_cwnd, v.alpha))
    # template_definitions.append(
    #     v.c_f[0][t] == z3.If(rhs >= cc.template_cca_lower_bound,
    #                          rhs, cc.template_cca_lower_bound))

    # template_definitions.append(v.c_f[0][t] == 4096)
    # template_definitions.append(v.c_f[0][t] == 0.01)


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    desired_string = d.to_string(cc, c, counter_example)
    ret = "{}\n{}, alpha={}.".format(df, desired_string,
                                     counter_example.eval(v.alpha))
    return ret


optimization_list = [
    Metric(cc.desired_util_f, 0.33, 1, 0.001, True),
    Metric(cc.desired_queue_bound_multiplier, 0, 4, 0.001, False),
    Metric(cc.desired_queue_bound_alpha, 0, 4, 0.001, False),
    Metric(cc.desired_loss_count_bound, 3, 3, 0.001, False),
    Metric(cc.desired_loss_amount_bound_multiplier, 0, 4, 0.001, False),
    Metric(cc.desired_loss_amount_bound_alpha, 0, 3, 0.001, False),
]

verifier = MySolver()
verifier.warn_undeclared = False
verifier.add(ccac_domain)
verifier.add(ccac_definitions)
verifier.add(environment)
verifier.add(z3.And(*template_definitions))
verifier.add(z3.Not(desired))
# verifier.add(desired)

# # Initial states
# verifier.add(v.c_f[0][first] <= 3 * c.C * (c.R + c.D))
# verifier.add(v.c_f[0][first] >= 1.5 * c.C * c.R)
# verifier.add(v.A[first] - v.L[first] - v.S[first] >= 0)
# verifier.add(v.A[first] - v.L[first] - v.S[first] <= 2 * c.C * (c.R + c.D))

# # Initial state H[loss=MD, noloss=combination]
# verifier.add(v.c_f[0][first] >= c.C * (c.R + c.D))
# verifier.add(v.c_f[0][first] <= (cc.history-1) * c.C * (c.R + c.D))
# verifier.add(v.A[first] - v.L[first] - v.S[first] >= 0)
# verifier.add(v.A[first-1] - v.L[first-1] - v.S[first-1] <= 1.5 * c.C * (c.R + c.D))

# verifier.add(v.c_f[0][first] == 3 * c.C * (c.R + c.D))
# verifier.add(v.A[first] - v.L[first] - v.S[first] > 1.5 * c.C * (c.R + c.D))

verifier.push()
for metric in optimization_list:
    if(metric.maximize):
        verifier.add(metric.z3ExprRef == metric.lo)
    else:
        verifier.add(metric.z3ExprRef == metric.hi)

write_solver(verifier, "tmp/test_verifier")

sat = verifier.check()
if(str(sat) == "sat"):
    model = verifier.model()
    print(get_counter_example_str(model))
    import ipdb; ipdb.set_trace()

else:
    # # Unsat core
    # dummy = MySolver()
    # dummy.warn_undeclared = False
    # dummy.set(unsat_core=True)

    # assertion_list = verifier.assertion_list
    # for assertion in assertion_list:
    #     for expr in unroll_assertions(assertion):
    #         dummy.add(expr)
    # assert(str(dummy.check()) == "unsat")
    # unsat_core = dummy.unsat_core()
    # print(len(unsat_core))
    # import ipdb; ipdb.set_trace()

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
