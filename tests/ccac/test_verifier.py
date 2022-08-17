import logging
import pandas as pd
from collections import namedtuple
import math
import pprint
import queue
from typing import List, NamedTuple
import numpy as np
import z3
from ccac.config import ModelConfig
from ccmatic.common import flatten, get_val_list
from ccmatic.verifier import (desired_high_util_low_delay, desired_high_util_low_loss, desired_high_util_low_loss_low_delay, get_cex_df,
                              setup_ccac, setup_ccac_definitions,
                              setup_ccac_environment)
from cegis.util import Metric, optimize_multi_var, optimize_var, unroll_assertions
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames, Variables

lag = 1
history = 4
use_loss = True
deterministic_loss = True
util_frac_val = 0.33
n_losses_val = 3
dynamic_buffer = True
buf_size = 1
max_ideal_queue_val = 2

# Verifier
# Dummy variables used to create CCAC formulation only
# Config
c = ModelConfig.default()
c.compose = True
c.cca = "paced"
c.simplify = False
c.calculate_qdel = False
c.calculate_qbound = True
c.C = 100
c.T = 9

s = MySolver()
v = Variables(c, s)
vn = VariableNames(v)
if(deterministic_loss):
    c.deterministic_loss = True
c.loss_oracle = True
if(use_loss):
    if(not dynamic_buffer):
        assert buf_size is not None
        c.buf_max = buf_size * c.C * (c.R + c.D)
    else:
        c.buf_max = z3.Real('buf_size')
else:
    c.buf_max = None
c.buf_min = c.buf_max
ccac_domain = z3.And(*s.assertion_list)
sd = setup_ccac_definitions(c, v)
se = setup_ccac_environment(c, v)
ccac_definitions = z3.And(*sd.assertion_list)
environment_assertions = se.assertion_list
exceed_queue_f = [[z3.Bool(f"Def__exceed_queue_{n},{t}") for t in range(c.T)]
                  for n in range(c.N)]  # definition variable
last_decrease_f = [[z3.Real(f"Def__last_decrease_{n},{t}") for t in range(c.T)]
                   for n in range(c.N)]  # definition variable
mode_f = np.array([[z3.Bool(f"Def__mode_{n},{t}") for t in range(c.T)]
                   for n in range(c.N)])  # definition variable

if(dynamic_buffer):
    # Buffer should have atleast one packet
    environment_assertions.append(c.buf_min > v.alpha)

    # environment_assertions.append(
    #     z3.Or(c.buf_min == c.C * (c.R + c.D),
    #           c.buf_min == 0.1 * c.C * (c.R + c.D)))

environment = z3.And(*environment_assertions)

assert c.N == 1

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
util_frac = z3.Real('util_frac')
n_losses = z3.Real('n_losses')
max_ideal_queue = z3.Real('max_ideal_queue')

loss_rate = n_losses / ((c.T-1) - first)
delay_bound = max_ideal_queue * c.C * (c.R + c.D)


# (desired, high_util, low_loss, ramp_up, ramp_down, total_losses) = \
#     desired_high_util_low_loss(c, v, first, util_frac, loss_rate)
# desired = z3.And(z3.Or(high_util, ramp_up), z3.Or(low_loss, ramp_down))
# (desired, high_util, low_delay, ramp_up, ramp_down) = \
#     desired_high_util_low_delay(c, v, first, util_frac, delay_bound)
(desired, high_util, low_loss, low_delay, ramp_up,
 ramp_down_cwnd, ramp_down_q, ramp_down_bq, total_losses) = \
    desired_high_util_low_loss_low_delay(
        c, v, first, util_frac, loss_rate, delay_bound)
assert isinstance(desired, z3.ExprRef)

lower_bound = 0.01
qsize_thresh = 2
definition_constrs = []

assert first >= 1
assert history >= lag
assert lag == c.R
assert c.R == 1

# definition_constrs.append(last_decrease_f[0][0] == v.A_f[0][0] - v.L_f[0][0])
definition_constrs.append(last_decrease_f[0][0] == v.S_f[0][0])
for t in range(1, first):
    definition_constrs.append(
        z3.Implies(v.c_f[0][t] < v.c_f[0][t-1],
                   last_decrease_f[0][t] == v.A_f[0][t] - v.L_f[0][t]))
    definition_constrs.append(
        z3.Implies(v.c_f[0][t] >= v.c_f[0][t-1],
                   last_decrease_f[0][t] == last_decrease_f[0][t-1]))
    # Const last_decrease in history
    # definition_constrs.append(
    #     last_decrease_f[0][t] == v.S_f[0][t])

for t in range(lag, c.T):
    for dt in range(c.T):
        definition_constrs.append(
            z3.Implies(z3.And(dt == qsize_thresh, v.qbound[t-lag][dt]),
                       exceed_queue_f[0][t]))
        definition_constrs.append(
            z3.Implies(z3.And(dt == qsize_thresh, z3.Not(v.qbound[t-lag][dt])),
                       z3.Not(exceed_queue_f[0][t])))


# for t in range(c.T):
#     definition_constrs.append(mode_f[0][t])

for t in range(1, c.T):
    # mode selection
    loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    definition_constrs.append(
        z3.If(loss_detected, mode_f[0][t],  # else
              z3.If(exceed_queue_f[0][t], z3.Not(mode_f[0][t]),
              mode_f[0][t] == mode_f[0][t-1])))
    # True means mode0 otherwise mode1
    # Check if we want this_decrease instead of exceed_queue_f

for t in range(first, c.T):
    delay_detected = exceed_queue_f[0][t]
    # When did decrease happen
    # TODO: see if we want to replace the last statement
    #  with (S_f[n][t-lag] > last_decrease[n][t-1])
    this_decrease = z3.And(delay_detected,
                           v.S_f[0][t-lag] > v.S_f[0][t-lag-1],
                           v.S_f[0][t-1-lag] >= last_decrease_f[0][t-1])

    definition_constrs.append(z3.Implies(
        this_decrease,
        last_decrease_f[0][t] == v.A_f[0][t] - v.L_f[0][t]))
    definition_constrs.append(z3.Implies(
        z3.Not(this_decrease),
        last_decrease_f[0][t] == last_decrease_f[0][t-1]))

    loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    # rhs_loss = v.c_f[0][t-lag] / 2

    # RoCC, AIMD hybrid
    rhs_loss = 0.5 * (v.S[t-1] - v.S[t-4])
    rhs_noloss = 2 * v.c_f[0][t-lag]

    # RoCC
    rhs_loss = (v.S[t-1] - v.S[t-4])
    rhs_noloss = (v.S[t-1] - v.S[t-4])

    # RoCC, AIMD hybrid
    rhs_loss = 0.5 * v.c_f[0][t-lag]
    rhs_loss = 0
    rhs_noloss = (v.S[t-1] - v.S[t-4])

    # mode switch between MIMD and RoCC
    rhs_mode0_if = 0.5 * v.c_f[0][t-lag] - 1/2
    rhs_mode0_else = 1.5 * v.c_f[0][t-lag]
    rhs_mode1_if = (v.S[t-1] - v.S[t-4])

    # RoCC + AIMD, no mode switching needed
    rhs_mode0_if = 0.5 * v.c_f[0][t-lag]
    rhs_mode0_else = (v.S[t-1] - v.S[t-4])
    rhs_mode1_if = (v.S[t-1] - v.S[t-4])

    # ramp down bq, aggressive
    rhs_mode0_if = 0
    rhs_mode0_else = 0.5 * v.c_f[0][t-lag] + 2 * (v.S[t-1] - v.S[t-4])
    rhs_mode1_if = 2 * (v.S[t-1] - v.S[t-4])

    # Multiple of RoCC on no loss and 0 on loss.
    rhs_mode0_if = 0
    rhs_mode0_else = 1.5 * (v.S[t-1] - v.S[t-4])
    rhs_mode1_if = 1.5 * (v.S[t-1] - v.S[t-4])

    rhs_mode0_if = 0
    rhs_mode0_else = 2 * (v.S[t-1] - v.S[t-4])
    rhs_mode1_if = 2 * (v.S[t-1] - v.S[t-4])

    rhs_mode0_if = 0
    rhs_mode0_else = 1 * (v.S[t-1] - v.S[t-4])
    rhs_mode1_if = 1 * (v.S[t-1] - v.S[t-4])


    rhs = z3.If(mode_f[0][t], z3.If(
        loss_detected, rhs_mode0_if, rhs_mode0_else), rhs_mode1_if)
    assert isinstance(rhs, z3.ArithRef)
    definition_constrs.append(
        v.c_f[0][t] == z3.If(rhs >= lower_bound, rhs, lower_bound))

    # definition_constrs.append(v.c_f[0][t] == 4096)
    # definition_constrs.append(v.c_f[0][t] == 0.01)


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
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
    df["qdelay"] = np.array(qdelay).astype(float)
    df["last_decrease_f"] = np.array(
        [counter_example.eval(x).as_fraction()
         for x in last_decrease_f[0]]).astype(float)
    df["this_decrease"] = [-1] + get_val_list(counter_example, [
        counter_example.eval(z3.And(
            exceed_queue_f[0][t],
            v.S_f[0][t-lag] > v.S_f[0][t-lag-1],
            v.S_f[0][t-1-lag] >= last_decrease_f[0][t-1]
        ))
        for t in range(1, c.T)])
    df["exceed_queue_f"] = [-1] + \
        get_val_list(counter_example, exceed_queue_f[0][1:])
    df["mode_f"] = get_val_list(counter_example, mode_f[0])

    ret = "{}".format(df)
    conds = {
        "high_util": high_util,
        "low_loss": low_loss,
        "low_delay": low_delay,
        "ramp_up": ramp_up,
        "ramp_down_cwnd": ramp_down_cwnd,
        "ramp_down_q": ramp_down_q,
        "ramp_down_bq": ramp_down_bq,
        "total_losses": total_losses
    }
    if(dynamic_buffer):
        conds["buffer"] = c.buf_min
        conds["alpha"] = v.alpha
    cond_list = []
    for cond_name, cond in conds.items():
        cond_list.append(
            "{}={}".format(cond_name, counter_example.eval(cond)))
    ret += "\n{}".format(", ".join(cond_list))
    # ret += " out of {} tsteps.".format(((c.T-1) - first))

    qbound_vals = []
    for qbound_list in v.qbound:
        qbound_val_list = get_val_list(counter_example, qbound_list)
        qbound_vals.append(qbound_val_list)
    ret += "\n{}".format(np.array(qbound_vals))

    return ret


print("Using c.buf_max={}.".format(c.buf_max))
print("Allowed losses:", loss_rate * ((c.T-1) - first))

verifier = MySolver()
verifier.warn_undeclared = False
verifier.add(ccac_domain)
verifier.add(ccac_definitions)
verifier.add(environment)
verifier.add(z3.And(*definition_constrs))
verifier.add(z3.Not(desired))

# How do the good traces look like:
# verifier.add(desired)
# verifier.add(v.L[c.T-2] > v.L[first])

verifier.push()
verifier.add(util_frac == util_frac_val)
verifier.add(max_ideal_queue == max_ideal_queue_val)
verifier.add(n_losses == n_losses_val)
# verifier.add(z3.And(high_util, v.L[-3] > v.L[first]))

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

optimization_list = [
    Metric(util_frac, 0.1, 1, 0.001, True),
    Metric(max_ideal_queue, 1, 16, 0.001, False),
    Metric(n_losses, 0, c.T-1-first, 0.001, False),
]

ret = optimize_multi_var(verifier, optimization_list)
df = pd.DataFrame(ret)
sort_columns = [x.name() for x in optimization_list]
sort_order = [x.maximize for x in optimization_list]
df = df.sort_values(by=sort_columns, ascending=sort_order)
print(df)

# for i, (variable, lo, hi, eps, maximize, val) in enumerate(optimization_list):
#     verifier.push()
#     for j in range(len(optimization_list)):
#         if(i != j):
#             print("Setting {} as {}".format(
#                 optimization_list[j][0].decl().name(), optimization_list[j][-1]))
#             verifier.add(optimization_list[j][0] == optimization_list[j][-1])
#     optimal_bounds = optimize_var(verifier, variable, lo, hi, eps, maximize)
#     # print(optimal_bounds)
#     if(maximize):
#         optimal_value = math.floor(optimal_bounds[0]/eps) * eps
#     else:
#         optimal_value = math.ceil(optimal_bounds[-1]/eps) * eps
#     print(optimal_value)
#     verifier.pop()
