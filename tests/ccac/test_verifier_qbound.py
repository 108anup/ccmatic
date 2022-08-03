import numpy as np
import z3
from ccac.model import loss_detected
from ccmatic.verifier import (desired_high_util_low_delay, get_cex_df,
                              setup_ccac_definitions, setup_ccac_environment)
from pyz3_utils.my_solver import MySolver

from ccac.config import ModelConfig
from ccac.variables import VariableNames, Variables

lag = 1
history = 1
use_loss = False
deterministic_loss = False
util_frac = 0.5
# n_losses = 1
ideal_max_queue = 8

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
if(deterministic_loss):
    c.deterministic_loss = True
c.loss_oracle = True
# No loss
if(use_loss):
    c.buf_min = c.C * (c.R + c.D)
    c.buf_max = c.buf_min
else:
    c.buf_max = None
    c.buf_min = None

ccac_domain = z3.And(*s.assertion_list)
sd = setup_ccac_definitions(c, v)
se = setup_ccac_environment(c, v)
ccac_definitions = z3.And(*sd.assertion_list)
environment = z3.And(*se.assertion_list)
exceed_queue_f = [[z3.Bool(f"Def__exceed_queue_{n},{t}") for t in range(c.T)]
                  for n in range(c.N)]  # definition variable
last_decrease_f = [[z3.Real(f"Def__last_decrease_{n},{t}") for t in range(c.T)]
                   for n in range(c.N)]  # definition variable

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
# loss_rate = n_losses / ((c.T-1) - first)
delay_bound = ideal_max_queue * c.C * (c.R + c.D)

(desired, high_util, low_delay, ramp_up, ramp_down) = \
    desired_high_util_low_delay(c, v, first, util_frac, delay_bound)
assert isinstance(desired, z3.ExprRef)

vn = VariableNames(v)

qsize_thresh = 5
definition_constrs = []

assert first >= 1
# assert history > lag
assert lag == c.R
assert c.R == 1

definition_constrs.append(last_decrease_f[0][0] == v.A_f[0][0] - v.L_f[0][0])  # v.S_f[0][0])
for t in range(1, first):
    definition_constrs.append(
        z3.Implies(v.c_f[0][t] < v.c_f[0][t-1],
                   last_decrease_f[0][t] == v.A_f[0][t] - v.L_f[0][t]))
    definition_constrs.append(
        z3.Implies(v.c_f[0][t] >= v.c_f[0][t-1],
                   last_decrease_f[0][t] == last_decrease_f[0][t-1]))
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

for t in range(first, c.T):
    loss_detected = exceed_queue_f[0][t]

    # assert first >= lag + 1
    this_decrease = z3.And(loss_detected, v.S[t-lag] > last_decrease_f[0][t-1])

    definition_constrs.append(z3.Implies(
        this_decrease, last_decrease_f[0][t] == v.A_f[0][t] - v.L_f[0][t]))
    definition_constrs.append(z3.Implies(
        z3.Not(this_decrease), last_decrease_f[0][t] == last_decrease_f[0][t-1]))

    rhs_loss = v.c_f[0][t-lag] / 2
    # rhs_loss = 0
    rhs_noloss = v.c_f[0][t-lag] + 1
    rhs = z3.If(this_decrease, rhs_loss, rhs_noloss)
    assert isinstance(rhs, z3.ArithRef)
    definition_constrs.append(v.c_f[0][t] == z3.If(rhs >= 0.01, rhs, 0.01))

    # definition_constrs.append(v.c_f[0][t] == 4096)
    # definition_constrs.append(v.c_f[0][t] == 0.01)


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    qdelay = []
    for t in range(c.T):
        assert bool(counter_example.eval(v.qbound[t][0]))
        added = False
        for dt in range(1, c.T):
            if(not bool(counter_example.eval(v.qbound[t][dt]))):
                qdelay.append(dt-1)
                added = True
                break
        if(not added):
            qdelay.append(c.T)
    assert len(qdelay) == c.T
    df["qdelay"] = np.array(qdelay).astype(float)
    df["last_decrease_f"] = [counter_example.eval(x).as_fraction()
                             for x in last_decrease_f[0]]
    # for t in range(c.T):
    #     print(t)
    #     print(
    #         counter_example.eval(
    #             z3.And(exceed_queue_f[0][t], v.S[t-1-lag] >= last_decrease_f[0][t-1]))
    #     )
    # import ipdb; ipdb.set_trace()
    df["this_decrease"] = [-1] + [
        bool(counter_example.eval(
            z3.And(exceed_queue_f[0][t], v.S[t-1-lag] >= last_decrease_f[0][t-1])))
        for t in range(1, c.T)]
    df["exceed_queue_f"] = [-1] + [bool(counter_example.eval(x)) for x in exceed_queue_f[0][1:]]
    df["qbound_thresh_f"] = [-1] + [bool(counter_example.eval(v.qbound[t][qsize_thresh])) for t in range(1, c.T)]

    ret = "{}".format(df.astype(float))
    conds = {
        "high_util": high_util,
        "low_delay": low_delay,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down,
    }
    cond_list = []
    for cond_name, cond in conds.items():
        cond_list.append(
            "{}={}".format(cond_name, counter_example.eval(cond)))
    ret += "\n{}.".format(", ".join(cond_list))
    return ret


print("Using c.buf_max={}.".format(c.buf_max))

verifier = MySolver()
verifier.warn_undeclared = False
verifier.add(ccac_domain)
verifier.add(ccac_definitions)
verifier.add(environment)
verifier.add(z3.And(*definition_constrs))
verifier.add(z3.Not(desired))

sat = verifier.check()
if(str(sat) == "sat"):
    model = verifier.model()
    print(get_counter_example_str(model))

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
