import numpy as np
import z3
from ccmatic.common import get_val_list
from ccmatic.verifier import (desired_high_util_low_delay,
                              desired_high_util_low_loss, get_cex_df,
                              setup_ccac_definitions, setup_ccac_environment)
from pyz3_utils.my_solver import MySolver

from ccac.config import ModelConfig
from ccac.variables import VariableNames, Variables

lag = 1
history = 4
use_loss = False
deterministic_loss = False
util_frac = 0.5
n_losses = 1
ideal_max_queue = 8

# Verifier
# Dummy variables used to create CCAC formulation only
c = ModelConfig.default()
c.compose = True
c.cca = "paced"
c.simplify = False
c.calculate_qdel = True
c.C = 100
c.T = 9
c.N = 2

s = MySolver()
v = Variables(c, s)
vn = VariableNames(v)
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

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
loss_rate = n_losses / ((c.T-1) - first)
delay_bound = ideal_max_queue * c.C * (c.R + c.D)

if(use_loss):
    (desired, high_util, low_loss, ramp_up, ramp_down, total_losses) = \
        desired_high_util_low_loss(c, v, first, util_frac, loss_rate)
    desired = z3.And(z3.Or(high_util, ramp_up), z3.Or(low_loss, ramp_down))
else:
    (desired, high_util, low_delay, ramp_up, ramp_down) = \
        desired_high_util_low_delay(c, v, first, util_frac, delay_bound)
assert isinstance(desired, z3.ExprRef)

definition_constrs = []
for n in range(c.N):
    for t in range(first, c.T):
        cond = True
        rhs_loss = v.S_f[n][t-1] - v.S_f[n][t-4] + 1/2
        # rhs_loss = v.c_f[n][t-1] / 2
        rhs_noloss = v.S_f[n][t-1] - v.S_f[n][t-4] + 1/2
        # rhs_noloss = v.c_f[n][t-lag] + 1
        rhs = z3.If(cond, rhs_loss, rhs_noloss)
        assert isinstance(rhs, z3.ArithRef)
        definition_constrs.append(v.c_f[n][t] == z3.If(rhs >= 0.01, rhs, 0.01))

        # definition_constrs.append(v.c_f[n][t] == c.C/c.N)
        # definition_constrs.append(v.c_f[n][t] == 4096)
        # definition_constrs.append(v.c_f[n][t] == 0.01)


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    qdelay = []
    for t in range(c.T):
        added = False
        for dt in range(c.T):
            if(bool(counter_example.eval(v.qdel[t][dt]))):
                qdelay.append(dt)
                added = True
                break
        if(not added):
            qdelay.append(c.T)
    assert len(qdelay) == c.T
    df["qdelay"] = np.array(qdelay).astype(float)

    tot_cwnd = [
        counter_example.eval(z3.Sum(*v.c_f[:, t])).as_fraction()
        for t in range(c.T)]
    df['tot_cwnd_t'] = tot_cwnd

    single_rocc_cwnd = ([-1 for t in range(first)] +
                        [counter_example.eval(v.S[t-1] - v.S[t-4]).as_fraction()
                         for t in range(first, c.T)])
    single_rocc_arrival = [counter_example.eval(
        v.A[t]).as_fraction() for t in range(first)]
    for t in range(first, c.T):
        single_rocc_arrival.append(
            max(single_rocc_arrival[t-1],
                counter_example.eval(
                    v.S[t-c.R] + (v.S[t-1] - v.S[t-4])).as_fraction())
        )
    df['single_rocc_arrival_t'] = single_rocc_arrival
    df['single_rocc_cwnd_t'] = single_rocc_cwnd

    df['diff_service_1,t'] = [-1] + [counter_example.eval(v.S_f[1][t] - v.S_f[1][t-1]).as_fraction() for t in range(1, c.T)]
    df['diff_service_0,t'] = [-1] + [counter_example.eval(v.S_f[0][t] - v.S_f[0][t-1]).as_fraction() for t in range(1, c.T)]
    df['diff_tot_service_t'] = [-1] + [counter_example.eval(v.S[t] - v.S[t-1]).as_fraction() for t in range(1, c.T)]

    df['diff_arrival_1,t'] = [-1] + [counter_example.eval(v.A_f[1][t] - v.A_f[1][t-1]).as_fraction() for t in range(1, c.T)]
    df['diff_arrival_0,t'] = [-1] + [counter_example.eval(v.A_f[0][t] - v.A_f[0][t-1]).as_fraction() for t in range(1, c.T)]
    df['diff_tot_arrival_t'] = [-1] + [counter_example.eval(v.A[t] - v.A[t-1]).as_fraction() for t in range(1, c.T)]

    df['queue_0,t'] = [counter_example.eval(v.A_f[0][t] - v.L_f[0][t] - v.S_f[0][t]).as_fraction() for t in range(c.T)]
    df['queue_1,t'] = [counter_example.eval(v.A_f[1][t] - v.L_f[1][t] - v.S_f[1][t]).as_fraction() for t in range(c.T)]

    df['cwnd_rocc,t'] = ([-1 for _ in range(first)]
                         + [max(0.01, counter_example.eval(v.S[t-1] - v.S[t-4]).as_fraction()) for t in range(first, c.T)])
    df['arrival_rocc,t'] = (
        [counter_example.eval(v.A[t]).as_fraction() for t in range(first)] + [max(
            counter_example.eval(v.A[t-1]).as_fraction(),
            counter_example.eval(v.S[t-c.R] + df['cwnd_rocc,t'][t]).as_fraction()) for t in range(first, c.T)])

    ret = "{}".format(df.astype(float))
    conds = {
        "high_util": high_util,
        "low_delay": low_delay,
        # "low_loss": low_loss,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down,
        # "total_losses": total_losses,
        # "measured_loss_rate": total_losses/((c.T-1) - first)
    }
    cond_list = []
    for cond_name, cond in conds.items():
        cond_list.append(
            "{}={}".format(cond_name, counter_example.eval(cond)))
    ret += "\n{}.".format(", ".join(cond_list))

    # qbound_vals = []
    # for qbound_list in v.qbound:
    #     qbound_val_list = get_val_list(counter_example, qbound_list)
    #     qbound_vals.append(qbound_val_list)
    # ret += "\n{}".format(np.array(qbound_vals))

    qdel_vals = []
    for qdel_list in v.qdel:
        qdel_val_list = get_val_list(counter_example, qdel_list)
        qdel_vals.append(qdel_val_list)
    ret += "\n{}".format(np.array(qdel_vals).astype(int))
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
# verifier.add(z3.And(high_util, v.L[-3] > v.L[first]))

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
