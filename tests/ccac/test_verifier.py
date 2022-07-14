import z3
from ccmatic.common import flatten
from ccmatic.verifier import (desired_high_util_low_loss, get_cex_df,
                              setup_ccac, setup_ccac_definitions,
                              setup_ccac_environment)
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames

lag = 1
history = 4
first = history

# Verifier
# Dummy variables used to create CCAC formulation only
c, s, v = setup_ccac()
vn = VariableNames(v)
c.buf_max = c.C * (c.R + c.D)
c.buf_min = c.buf_max
ccac_domain = z3.And(*s.assertion_list)
sd = setup_ccac_definitions(c, v, use_loss_oracle=True)
se = setup_ccac_environment(c, v)
ccac_definitions = z3.And(*sd.assertion_list)
environment = z3.And(*se.assertion_list)

conditional_vvars = []
if(not c.compose):
    conditional_vvars.append(v.epsilon)
conditional_dvars = []
if(c.calculate_qdel):
    conditional_dvars.append(v.qdel)

assert c.N == 1
verifier_vars = flatten(
    [v.A_f[0][:history], v.c_f[0][:history], v.S_f, v.W,
     v.L_f, v.Ld_f, v.dupacks, v.alpha, conditional_vvars, v.C0])
definition_vars = flatten(
    [v.A_f[0][history:], v.A, v.c_f[0][history:],
     v.r_f, v.S, v.L, v.timeout_f, conditional_dvars])

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
util_frac = 0.505
loss_rate = 1 / ((c.T-1) - first)

(desired, high_util, low_loss, ramp_up, ramp_down, measured_loss_rate) = \
    desired_high_util_low_loss(c, v, first, util_frac, loss_rate)
desired = z3.And(z3.Or(high_util, ramp_up), z3.Or(low_loss, ramp_down))
assert isinstance(desired, z3.ExprRef)

definition_constrs = []
for t in range(first, c.T):
    cond = v.Ld_f[0][t-c.R] > v.Ld_f[0][t-c.R-1]
    # rhs_loss = v.c_f[0][t-lag] / 2
    rhs_loss = 0
    rhs_noloss = v.c_f[0][t-lag] + 1
    rhs = z3.If(cond, rhs_loss, rhs_noloss)
    assert isinstance(rhs, z3.ArithRef)
    definition_constrs.append(v.c_f[0][t] == z3.If(rhs >= 0.01, rhs, 0.01))

    # definition_constrs.append(v.c_f[0][t] == 4096)
    # definition_constrs.append(v.c_f[0][t] == 0.01)


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    ret = "{}".format(df.astype(float))
    conds = {
        "high_util": high_util,
        "low_loss": low_loss,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down,
        "measured_loss_rate": measured_loss_rate
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
