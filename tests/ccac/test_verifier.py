from gc import get_count
import z3
from ccac.variables import VariableNames
from ccmatic.common import flatten
from ccmatic.verifier import (get_cex_df, setup_ccac, setup_ccac_definitions,
                              setup_ccac_environment)
from pyz3_utils.my_solver import MySolver


lag = 1
history = 1
first = history

# Verifier
# Dummy variables used to create CCAC formulation only
c, s, v = setup_ccac()
vn = VariableNames(v)
c.buf_max = c.C * (c.R + c.D)
ccac_domain = z3.And(*s.assertion_list)
sd = setup_ccac_definitions(c, v)
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
util_frac = 0.5
delay_bound = 0.5 * c.C * (c.R + c.D)

cond_list = []
for t in range(first, c.T):
    cond_list.append(v.A[t] - v.L[t] - v.S[t] <= delay_bound)
# Queue seen by a new packet should not be more that delay_bound
low_delay = z3.And(*cond_list)
# Serviced should be at least util_frac that could have been serviced
high_util = v.S[-1] - v.S[first] >= util_frac * c.C * (c.T-1-first-c.D)
# If the cwnd0 is very low then CCA should increase cwnd
ramp_up = v.c_f[0][-1] > v.c_f[0][first]
# If the queue is large to begin with then, CCA should cause queue to decrease.
ramp_down = v.A[-1] - v.L[-1] - v.S[-1] < v.A[first] - v.L[first] - v.S[first]

desired = z3.And(
    z3.Or(high_util, ramp_up),
    z3.Or(low_delay, ramp_down)
)
assert isinstance(desired, z3.ExprRef)

definition_constrs = []
for t in range(first, c.T):
    cond = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    rhs_loss = v.c_f[0][t-lag] / 2
    rhs_noloss = v.c_f[0][t-lag] + 1
    rhs = z3.If(cond, rhs_loss, rhs_noloss)
    assert isinstance(rhs, z3.ArithRef)
    definition_constrs.append(v.c_f[0][t] == z3.If(rhs >= 0.01, rhs, 0.01))


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn)
    queue_t = []
    for t in range(c.T):
        queue_t.append(
            counter_example.eval(
                v.A[t] - v.L[t] - v.S[t]
            ).as_fraction())
    df["queue_t"] = queue_t
    ret = "{}".format(df.astype(float))
    conds = {
        "high_util": high_util,
        "low_delay": low_delay,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down
    }
    cond_list = []
    for cond_name, cond in conds.items():
        cond_list.append(
            "{}={}".format(cond_name, counter_example.eval(cond)))
    ret += "\n{}.".format(", ".join(cond_list))
    return ret


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
