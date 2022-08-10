import z3
from ccmatic.common import flatten
from ccmatic.verifier import (desired_high_util_low_delay, desired_high_util_low_loss, desired_high_util_low_loss_low_delay, get_cex_df,
                              setup_ccac, setup_ccac_definitions,
                              setup_ccac_environment)
from cegis.util import unroll_assertions
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames

lag = 1
history = 4
use_loss = True
deterministic_loss = True
util_frac = 0.5
n_losses = 2
dynamic_buffer = True
buf_size = 1
max_ideal_queue = 2

# Verifier
# Dummy variables used to create CCAC formulation only
c, s, v = setup_ccac()
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
loss_rate = n_losses / ((c.T-1) - first)
delay_bound = max_ideal_queue * c.C * (c.R + c.D)

# (desired, high_util, low_loss, ramp_up, ramp_down, total_losses) = \
#     desired_high_util_low_loss(c, v, first, util_frac, loss_rate)
# desired = z3.And(z3.Or(high_util, ramp_up), z3.Or(low_loss, ramp_down))
# (desired, high_util, low_delay, ramp_up, ramp_down) = \
#     desired_high_util_low_delay(c, v, first, util_frac, delay_bound)
(desired, high_util, low_loss, low_delay, ramp_up, ramp_down, total_losses) = \
    desired_high_util_low_loss_low_delay(
        c, v, first, util_frac, loss_rate, delay_bound)
assert isinstance(desired, z3.ExprRef)

definition_constrs = []
for t in range(first, c.T):
    cond = v.Ld_f[0][t] > v.Ld_f[0][t-1]
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
        "low_delay": low_delay,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down,
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
