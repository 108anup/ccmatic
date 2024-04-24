import z3
from ccmatic import CCmatic
from ccmatic.cegis import CegisConfig, VerifierType
from ccmatic.common import try_except
from ccmatic.verifier.cbr_delay import CBRDelayLink
from cegis.util import z3_max
from pyz3_utils.my_solver import MySolver

cc = CegisConfig()
cc.name = ""
cc.synth_ss = False
cc.infinite_buffer = False
cc.dynamic_buffer = True
cc.buffer_size_multiplier = 1/4

cc.app_limited = False
cc.app_fixed_avg_rate = True
cc.app_rate = None  # 0.5 * cc.C
cc.app_burst_factor = 1

cc.template_qdel = True
cc.template_queue_bound = False
cc.template_fi_reset = False
cc.template_beliefs = True
cc.template_beliefs_use_buffer = False
cc.template_beliefs_use_max_buffer = False
cc.N = 2
cc.T = 7
cc.history = cc.R
cc.cca = "none"

cc.rate_or_window = 'rate'
cc.use_belief_invariant = True
cc.fix_stale__min_c = False
cc.fix_stale__max_c = False
cc.min_maxc_minc_gap_mult = (10+1)/(10-1)
cc.min_maxc_minc_gap_mult = 1
cc.maxc_minc_change_mult = 1.1

# These don't matter here.
cc.desired_no_large_loss = True
cc.desired_util_f = 0.1
cc.desired_queue_bound_multiplier = 4
cc.desired_queue_bound_alpha = 3
cc.desired_loss_count_bound = (cc.T-1)/2 + 1
cc.desired_large_loss_count_bound = 0   # if NO_LARGE_LOSS else (cc.T-1)/2
# We don't expect losses in steady state. Losses only happen when beliefs
# are changing.
cc.desired_loss_amount_bound_multiplier = (cc.T-1)/2 + 1  # 0
cc.desired_loss_amount_bound_alpha = cc.T-1  # (cc.T-1)/2 - 1  # 4

cc.opt_cegis = True
cc.opt_ve = False
cc.opt_pdt = False
cc.opt_wce = False
cc.feasible_response = False

cc.verifier_type = VerifierType.cbrdelay

cc.send_min_alpha = True

link = CCmatic(cc)
try_except(link.setup_config_vars)
c, _, v = link.c, link.s, link.v
link.setup_cegis_loop(
    True,
    [], [], None)

assert isinstance(c, CBRDelayLink.LinkModelConfig)
assert isinstance(v, CBRDelayLink.LinkVariables)

verifier = MySolver()
verifier.warn_undeclared = False

# CCA solution
assert c.R == 1
for n in range(c.N):
    for t in range(cc.history, c.T):
        rate_choice = z3.If(v.bq_belief1[n][t-c.R] > v.alpha,
                            v.alpha,
                            3 * v.min_c_lambda[n][t-c.R] + v.alpha)
        verifier.add(v.r_f[n][t] == z3_max(v.alpha, rate_choice))

        # Remove any cwnd limits
        verifier.add(v.c_f[n][t] == v.A_f[n][t-1] - v.S_f[n][t-1] + v.r_f[n][t] * 1000)

verifier.add(link.environment)
verifier.add(link.definitions)

# Objectives
# desired = False
d = link.d
beliefs_shrink = z3.Or(
    z3.And(v.min_c_lambda[0][0] < v.min_c_lambda[0][c.T - 1], v.min_c_lambda[1][0] <= v.min_c_lambda[1][c.T - 1]),
    z3.And(v.min_c_lambda[1][0] < v.min_c_lambda[1][c.T - 1], v.min_c_lambda[0][0] <= v.min_c_lambda[0][c.T - 1]),
)
agg_util_inv = z3.Or(d.fefficient, d.ramp_up_cwnd, d.ramp_up_bq, d.ramp_up_queue)
gap_decrease = z3.Implies(
    v.min_c_lambda[0][0] > 10 * v.min_c_lambda[1][0],
    z3.And(
        # v.min_c_lambda[0][c.T - 1] < v.min_c_lambda[0][0],
        v.min_c_lambda[1][c.T - 1] >= v.min_c_lambda[1][0] + v.alpha/10,
    ),
)
desired = z3.And(
    z3.Or(beliefs_shrink, agg_util_inv)
)  # This passes implying that CCA ensures freedom from arbitrary underutilization
# desired = z3.And(z3.Or(beliefs_shrink, agg_util_inv), z3.Or(gap_decrease))
# desired = False

verifier.add(z3.Not(desired))
sat = verifier.check()
print(sat)
if(str(sat) == "sat"):
    model = verifier.model()
    print(link.get_counter_example_str(model, link.verifier_vars))

    # def print_desired_vals():
    print("Beliefs shrink: ", model.eval(beliefs_shrink))
    print("Agg util invariant: ", model.eval(agg_util_inv))
    print("Gap decrease: ", model.eval(gap_decrease))

import ipdb; ipdb.set_trace()

"""
# LinkModelConfig = CBRDelayLink.LinkModelConfig

# pre = ""

# c = LinkModelConfig.default()
# c.N = 2
# c.D = 1
# c.R = c.D
# c.T = 10
# c.C = z3.Real(f"{pre}C")
# c.buf_max = z3.Real(f'{pre}buf_size')
# c.buf_min = c.buf_max
# # c.dupacks = None
# # c.cca = "dont_use"
# c.compose = None
# c.alpha = 1
# # c.pacing = None


# c.compose = True
# c.simplify = False
# c.T = 10
# c.R = 1
# c.D = c.R

# c.loss_oracle = True
# c.deterministic_loss = True


# c.N = 2

# c.calculate_qdel = True
# c.feasible_response = False
# c.mode_switch = False
"""
