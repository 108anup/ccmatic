import logging

import pandas as pd
import z3
from ccmatic.cegis import CegisConfig
from ccmatic.verifier import (get_cex_df, get_desired_necessary,
                              setup_cegis_basic)
from cegis.util import Metric, optimize_multi_var
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames, Variables

cc = CegisConfig()
# cc.T = 15
cc.feasible_response = True

cc.infinite_buffer = False
cc.dynamic_buffer = False
cc.buffer_size_multiplier = 1
cc.template_queue_bound = False
cc.template_mode_switching = False

cc.desired_util_f = z3.Real('desired_util_f')
cc.desired_queue_bound_multiplier = z3.Real('desired_queue_bound_multiplier')
cc.desired_loss_count_bound = z3.Real('desired_loss_count_bound')
cc.desired_loss_amount_bound_multiplier = z3.Real('desired_loss_amount_bound')

assert(not cc.template_queue_bound)
assert(not cc.template_mode_switching)

(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)

# variables for other cca being compared to
alt_prefix = "alt"
(c_alt, s_alt, v_alt,
 ccac_domain_alt, ccac_definitions_alt, environment_alt,
 verifier_vars_alt, definition_vars_alt) = setup_cegis_basic(cc, alt_prefix)

d = get_desired_necessary(cc, c, v)
desired = d.desired_necessary

d_alt = get_desired_necessary(cc, c_alt, v_alt)
desired_alt = d_alt.desired_necessary

vn = VariableNames(v)
vn_alt = VariableNames(v_alt)
first = cc.history  # First cwnd idx decided by synthesized cca

same_decisions = []
for vvar in verifier_vars:
    v_alt_var = z3.Const(f"{alt_prefix}__{vvar.decl().name()}", vvar.sort())
    same_decisions.append(v_alt_var == vvar)

template_definitions = []
for t in range(first, c.T):

    # CCA main
    loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
    acked_bytes = v.S_f[0][t-c.R] - v.S_f[0][t-cc.history]

    rhs_loss = 1/2 * v.c_f[0][t-c.R]
    rhs_noloss = 1/2 * acked_bytes + 1/2 * v.c_f[0][t-c.R]

    rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
    assert isinstance(rhs, z3.ArithRef)

    template_definitions.append(
        v.c_f[0][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                             rhs, cc.template_cca_lower_bound))

    # CCA alt
    loss_detected = v_alt.Ld_f[0][t] > v_alt.Ld_f[0][t-1]
    acked_bytes = v_alt.S_f[0][t-c.R] - v_alt.S_f[0][t-cc.history]

    rhs_loss = 1/2 * v_alt.c_f[0][t-c.R]
    rhs_noloss = acked_bytes

    rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
    assert isinstance(rhs, z3.ArithRef)

    template_definitions.append(
        v_alt.c_f[0][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                                 rhs, cc.template_cca_lower_bound))


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    desired_string = d.to_string(cc, c, counter_example)
    ret = "{}\n{}.".format(df, desired_string)

    df_alt = get_cex_df(counter_example, v_alt, vn_alt, c_alt)
    desired_string_alt = d_alt.to_string(cc, c_alt, counter_example)
    ret += "\n"
    ret += "{}\n{}.".format(df_alt, desired_string_alt)

    return ret


verifier = MySolver()
verifier.warn_undeclared = False
definitions = z3.And(ccac_domain, ccac_definitions)
definitions_alt = z3.And(ccac_domain_alt, ccac_definitions_alt)
specification = z3.Implies(environment, desired)
specification_alt = z3.Implies(environment_alt, desired_alt)

verifier.add(definitions)
verifier.add(definitions_alt)

# This is how 2 CCAs are defined
verifier.add(z3.And(*template_definitions))

# Under same verifier decisions
verifier.add(z3.And(*same_decisions))

# Find a trace such that main works but alt does not
verifier.add(specification)
verifier.add(z3.Not(specification_alt))
verifier.add(d.fefficient)
verifier.add(z3.Not(d_alt.fefficient))


optimization_list = [
    Metric(cc.desired_util_f, 0.8, 1, 0.001, True),
    Metric(cc.desired_queue_bound_multiplier, 1, 2, 0.001, False),
    Metric(cc.desired_loss_count_bound, 0, 3, 0.001, False),
    Metric(cc.desired_loss_amount_bound_multiplier, 0, 2, 0.001, False),
]

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

    import ipdb; ipdb.set_trace()