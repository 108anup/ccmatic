import numpy as np
import z3
from ccmatic.cegis import CegisConfig
from ccmatic.common import get_val_list
from ccmatic.verifier import (get_cex_df, get_desired_necessary, get_desired_ss_invariant,
                              get_periodic_constraints, setup_cegis_basic)
from cegis.util import Metric
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames

cc = CegisConfig()
cc.history = 4
cc.infinite_buffer = True
cc.template_queue_bound = False
cc.N = 2
cc.T = 15
# cc.synth_ss = True

cc.desired_util_f = z3.Real('desired_util_f')
cc.desired_queue_bound_multiplier = z3.Real('desired_queue_bound_multiplier')
cc.desired_loss_count_bound = z3.Real('desired_loss_bound')
cc.desired_loss_amount_bound_multiplier = z3.Real('desired_loss_amount_bound')
(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)

if(cc.synth_ss):
    d = get_desired_ss_invariant(cc, c, v)
    desired = d.desired_invariant
else:
    d = get_desired_necessary(cc, c, v)
    desired = d.desired_necessary

domain_clauses = []
if(cc.synth_ss):
    assert d.steady_state_variables
    sv_dict = {sv.name: sv for sv in d.steady_state_variables}
    domain_clauses.extend([
        sv_dict['cwnd'].lo == 0.5 * c.C * (c.R),
        sv_dict['cwnd'].hi == (cc.history-1) * c.C * (c.R + c.D),
        sv_dict['queue'].lo == 0,
        sv_dict['queue'].hi == 3 * c.C * (c.R + c.D),
    ])

vn = VariableNames(v)
first = cc.history  # First cwnd idx decided by synthesized cca
template_definitions = []
for n in range(c.N):
    for t in range(first, c.T):
        acked_bytes = v.S_f[n][t-c.R] - v.S_f[n][t-cc.history]

        cond = True

        rhs_loss = acked_bytes + 1/2
        rhs_noloss = acked_bytes + 1/2

        # rhs_loss = v.c_f[n][t-1] / 2
        # rhs_noloss = v.c_f[n][t-lag] + 1

        rhs = z3.If(cond, rhs_loss, rhs_noloss)
        assert isinstance(rhs, z3.ArithRef)
        template_definitions.append(
            v.c_f[n][t] == z3.If(rhs >= cc.template_cca_lower_bound,
                                 rhs, cc.template_cca_lower_bound))

        # template_definitions.append(v.c_f[n][t] == c.C/c.N)
        # template_definitions.append(v.c_f[n][t] == 4096)
        # template_definitions.append(v.c_f[n][t] == cc.template_cca_lower_bound)


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)

    tot_cwnd = [
        counter_example.eval(z3.Sum(*v.c_f[:, t])).as_fraction()
        for t in range(c.T)]
    df['tot_cwnd_t'] = tot_cwnd

    rocc_cwnd = ([-1 for t in range(first)] +
                 [max(
                     cc.template_cca_lower_bound,
                     counter_example.eval(v.S[t-1] - v.S[t-4]).as_fraction())
                  for t in range(first, c.T)])
    rocc_arrival = [counter_example.eval(
        v.A[t]).as_fraction() for t in range(first)]
    for t in range(first, c.T):
        rocc_arrival.append(max(
            rocc_arrival[t-1],
            counter_example.eval(v.S[t-c.R] + rocc_cwnd[t]).as_fraction()))
    df['arrival_rocc_t'] = rocc_arrival
    df['cwnd_rocc_t'] = rocc_cwnd

    for n in range(c.N):
        df[f'diff_service_{n},t'] = [-1] + \
            [counter_example.eval(v.S_f[n][t] - v.S_f[n][t-1]).as_fraction()
             for t in range(1, c.T)]
    df['diff_tot_service_t'] = [-1] + \
        [counter_example.eval(v.S[t] - v.S[t-1]).as_fraction()
         for t in range(1, c.T)]

    for n in range(c.N):
        df[f'diff_arrival_{n},t'] = [-1] + \
            [counter_example.eval(v.A_f[n][t] - v.A_f[n][t-1]).as_fraction()
             for t in range(1, c.T)]
    df['diff_tot_arrival_t'] = [-1] + \
        [counter_example.eval(v.A[t] - v.A[t-1]).as_fraction()
         for t in range(1, c.T)]

    for n in range(c.N):
        df[f'queue_{n},t'] = [
            counter_example.eval(
                v.A_f[n][t] - v.L_f[n][t] - v.S_f[n][t]).as_fraction()
            for t in range(c.T)]

    desired_string = d.to_string(cc, c, counter_example)
    ret = "{}\n{}.".format(df.astype(float), desired_string)

    qdel_vals = []
    for qdel_list in v.qdel:
        qdel_val_list = get_val_list(counter_example, qdel_list)
        qdel_vals.append(qdel_val_list)
    ret += "\n{}".format(np.array(qdel_vals).astype(int))
    return ret


optimization_list = [
    Metric(cc.desired_util_f, 0.1, 1, 0.001, True),
    Metric(cc.desired_queue_bound_multiplier, 1, 8, 0.001, False),
    Metric(cc.desired_loss_count_bound, 0, 0, 0.001, False),
    Metric(cc.desired_loss_amount_bound_multiplier, 0, 0, 0.001, False),
]

verifier = MySolver()
verifier.warn_undeclared = False
verifier.add(ccac_domain)
verifier.add(ccac_definitions)
verifier.add(environment)
verifier.add(z3.And(*domain_clauses))
verifier.add(z3.And(*template_definitions))
verifier.add(z3.Not(desired))

# # Initial state
# verifier.add(z3.Sum(*[v.c_f[n][first] for n in range(c.N)]) >= c.C * (c.R + c.D))
# verifier.add(z3.Sum(*[v.c_f[n][first] for n in range(c.N)]) <= (cc.history-1) * c.C * (c.R + c.D))
# verifier.add(v.A[first] - v.L[first] - v.S[first] >= 0)
# verifier.add(v.A[first-1] - v.L[first-1] - v.S[first-1] <= 1.5 * c.C * (c.R + c.D))


# Check performant trace
# verifier.add(desired)

# Periodic counter example
# verifier.add(get_periodic_constraints(cc, c, v))
# verifier.add(z3.And(high_util, v.L[-3] > v.L[first]))

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
print("Done")
