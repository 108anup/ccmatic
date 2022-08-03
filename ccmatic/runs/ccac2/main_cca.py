from fractions import Fraction
from typing import List, Set
import z3
from ccac2.config import Config
from ccac2.model import ModelVariables, all_constraints, config_constraints
from ccmatic.verifier.ccac2_stubs import get_cegis_vars, separate_initial_conditions, setup_definitions, setup_environment
from cegis import Cegis
from pyz3_utils.helpers import extract_all_aux_vars, extract_vars
from pyz3_utils.my_solver import MySolver


c = Config()
c.unsat_core = False
c.F = 2
c.inf_buf = False
init_rtts = 4
init_tsteps = 6
total_rtts = 10
c.T = 20  # total tsteps
assert c.T > init_tsteps
c.check()

s = MySolver()
v = ModelVariables(c, s)

s.add(v.times[-1].time >= total_rtts)
separate_initial_conditions(c, s, v, init_tsteps, init_rtts)
config_constraints(c, s, v)

ccac2_domain = z3.And(*s.assertion_list)
verifier_vars, definition_vars = get_cegis_vars(c, v, init_tsteps)
sd = setup_definitions(c, v)
se = setup_environment(c, v)

def_aux_vars = extract_all_aux_vars(z3.And(*sd.assertion_list))
env_aux_vars = extract_all_aux_vars(z3.And(*se.assertion_list))
assert len(env_aux_vars) == 0

definition_vars.extend(def_aux_vars)

# TODO(108anup): cross check that union of definition and env vars is same as
#  original vars.
environment = z3.And(*se.assertion_list)


# Search constr
coeffs = {
    ''
}
search_range = [Fraction(i, 2) for i in range(-4, 5)]
domain_clauses = []
for coeff in flatten(list(coeffs.values())) + flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range]))
domain_clauses.append(z3.Or(
    *[qsize_thresh == val for val in qsize_thresh_choices]))
search_constraints = z3.And(*domain_clauses)
assert(isinstance(search_constraints, z3.ExprRef))

# Definitions (Template)
definition_constrs = []


def get_product_ite(coeff, rvar, cdomain=search_range):
    term_list = []
    for val in cdomain:
        term_list.append(z3.If(coeff == val, val * rvar, 0))
    return z3.Sum(*term_list)


assert first >= 1
for t in range(first, c.T):
    assert history > lag
    assert lag == 1
    assert c.R == 1
    # loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]

    # This is meaningless as c.C * (c.R + c.D) is unknown...
    # loss_detected = (v.A_f[0][t-lag] - v.Ld_f[0][t]
    #                  - v.S_f[0][t-lag] >= qsize_thresh * c.C * (c.R + c.D))

    for dt in range(c.T):
        definition_constrs.append(
            z3.Implies(z3.And(dt == qsize_thresh, v.qbound[t-lag][dt]),
                       exceed_queue_f[0][t]))
        definition_constrs.append(
            z3.Implies(z3.And(dt == qsize_thresh, z3.Not(v.qbound[t-lag][dt])),
                       z3.Not(exceed_queue_f[0][t])))
    loss_detected = exceed_queue_f[0][t]

    acked_bytes = v.S_f[0][t-lag] - v.S_f[0][t-history]
    rhs_loss = (get_product_ite(coeffs['c_f[0]_loss'], v.c_f[0][t-lag])
                + get_product_ite(coeffs['ack_f[0]_loss'], acked_bytes)
                + consts['c_f[0]_loss'])
    rhs_noloss = (get_product_ite(coeffs['c_f[0]_noloss'], v.c_f[0][t-lag])
                  + get_product_ite(coeffs['ack_f[0]_noloss'], acked_bytes)
                  + consts['c_f[0]_noloss'])
    rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
    assert isinstance(rhs, z3.ArithRef)
    definition_constrs.append(
        v.c_f[0][t] == z3.If(rhs >= lower_bound, rhs, lower_bound)
    )

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_domain, ccac_definitions, *definition_constrs)
assert isinstance(definitions, z3.ExprRef)

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())) + [qsize_thresh])


try:
    cg = Cegis(generator_vars, verifier_vars, definition_vars,
               search_constraints, definitions, specification, ctx,
               known_solution)
    # cg.get_solution_str = get_solution_str
    # cg.get_counter_example_str = get_counter_example_str
    # cg.get_generator_view = get_generator_view
    # cg.get_verifier_view = get_verifier_view
    # run_verifier = functools.partial(
    #     run_verifier_incomplete, c=c, v=v, ctx=ctx)
    # cg.run_verifier = run_verifier
    cg.run()

except Exception:
    import sys
    import traceback

    import ipdb
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)