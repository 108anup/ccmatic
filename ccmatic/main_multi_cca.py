import functools
import logging
from fractions import Fraction
from typing import List

import z3
from ccac.config import ModelConfig
from ccac.variables import VariableNames, Variables
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

import ccmatic.common  # Used for side effects
from ccmatic.cegis import CegisCCAGen
from ccmatic.common import flatten

from .verifier import (desired_high_util_low_delay, get_cex_df, get_gen_cex_df,
                       run_verifier_incomplete, setup_ccac_definitions,
                       setup_ccac_environment)

logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


DEBUG = False
lag = 1
history = 4
use_loss = False
deterministic_loss = False
util_frac = 0.5
# n_losses = 1
ideal_max_queue = 2

# Verifier
# Dummy variables used to create CCAC formulation only
# Config
# c, s, v = setup_ccac()
c = ModelConfig.default()
c.compose = True
c.cca = "paced"
c.simplify = False
c.calculate_qdel = False
c.C = 100
c.T = 9
c.N = 2

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

conditional_vvars = []
if(not c.compose):
    conditional_vvars.append(v.epsilon)
conditional_dvars = []
if(c.calculate_qdel):
    conditional_dvars.append(v.qdel)

assert c.N == 2

# Loss detected at time 0 is unconstrained...
# Let verifier choose it, it is not used anywhere.
if(deterministic_loss):
    verifier_vars = flatten(
        [v.A_f[0][:history], v.c_f[0][:history], v.S_f, v.W, v.Ld_f[0][0],
         v.dupacks, v.alpha, conditional_vvars, v.C0]) + flatten(
        [v.A_f[1][:history], v.c_f[1][:history], v.Ld_f[1][0]])
    definition_vars = flatten(
        [v.A_f[0][history:], v.A, v.c_f[0][history:], v.L_f, v.Ld_f[0][1:],
         v.r_f, v.S, v.L, v.timeout_f, conditional_dvars]) + flatten(
        [v.A_f[1][history:], v.A, v.c_f[1][history:], v.Ld_f[1][1:]])

else:
    verifier_vars = flatten(
        [v.A_f[0][:history], v.c_f[0][:history], v.S_f, v.W, v.L_f,
         v.Ld_f[0][0], v.dupacks, v.alpha, conditional_vvars, v.C0]) + \
        flatten([v.A_f[1][:history], v.c_f[1][:history], v.Ld_f[1][0]])
    definition_vars = flatten(
        [v.A_f[0][history:], v.A, v.c_f[0][history:], v.Ld_f[0][1:],
         v.r_f, v.S, v.L, v.timeout_f, conditional_dvars]) + flatten(
        [v.A_f[1][history:], v.A, v.c_f[1][history:], v.Ld_f[1][1:]])

# No loss
if(not use_loss):
    verifier_vars = flatten(
        [v.A_f[0][:history], v.c_f[0][:history], v.S_f, v.W,
         v.L_f, v.Ld_f, v.dupacks, v.alpha, conditional_vvars, v.C0]) + \
        flatten([v.A_f[1][:history], v.c_f[1][:history]])
    definition_vars = flatten(
        [v.A_f[0][history:], v.A, v.c_f[0][history:],
         v.r_f, v.S, v.L, v.timeout_f, conditional_dvars]) + flatten(
        [v.A_f[1][history:], v.A, v.c_f[1][history:]])

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
# loss_rate = n_losses / ((c.T-1) - first)
delay_bound = ideal_max_queue * c.C * (c.R + c.D)

(desired, high_util, low_delay, ramp_up, ramp_down) = \
    desired_high_util_low_delay(c, v, first, util_frac, delay_bound)
assert isinstance(desired, z3.ExprRef)

# Generator definitions
# Loss/no-loss mean high-queue and low-queue
vn = VariableNames(v)
lower_bound = 0.01
coeffs = {
    'c_f[n]_loss': z3.Real('Gen__coeff_c_f[n]_loss'),
    'c_f[n]_noloss': z3.Real('Gen__coeff_c_f[n]_noloss'),
    'ack_f[n]_loss': z3.Real('Gen__coeff_ack_f[n]_loss'),
    'ack_f[n]_noloss': z3.Real('Gen__coeff_ack_f[n]_noloss')
}

consts = {
    'c_f[n]_loss': z3.Real('Gen__const_c_f[n]_loss'),
    'c_f[n]_noloss': z3.Real('Gen__const_c_f[n]_noloss'),
}

qsize_thresh = z3.Real('Gen__const_qsize_thresh')
qsize_thresh_choices = [Fraction(i, 8) for i in range(2 * 8 + 1)]
assert isinstance(qsize_thresh, z3.ArithRef)

# Search constr
search_range = [Fraction(i, 2) for i in range(5)]
# search_range = [-1, 0, 1]
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
for n in range(c.N):
    for t in range(first, c.T):
        assert history > lag
        assert lag == 1
        assert c.R == 1
        # loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]
        loss_detected = (v.A_f[n][t-1] - v.Ld_f[n][t] - v.S_f[n][t-1]
                         >= qsize_thresh * c.C * (c.R + c.D))
        acked_bytes = v.S_f[0][t-lag] - v.S_f[0][t-history]
        rhs_loss = (get_product_ite(coeffs['c_f[n]_loss'], v.c_f[0][t-lag])
                    + get_product_ite(coeffs['ack_f[n]_loss'], acked_bytes)
                    + consts['c_f[n]_loss'])
        rhs_noloss = (get_product_ite(coeffs['c_f[n]_noloss'], v.c_f[0][t-lag])
                      + get_product_ite(coeffs['ack_f[n]_noloss'], acked_bytes)
                      + consts['c_f[n]_noloss'])
        rhs = z3.If(loss_detected, rhs_loss, rhs_noloss)
        assert isinstance(rhs, z3.ArithRef)
        definition_constrs.append(
            v.c_f[n][t] == z3.If(rhs >= lower_bound, rhs, lower_bound)
        )

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_domain, ccac_definitions, *definition_constrs)
assert isinstance(definitions, z3.ExprRef)

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())) + [qsize_thresh])


# Method overrides
# These use function closures, hence have to be defined here.
# Can use partial functions to use these elsewhere.


def get_counter_example_str(counter_example: z3.ModelRef,
                            verifier_vars: List[z3.ExprRef]) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    ret = "{}".format(df)
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


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef], n_cex: int) -> str:
    rhs_loss = (f"{solution.eval(coeffs['c_f[n]_loss'])}"
                f"c_f[n][t-{lag}]"
                f" + {solution.eval(coeffs['ack_f[n]_loss'])}"
                f"(S_f[n][t-{lag}]-S_f[n][t-{history}])"
                f" + {solution.eval(consts['c_f[n]_loss'])}")
    rhs_noloss = (f"{solution.eval(coeffs['c_f[n]_noloss'])}"
                  f"c_f[n][t-{lag}]"
                  f" + {solution.eval(coeffs['ack_f[n]_noloss'])}"
                  f"(S_f[n][t-{lag}]-S_f[n][t-{history}])"
                  f" + {solution.eval(consts['c_f[n]_noloss'])}")
    ret = (f"if(A_f[n][t-1] - Ld_f[n][t] - S_f[n][t-1]"
           f" >= {solution.eval(qsize_thresh)} * C * (R + D)):\n"
           f"\tc_f[n][t] = max({lower_bound}, {rhs_loss})\n"
           f"else:\n"
           f"\tc_f[n][t] = max({lower_bound}, {rhs_noloss})")
    return ret


def get_verifier_view(
            counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
            definition_vars: List[z3.ExprRef]) -> str:
    return get_counter_example_str(counter_example, verifier_vars)


def get_generator_view(solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                       definition_vars: List[z3.ExprRef], n_cex: int) -> str:
    gen_view_str = "{}".format(get_gen_cex_df(solution, v, vn, n_cex, c))
    return gen_view_str


# Known solution
known_solution = None

# known_solution_list = []
# known_solution_list.append(coeffs['c_f[0]_loss'] == 1/2)
# known_solution_list.append(coeffs['ack_f[0]_loss'] == 0)
# known_solution_list.append(consts['c_f[0]_loss'] == 0)

# known_solution_list.append(coeffs['c_f[0]_noloss'] == 1)
# known_solution_list.append(coeffs['ack_f[0]_noloss'] == 0)
# known_solution_list.append(consts['c_f[0]_noloss'] == 1)
# known_solution = z3.And(*known_solution_list)
# assert(isinstance(known_solution, z3.ExprRef))

# Debugging:
if DEBUG:
    search_constraints = z3.And(search_constraints, known_solution)
    assert(isinstance(search_constraints, z3.ExprRef))

    # known_solver = MySolver()
    # known_solver.warn_undeclared = False
    # known_solver.add(known_solution)
    # print(known_solver.check())
    # print(known_solver.model())

    # Search constraints
    with open('tmp/search.txt', 'w') as f:
        f.write(search_constraints.sexpr())

    # Definitions (including template)
    with open('tmp/definitions.txt', 'w') as f:
        f.write(definitions.sexpr())

try:
    cg = CegisCCAGen(generator_vars, verifier_vars, definition_vars,
                     search_constraints, definitions, specification, ctx,
                     known_solution)
    cg.get_solution_str = get_solution_str
    cg.get_counter_example_str = get_counter_example_str
    cg.get_generator_view = get_generator_view
    cg.get_verifier_view = get_verifier_view
    run_verifier = functools.partial(
        run_verifier_incomplete, c=c, v=v, ctx=ctx)
    cg.run_verifier = run_verifier
    cg.run()

except Exception:
    import sys
    import traceback

    import ipdb
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
