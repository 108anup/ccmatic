import functools
import logging
from fractions import Fraction
from typing import List

import numpy as np
import z3
from ccac.config import ModelConfig
from ccac.variables import VariableNames, Variables
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

import ccmatic.common  # Used for side effects
from ccmatic.cegis import CegisCCAGen
from ccmatic.common import flatten

from .verifier import (desired_high_util_low_delay, get_cegis_vars,
                       get_cex_df, get_gen_cex_df, run_verifier_incomplete,
                       setup_ccac_definitions, setup_ccac_environment)

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
verifier_vars, definition_vars = get_cegis_vars(c, v, history)
exceed_queue_f = [[z3.Bool(f"Def__exceed_queue_{n},{t}") for t in range(c.T)]
                  for n in range(c.N)]  # definition variable
last_decrease_f = [[z3.Real(f"Def__last_decrease_{n},{t}") for t in range(c.T)]
                   for n in range(c.N)]  # definition variable
definition_vars.extend(flatten(exceed_queue_f))
definition_vars.extend(flatten(last_decrease_f))

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
    'c_f[0]_loss': z3.Real('Gen__coeff_c_f[0]_loss'),
    'c_f[0]_noloss': z3.Real('Gen__coeff_c_f[0]_noloss'),
    'ack_f[0]_loss': z3.Real('Gen__coeff_ack_f[0]_loss'),
    'ack_f[0]_noloss': z3.Real('Gen__coeff_ack_f[0]_noloss')
}

consts = {
    'c_f[0]_loss': z3.Real('Gen__const_c_f[0]_loss'),
    'c_f[0]_noloss': z3.Real('Gen__const_c_f[0]_noloss'),
}

# This is in multiples of Rm
qsize_thresh = z3.Real('Gen__const_qsize_thresh')
# qsize_thresh_choices = [Fraction(i, 8) for i in range(2 * 8 + 1)]
qsize_thresh_choices = [x for x in range(1, c.T)]
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
assert history >= lag
assert lag == c.R
assert c.R == 1

# definition_constrs.append(last_decrease_f[0][0] == v.A_f[0][0] - v.L_f[0][0])
definition_constrs.append(last_decrease_f[0][0] == v.S_f[0][0])
for t in range(1, first):
    definition_constrs.append(
        z3.Implies(v.c_f[0][t] < v.c_f[0][t-1],
                   last_decrease_f[0][t] == v.A_f[0][t] - v.L_f[0][t]))
    definition_constrs.append(
        z3.Implies(v.c_f[0][t] >= v.c_f[0][t-1],
                   last_decrease_f[0][t] == last_decrease_f[0][t-1]))
    # Const last_decrease in history
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
    # loss_detected = v.Ld_f[0][t] > v.Ld_f[0][t-1]

    # This is meaningless as c.C * (c.R + c.D) is unknown...
    # loss_detected = (v.A_f[0][t-lag] - v.Ld_f[0][t]
    #                  - v.S_f[0][t-lag] >= qsize_thresh * c.C * (c.R + c.D))

    loss_detected = exceed_queue_f[0][t]

    # Decrease this time iff queue exceeded AND new qdelay measurement (i.e.,
    # new packets received) AND in the previous cycle we had received all the
    # packets sent since last decrease (S_f[n][t-lag-1] >=
    # last_decrease[n][t-1])
    # TODO: see if we want to replace the last statement
    #  with (S_f[n][t-lag] > last_decrease[n][t-1])
    this_decrease = z3.And(loss_detected,
                           v.S_f[0][t-lag] > v.S_f[0][t-lag-1],
                           v.S_f[0][t-1-lag] >= last_decrease_f[0][t-1])

    definition_constrs.append(z3.Implies(
        this_decrease,
        last_decrease_f[0][t] == v.A_f[0][t] - v.L_f[0][t]))
    definition_constrs.append(z3.Implies(
        z3.Not(this_decrease),
        last_decrease_f[0][t] == last_decrease_f[0][t-1]))

    acked_bytes = v.S_f[0][t-lag] - v.S_f[0][t-history]
    rhs_loss = (get_product_ite(coeffs['c_f[0]_loss'], v.c_f[0][t-lag])
                + get_product_ite(coeffs['ack_f[0]_loss'], acked_bytes)
                + consts['c_f[0]_loss'])
    rhs_noloss = (get_product_ite(coeffs['c_f[0]_noloss'], v.c_f[0][t-lag])
                  + get_product_ite(coeffs['ack_f[0]_noloss'], acked_bytes)
                  + consts['c_f[0]_noloss'])
    rhs = z3.If(this_decrease, rhs_loss, rhs_noloss)
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


# Method overrides
# These use function closures, hence have to be defined here.
# Can use partial functions to use these elsewhere.


def get_counter_example_str(counter_example: z3.ModelRef,
                            verifier_vars: List[z3.ExprRef]) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    qdelay = []
    for t in range(c.T):
        assert bool(counter_example.eval(v.qbound[t][0]))
        for dt in range(c.T-1, -1, -1):
            value = counter_example.eval(v.qbound[t][dt])
            try:
                bool_value = bool(value)
            except z3.z3types.Z3Exception:
                bool_value = True
            if(bool_value):
                qdelay.append(dt+1)
                break
    assert len(qdelay) == c.T
    df["qdelay"] = np.array(qdelay).astype(float)
    df["last_decrease_f"] = np.array(
        [counter_example.eval(x).as_fraction()
         for x in last_decrease_f[0]]).astype(float)
    df["this_decrease"] = [-1] + [
        bool(counter_example.eval(z3.And(
            exceed_queue_f[0][t],
            v.S_f[0][t-lag] > v.S_f[0][t-lag-1],
            v.S_f[0][t-1-lag] >= last_decrease_f[0][t-1]
        )))
        for t in range(1, c.T)]
    df["exceed_queue_f"] = [-1] + \
        [bool(counter_example.eval(x)) for x in exceed_queue_f[0][1:]]

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
    rhs_loss = (f"{solution.eval(coeffs['c_f[0]_loss'])}"
                f"c_f[0][t-{lag}]"
                f" + {solution.eval(coeffs['ack_f[0]_loss'])}"
                f"(S_f[0][t-{lag}]-S_f[0][t-{history}])"
                f" + {solution.eval(consts['c_f[0]_loss'])}")
    rhs_noloss = (f"{solution.eval(coeffs['c_f[0]_noloss'])}"
                  f"c_f[0][t-{lag}]"
                  f" + {solution.eval(coeffs['ack_f[0]_noloss'])}"
                  f"(S_f[0][t-{lag}]-S_f[0][t-{history}])"
                  f" + {solution.eval(consts['c_f[0]_noloss'])}")
    ret = (f"if(qbound[t-1][{solution.eval(qsize_thresh)}]):\n"
           f"\tc_f[0][t] = max({lower_bound}, {rhs_loss})\n"
           f"else:\n"
           f"\tc_f[0][t] = max({lower_bound}, {rhs_noloss})")
    return ret


def get_verifier_view(
            counter_example: z3.ModelRef, verifier_vars: List[z3.ExprRef],
            definition_vars: List[z3.ExprRef]) -> str:
    return get_counter_example_str(counter_example, verifier_vars)


def get_generator_view(solution: z3.ModelRef, generator_vars: List[z3.ExprRef],
                       definition_vars: List[z3.ExprRef], n_cex: int) -> str:
    df = get_gen_cex_df(solution, v, vn, n_cex, c)
    gen_view_str = "{}".format(df)
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