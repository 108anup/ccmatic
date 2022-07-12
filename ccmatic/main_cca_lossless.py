import logging
import numpy as np
import pandas as pd
from fractions import Fraction
from typing import List, Tuple, Optional

import z3
from ccac.variables import VariableNames

from ccmatic.common import flatten, get_name_for_list
from ccmatic.cegis import CegisCCAGen
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver
from pyz3_utils.binary_search import BinarySearch
from pyz3_utils.small_denom import find_small_denom_soln

from .verifier import (desired_high_util_low_delay, setup_ccac,
                       setup_ccac_definitions, setup_ccac_environment)


logger = logging.getLogger('cca_gen')
GlobalConfig().default_logger_setup(logger)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


lag = 1
history = 4

# Verifier
# Dummy variables used to create CCAC formulation only
c, s, v = setup_ccac()
ccac_domain = z3.And(*s.assertion_list)
sd = setup_ccac_definitions(c, v)
se = setup_ccac_environment(c, v)
ccac_definitions = z3.And(*sd.assertion_list)
environment = z3.And(*se.assertion_list)

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
util_frac = 0.5
delay_bound = 2 * c.C * (c.R + c.D)

(desired, high_util, low_delay, ramp_up, ramp_down) = \
    desired_high_util_low_delay(c, v, first, util_frac, delay_bound)
assert isinstance(desired, z3.ExprRef)

# Generator definitions
vn = VariableNames(v)
rhs_var_symbols = ['S_f[0]']
# rhs_var_symbols = ['c_f[0]', 'S_f[0]']
lhs_var_symbols = ['c_f[0]']
lvar_lower_bounds = {
    'c_f[0]': 0.01
}
n_coeffs = len(rhs_var_symbols) * history
n_const = 1

# Coeff for determining rhs var, of lhs var, at shift t
coeffs = {
    lvar: [
        [z3.Real('Gen__coeff_{}_{}_{}'.format(lvar, rvar, h))
         for h in range(history)]
        for rvar in rhs_var_symbols
    ] for lvar in lhs_var_symbols
}
consts = {
    lvar: z3.Real('Gen__const_{}'.format(lvar))
    for lvar in lhs_var_symbols
}

# Search constr
search_range = [Fraction(i, 2) for i in range(-4, 5)]
search_range = [-1, 0, 1]
domain_clauses = []
for coeff in flatten(list(coeffs.values())) + flatten(list(consts.values())):
    domain_clauses.append(z3.Or(*[coeff == val for val in search_range]))
search_constraints = z3.And(*domain_clauses)

# Definitions
definition_constrs = []


def get_product_ite(coeff, rvar):
    term_list = []
    for val in search_range:
        term_list.append(z3.If(coeff == val, val * rvar, 0))
    return z3.Sum(*term_list)


def get_expr(lvar_symbol, t) -> z3.ArithRef:
    term_list = []
    for rvar_idx in range(len(rhs_var_symbols)):
        rvar_symbol = rhs_var_symbols[rvar_idx]
        for h in range(history):
            this_coeff = coeffs[lvar_symbol][rvar_idx][h]
            time_idx = t - lag - h
            rvar = eval('v.{}'.format(rvar_symbol))
            this_term = get_product_ite(this_coeff, rvar[time_idx])
            term_list.append(this_term)
    expr = z3.Sum(*term_list) + consts[lvar_symbol]
    assert isinstance(expr, z3.ArithRef)
    return expr


for lvar_symbol in lhs_var_symbols:
    lower_bound = lvar_lower_bounds[lvar_symbol]
    for t in range(first, c.T):
        lvar = eval('v.{}'.format(lvar_symbol))
        rhs = get_expr(lvar_symbol, t)
        definition_constrs.append(
            lvar[t] == z3.If(rhs >= lower_bound, rhs, lower_bound))

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_domain, ccac_definitions, *definition_constrs)

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))
conditional_vvars = []
if(not c.compose):
    conditional_vvars.append(v.epsilon)
conditional_dvars = []
if(c.calculate_qdel):
    conditional_dvars.append(v.qdel)

verifier_vars = flatten(
    [v.A_f[0][:history], v.c_f[0][:history], v.S_f, v.W,
     v.L_f, v.dupacks, v.alpha, conditional_vvars, v.C0])
definition_vars = flatten(
    [v.A_f[0][history:], v.A, v.c_f[0][history:],
     v.r_f, v.Ld_f, v.S, v.L, v.timeout_f, conditional_dvars])


# Method overrides
# These use function closures, hence have to be defined here.
# Can use partial functions to use these elsewhere.


def get_counter_example_str(counter_example: z3.ModelRef,
                            verifier_vars: List[z3.ExprRef]) -> str:
    def _get_model_value(l):
        ret = []
        for vvar in l:
            ret.append(counter_example.eval(vvar).as_fraction())
        return ret
    cex_dict = {
        get_name_for_list(vn.A_f[0]): _get_model_value(v.A_f[0]),
        get_name_for_list(vn.c_f[0]): _get_model_value(v.c_f[0]),
        get_name_for_list(vn.S_f[0]): _get_model_value(v.S_f[0]),
        get_name_for_list(vn.W): _get_model_value(v.W),
        get_name_for_list(vn.L): _get_model_value(v.L),
    }
    df = pd.DataFrame(cex_dict).astype(float)
    ret = "{}".format(df)
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


def get_solution_str(solution: z3.ModelRef,
                     generator_vars: List[z3.ExprRef]) -> str:
    assert(len(lhs_var_symbols) == 1)
    lvar_symbol = "c_f[0]"
    rhs_expr = ""
    for rvar_idx in range(len(rhs_var_symbols)):
        rvar_symbol = rhs_var_symbols[rvar_idx]
        for h in range(history):
            this_coeff = coeffs[lvar_symbol][rvar_idx][h]
            this_coeff_val = solution.eval(this_coeff).as_fraction()
            if(this_coeff_val != 0):
                rhs_expr += "+ {}{}[t-{}] ".format(
                    this_coeff_val, rvar_symbol, h+1)
    this_const = consts[lvar_symbol]
    this_const_val = solution.eval(this_const)
    rhs_expr += "+ {}".format(this_const_val)
    ret = "{}[t] = max({}, {})".format(
        lvar_symbol, lvar_lower_bounds[lvar_symbol], rhs_expr)
    return ret


def maximize_gap(
    verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    """
    Adds constraints as a side effect to the verifier formulation
    that try to maximize the gap between
    service curve and the upper bound of service curve (based on waste choices).
    """
    s = verifier
    # Get c, v, from closure

    orig_sat = s.check()
    if str(orig_sat) != "sat":
        return orig_sat, None

    orig_model = s.model()
    cur_min = np.inf
    for t in range(1, c.T):
        if(orig_model.eval(v.W[t] > v.W[t-1])):
            this_gap = orig_model.eval(
                v.C0 + c.C * t - v.W[t] - v.S[t]).as_fraction()
            cur_min = min(cur_min, this_gap)

    # if(cur_min == np.inf):
    #     return orig_sat

    binary_search = BinarySearch(0, c.C * c.D, c.C * c.D/64)
    min_gap = z3.Real('min_gap', ctx=ctx)

    for t in range(1, c.T):
        s.add(z3.Implies(
            v.W[t] > v.W[t-1],
            min_gap <= v.C0 + c.C * t - v.W[t] - v.S[t]))

    while True:
        pt = binary_search.next_pt()
        if pt is None:
            break

        s.push()
        s.add(min_gap >= pt)
        sat = s.check()
        s.pop()

        if(str(sat) == 'sat'):
            binary_search.register_pt(pt, 1)
        elif str(sat) == "unknown":
            binary_search.register_pt(pt, 2)
        else:
            assert str(sat) == "unsat", f"Unknown value: {str(sat)}"
            binary_search.register_pt(pt, 3)

    best_gap, _, _ = binary_search.get_bounds()
    s.add(min_gap >= best_gap)

    sat = s.check()
    assert sat == orig_sat
    model = s.model()

    logger.info("Improved gap from {} to {}".format(cur_min, best_gap))
    return sat, model


def run_verifier(
    verifier: MySolver
) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
    _, _ = maximize_gap(verifier)
    sat, _, model = find_small_denom_soln(verifier, 4096)
    return sat, model


# Debugging:
if False:
    # Known solution
    lvar_symbol = "c_f[0]"
    rvar_idx = 0
    known_solution = z3.And(coeffs[lvar_symbol][rvar_idx][0] == 1,
                            coeffs[lvar_symbol][rvar_idx][1] == 0,
                            coeffs[lvar_symbol][rvar_idx][2] == 0,
                            coeffs[lvar_symbol][rvar_idx][3] == -1,
                            consts[lvar_symbol] == 0)
    search_constraints = z3.And(search_constraints, known_solution)

    # Definitions (including template)
    with open('definitions.txt', 'w') as f:
        assert(isinstance(definitions, z3.ExprRef))
        f.write(definitions.sexpr())

try:

    cg = CegisCCAGen(generator_vars, verifier_vars, definition_vars,
                     search_constraints, definitions, specification, ctx)
    cg.get_solution_str = get_solution_str
    cg.get_counter_example_str = get_counter_example_str
    cg.run_verifier = run_verifier
    cg.run()

except Exception:
    import sys
    import traceback

    import ipdb
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    ipdb.post_mortem(tb)
