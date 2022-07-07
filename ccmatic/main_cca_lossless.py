from fractions import Fraction

import z3
from ccac.variables import VariableNames

from ccmatic.common import flatten
from ccmatic.cegis import CegisCCAGen

from .verifier import (desired_high_util_low_delay, setup_ccac,
                       setup_ccac_definitions, setup_ccac_environment)

lag = 1
history = 4

# Verifier
# Dummy variables used to create CCAC formulation only
c, v, = setup_ccac()
sd = setup_ccac_definitions(c, v)
se = setup_ccac_environment(c, v)
ccac_definitions = sd.assertions()
environment = se.assertions()

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
util_frac = 0.50
delay_bound = 1.8 * c.C * (c.R + c.D)

desired = desired_high_util_low_delay(c, v, first, util_frac, delay_bound)
assert isinstance(desired, z3.ExprRef)

# Generator definitions
vn = VariableNames(v)
rhs_var_symbols = ['c_f[0]', 'S_f[0]']
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
domain_clauses = []
for coeff in flatten(coeffs.values()):
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
            this_term = get_product_ite(rvar[time_idx], this_coeff)
            term_list.append(this_term)
    expr = z3.Sum(*term_list)
    assert isinstance(expr, z3.ArithRef)
    return expr


for lvar_symbol in lhs_var_symbols:
    lower_bound = lvar_lower_bounds[lvar_symbol]
    for t in range(first, c.T):
        lvar = eval('v.{}'.format(lvar_symbol))
        rhs = get_expr(lvar_symbol, t)
        definition_constrs.append(
            lvar == z3.If(rhs >= lower_bound, rhs, lower_bound))

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)
definitions = z3.And(ccac_definitions, *definition_constrs)

generator_vars = (flatten(list(coeffs.values())) +
                  flatten(list(consts.values())))
verifier_vars = flatten(
    [v.A_f[:history], v.c_f[:history], v.S_f, v.W,
     v.L_f, v.epsilon, v.dupacks, v.alpha])
definition_vars = flatten(
    [v.A_f[history:], v.A, v.c_f[history:],
     v.r_f, v.Ld_f, v.S, v.L, v.timeout_f, v.qdel])

cg = CegisCCAGen(generator_vars, verifier_vars, definition_vars,
                 search_constraints, definitions, specification, ctx)
cg.run()