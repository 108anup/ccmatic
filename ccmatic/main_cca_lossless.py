import z3

from .verifier import setup_ccac

history = 4

# Verifier
# Dummy variables used to create CCAC formulation only
c, s, v = setup_ccac("paced")
# Shouldn't be any loss at t0 otherwise cwnd is high and q is still 0.
s.add(v.L_f[0][0] == 0)
environment = s.assertions()

# Desired properties
first = history  # First cwnd idx decided by synthesized cca
util_frac = 0.50
delay_bound = 1.8 * c.C * (c.R + c.D)

cond_list = []
for t in range(first, c.T):
    cond_list.append(v.A[t] - v.L[t] - v.S[t] <= delay_bound)
# Queue seen by a new packet should not be more that delay_bound
low_delay = z3.And(*cond_list)
# Serviced shoulf be at least util_frac that could have been serviced
high_util = v.S[-1] - v.S[first] >= util_frac * c.C * (c.T-1-first-c.D)
# If the cwnd0 is very low then CCA should increase cwnd
ramp_up = v.c_f[0][-1] > v.c_f[0][first]
# If the queue is large to begin with then, CCA should cause queue to decrease.
ramp_down = v.A[-1] - v.L[-1] - v.S[-1] < v.A[first] - v.L[first] - v.S[first]

desired = z3.And(
    z3.Or(high_util, ramp_up),
    z3.Or(low_delay, ramp_down))
assert isinstance(desired, z3.ExprRef)

# Generator definitions
rhs_vars = [v.c_f[0], v.S_f[0]]
lhs_vars = [v.c_f[0]]
n_coeffs = len(rhs_vars) * history
n_const = 1

# Coeff for determining rhs var, of lhs var, at shift t
coeffs = {
    lvar: [
        [z3.Real('Gen__coeff_{}_{}_{}'.format(lvar, rvar, t))
         for t in range(history)]
        for rvar in rhs_vars
    ] for lvar in lhs_vars
}

# Search constr

# Definitions
definitions = None

# CCmatic inputs
ctx = z3.main_ctx()
specification = z3.Implies(environment, desired)