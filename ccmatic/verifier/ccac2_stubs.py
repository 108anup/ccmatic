import z3
from typing import List, Tuple
from ccac2.config import Config
from ccac2.model import ModelVariables, initial, monotone, total
from pyz3_utils.my_solver import MySolver
from pyz3_utils.nonlinear import Piecewise


def separate_initial_conditions(
        c: Config, s: MySolver, v: ModelVariables,
        init_tsteps: int, init_rtts: int):
    # Add constraints like this to make things deterministic or
    # non-deterministic. E.g., self.times[5].time > 4 + self.times[0].time

    s.add(v.times[init_tsteps].time == v.times[0].time + init_rtts)


def loss(c: Config, s: MySolver, v: ModelVariables):
    for t in range(c.T):
        ts = v.times[t]
        if not c.inf_buf:
            s.add(ts.A - ts.L <= c.C * ts.time - ts.W + v.buf)

            if t == 0:
                continue
            # We can make loss deterministic since we assume all curves are
            # joined by straight lines. This does not lose generality since
            # `times[t].time` is a variable. Thus Z3 can approximate any
            # curve it likes with a piecewise linear curve (well...as long
            # as it fits within c.T points)

            s.add(z3.Implies(ts.L > v.times[t-1].L,
                             ts.A - ts.L ==
                             c.C * ts.time - ts.W + v.buf))
        else:
            s.add(ts.L == v.times[0].L)


def service_waste(c: Config, s: MySolver, v: ModelVariables):
    ''' The heart of the CCAC model '''
    for t in range(c.T):
        ts = v.times[t]
        for f in range(c.F):
            fl = ts.flows[f]
            s.add(fl.S <= fl.A - fl.L)
        s.add(ts.S <= c.C * ts.time - ts.W)

        # Do things at time ts.time - c.D

        # To begin with, if ts.time - c.D > 0, it should exist in the past
        s.add(z3.Or(ts.time < c.D,
              *[v.times[pt].time == ts.time - c.D for pt in range(t)]))

        # If ts.time - c.D < 0, then give maximum slack. This corresponds
        # to no wastage when t < 0
        s.add(c.C * (ts.time - c.D) - v.times[0].W <= ts.S)

        for pt in range(t):
            pts = v.times[t]
            s.add(z3.Implies(ts.time - c.D == pts.time,
                             c.C * pts.time - pts.W <= ts.S))

        if c.compose:
            if t > 0:
                s.add(z3.Implies(ts.W > v.times[t-1].W,
                                 ts.A - ts.L <= c.C * ts.time - ts.W))
        else:
            if t > 0:
                s.add(z3.Implies(ts.W > v.times[t-1].W,
                                 ts.A - ts.L <= ts.S + v.epsilon))


def get_cegis_vars(
    c: Config, v: ModelVariables, init_tsteps: int
) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
    verifier_vars = []
    verifier_vars.append(v.buf)
    if not c.compose:
        verifier_vars.append(v.epsilon)
    if not c.inf_buf:
        verifier_vars.append(v.buf)  # Buffer size
    for time in v.times:
        verifier_vars.append(time.W)  # Total waste
        verifier_vars.append(time.time)  # This time
        for flow in time.flows:
            verifier_vars.append(flow.S)

    for time in v.times[:init_tsteps]:
        for flow in time.flows:
            # Arrival and loss
            verifier_vars.append(flow.A)
            verifier_vars.append(flow.L)
            verifier_vars.append(flow.Ld)
            verifier_vars.append(flow.rtt)

            verifier_vars.append(flow.cwnd)
            verifier_vars.append(flow.rate)

    definition_vars = []
    for time in v.times:
        # Totals
        definition_vars.append(time.A)
        definition_vars.append(time.S)
        definition_vars.append(time.L)

    for time in v.times[init_tsteps:]:
        for flow in time.flows:
            # Arrival and loss
            definition_vars.append(flow.A)
            definition_vars.append(flow.L)
            definition_vars.append(flow.Ld)
            definition_vars.append(flow.rtt)

            definition_vars.append(flow.cwnd)
            definition_vars.append(flow.rate)

    return verifier_vars, definition_vars


def setup_definitions(c: Config, v: ModelVariables) -> MySolver:
    s = MySolver()
    s.warn_undeclared = False

    # This function should be called before using any Piecewise operations.
    # TODO: enforce above.

    # All constraints for aux vars do into definitions
    for x in v.delta_t:
        if(isinstance(x, Piecewise)):
            x.s = s

    total(c, s, v)  # Def only. Constr not needed.
    initial(c, s, v)  # Verifier only. CCAC2 domain basically. Either env or def.
    loss(c, s, v)  # Def to compute loss (with finite or inf buffer).

    return s


def setup_environment(c: Config, v: ModelVariables) -> MySolver:
    s = MySolver()
    s.warn_undeclared = False

    monotone(c, s, v)  # Arrival is def only. But that constr is not needed.
    service_waste(c, s, v)  # what all service/waste curves are allowed

    return s