from typing import List
import z3
from ccac.config import ModelConfig
from ccac.model import make_solver
from ccac.utils import make_periodic
from ccac.variables import VariableNames
from ccmatic import CCmatic

from ccmatic.cegis import CegisConfig, VerifierType
from ccmatic.common import try_except
from ccmatic.verifier import get_cex_df, get_periodic_constraints, plot_cex
from ccmatic.verifier.assumptions import get_cca_definition, get_periodic_constraints_ccac
from cegis import Cegis
from cegis.util import z3_max
from pyz3_utils.my_solver import MySolver


def setup(buffer="infinite", buf_size=1, T=6, cca="none"):
    cc = CegisConfig()
    cc.verifier_type = VerifierType.cbrdelay
    cc.name = "cbrdelay"
    cc.synth_ss = False

    cc.buffer_size_multiplier = buf_size
    if (buffer == "finite"):
        cc.infinite_buffer = False
        cc.dynamic_buffer = False
    elif (buffer == "infinite"):
        cc.infinite_buffer = True
        cc.dynamic_buffer = False
    elif (buffer == "dynamic"):
        cc.infinite_buffer = False
        cc.dynamic_buffer = True
    else:
        assert False

    cc.template_qdel = True
    cc.template_queue_bound = False
    cc.template_fi_reset = False
    # cc.template_beliefs = True
    cc.N = 1
    cc.T = T
    cc.history = 2 * cc.R
    cc.cca = cca

    cc.desired_util_f = 0.5
    cc.desired_queue_bound_multiplier = 4
    cc.desired_queue_bound_alpha = 3
    if(cc.infinite_buffer):
        cc.desired_loss_count_bound = 0
        cc.desired_large_loss_count_bound = 0
        cc.desired_loss_amount_bound_multiplier = 0
        cc.desired_loss_amount_bound_alpha = 0
    else:
        cc.desired_loss_count_bound = 3
        cc.desired_large_loss_count_bound = 3
        cc.desired_loss_amount_bound_multiplier = 3
        cc.desired_loss_amount_bound_alpha = 3

    cc.feasible_response = False
    # cc.send_min_alpha = True

    link = CCmatic(cc)
    try_except(link.setup_config_vars)

    search_constraints = True
    template_definitions = []
    generator_vars = []

    def get_solution_str(
        solution: z3.ModelRef, generator_vars: List[z3.ExprRef], n_cex: int):
        return ""

    link.setup_cegis_loop(
        search_constraints,
        template_definitions, generator_vars, get_solution_str)
    # link.critical_generator_vars = critical_generator_vars
    print(f"{cc.name}: {cc.desire_tag()}")

    return cc, link


def test_cbrdelay_basic():
    cc, link = setup(cca="bbr", T=10)
    c, _, v = link.c, link.s, link.v

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(link.definitions)
    verifier.add(link.environment)
    # verifier.add(z3.Not(link.desired))
    cca_definitions = get_cca_definition(c, v)
    verifier.add(cca_definitions)
    # verifier.add(get_periodic_constraints_ccac(cc, c, v))

    # Periodic
    for t in range(cc.history):
        last = c.T+t-cc.history
        for n in range(c.N):
            verifier.add(v.c_f[n][t] == v.c_f[n][last])
            verifier.add(v.r_f[n][t] == v.r_f[n][last])
        verifier.add(v.A[t]-v.L[t]-v.S[t] == v.A[last]-v.L[last]-v.S[last])
        verifier.add(v.C0 + c.C * t - v.W[t] - (v.A[t] - v.L[t]) == v.C0 + c.C * last - v.W[last] - (v.A[last] - v.L[last]))

    verifier.add(v.L[0] == 0)
    verifier.add(v.S[-1] - v.S[0] < 0.1 * c.C * c.T)

    # assert c.N == 1
    # # Not cwnd limited
    # for t in range(c.R, c.T):
    #     verifier.add(v.S[t-c.R]+v.c_f[0][t] > v.A[t])

    assert c.N == 1
    max_rate = [0]
    for t in range(c.R, c.T):
        max_rate.append(z3_max(
            max_rate[-1], (v.S[t] - v.S[t-c.R])/c.R
        ))
    verifier.add(max_rate[c.R] == max_rate[-1])

    sat = verifier.check()
    print(sat)
    if(str(sat) == "sat"):
        model = verifier.model()
        vn = VariableNames(v)
        df = get_cex_df(model, v, vn, c)
        plot_cex(model, df, c, v, "tmp/cbrdelay_bbr.pdf")
        print(link.get_counter_example_str(model, link.verifier_vars))
        import ipdb; ipdb.set_trace()


def bbr_low_util(timeout=10):
    '''Finds an example trace where BBR has < 10% utilization. It can be made
    arbitrarily small, since BBR can get arbitrarily small throughput in our
    model.

    You can simplify the solution somewhat by setting simplify=True, but that
    can cause small numerical errors which makes the solution inconsistent. See
    README for details.

    '''
    c = ModelConfig.default()
    c.compose = True
    c.cca = "bbr"
    # Simplification isn't necessary, but makes the output a bit easier to
    # understand
    c.simplify = False
    s, v = make_solver(c)
    # Consider the no loss case for simplicity
    s.add(v.L[0] == 0)
    # Ask for < 10% utilization. Can be made arbitrarily small
    s.add(v.S[-1] - v.S[0] < 0.1 * c.C * c.T)
    make_periodic(c, s, v, 2 * c.R)
    sat = s.check()
    if(str(sat) == "sat"):
        model = s.model()
        vn = VariableNames(v)
        df = get_cex_df(model, v, vn, c)
        plot_cex(model, df, c, v, "tmp/cbrdelay_bbr.pdf")
        print(df)
        import ipdb; ipdb.set_trace()


if(__name__ == "__main__"):
    test_cbrdelay_basic()
    # bbr_low_util()