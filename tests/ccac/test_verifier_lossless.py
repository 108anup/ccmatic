import pandas as pd
import numpy as np
import logging

import pandas as pd
import z3
from ccmatic.cegis import CegisConfig
from ccmatic.verifier import (get_cex_df, get_periodic_constraints,
                              setup_cegis_basic)
from ccmatic import get_desired_necessary
from ccmatic.verifier.assumptions import get_cca_definition
from cegis.util import Metric, optimize_multi_var
from pyz3_utils.common import GlobalConfig
from pyz3_utils.my_solver import MySolver

from ccac.variables import VariableNames

cc = CegisConfig()
cc.history = 4
# cc.history = cc.R + cc.D
# cc.T = 15
# cc.T = 8 + cc.history * 2
# cc.compose = False
cc.infinite_buffer = True

cc.desired_util_f = z3.Real('desired_util_f')
cc.desired_queue_bound_multiplier = z3.Real('desired_queue_bound_multiplier')
cc.desired_queue_bound_alpha = z3.Real('desired_queue_bound_alpha')
cc.desired_loss_count_bound = z3.Real('desired_loss_count_bound')
cc.desired_loss_amount_bound_multiplier = z3.Real('desired_loss_amount_bound')
cc.desired_loss_amount_bound_alpha = z3.Real('desired_loss_amount_alpha')

# cc.template_qdel = True
# cc.template_loss_oracle = False

cc.cca = 'paced'
if(cc.cca == 'copa'):
    cc.template_qdel = True
if(cc.cca == 'aimd'):
    cc.template_loss_oracle = False

(c, s, v,
 ccac_domain, ccac_definitions, environment,
 verifier_vars, definition_vars) = setup_cegis_basic(cc)
if(cc.cca != 'paced'):
    cca_definitions = get_cca_definition(c, v)

d = get_desired_necessary(cc, c, v)
desired = d.desired_necessary
# desired = d.desired_in_ss

vn = VariableNames(v)
first = cc.history  # First cwnd idx decided by synthesized cca


def get_counter_example_str(counter_example: z3.ModelRef) -> str:
    df = get_cex_df(counter_example, v, vn, c)
    desired_string = d.to_string(cc, c, counter_example)
    ret = "{}\n{}.".format(df, desired_string)
    return ret


def check_solution(solution: pd.Series):
    template_definitions = []
    for t in range(first, c.T):

        # rhs = v.S_f[0][t-c.R] - v.S_f[0][t-cc.history]

        # rhs = v.S_f[0][t-c.R] - v.S_f[0][t-2] + 3/2
        # # rhs = 3/2*v.S_f[0][t-c.R] - 3/2*v.S_f[0][t-2] + 2
        # # rhs = 3/2*v.S_f[0][t-c.R] - v.S_f[0][t-2] - 1/2*v.S_f[0][t-4]
        # rhs = 3/2*v.S_f[0][t-c.R] - v.S_f[0][t-2] - 1/2*v.S_f[0][t-3] + 2
        # rhs = 3/2*v.S_f[0][t-c.R] - 1/2*v.S_f[0][t-2] - v.S_f[0][t-3]
        # rhs = v.S_f[0][t-c.R] - v.S_f[0][t-4]
        # rhs = 1/2*v.S_f[0][t-c.R] + 1/2*v.S_f[0][t-2] + 1*v.S_f[0][t-3] - 2*v.S_f[0][t-4] +1/2
        if(solution is None):
            # rhs = v.S_f[0][t-c.R] + v.S_f[0][t-2] - v.S_f[0][t-3] - v.S_f[0][t-4]
            rhs = v.S_f[0][t-c.R] - v.S_f[0][t-3] + v.alpha

        else:
            rhs = v.alpha * solution['Gen__const_c_f']
            for shift in range(cc.history):
                rhs += (solution[f'Gen__coeff_c_f_S_f_{shift}']
                        * v.S_f[0][t-c.R-shift])

        # rhs = v.S_f[0][t-1] + v.S_f[0][t-2] - (v.S_f[0][t-3] + v.S_f[0][t-4])
        assert isinstance(rhs, z3.ArithRef)
        template_definitions.append(
            v.c_f[0][t] == z3.If(rhs >= v.alpha,
                                 rhs, v.alpha))

        # template_definitions.append(v.c_f[0][t] == 4096)
        # template_definitions.append(v.c_f[0][t] == 0.01)

    verifier = MySolver()
    verifier.warn_undeclared = False
    verifier.add(ccac_domain)
    verifier.add(ccac_definitions)
    verifier.add(environment)
    if(c.cca == 'paced'):
        verifier.add(z3.And(*template_definitions))
    else:
        verifier.add(cca_definitions)
    verifier.add(z3.Not(desired))

    # # Periodic cex
    # for h in range(cc.history):
    #     last = cc.T-1 - (cc.history-1)
    #     for n in range(cc.N):
    #         verifier.add(v.c_f[n][h] == v.c_f[n][last+h])
    #     verifier.add(v.A[h] - v.L[h] - v.S[h] ==
    #                  v.A[last+h] - v.L[last+h] - v.S[last+h])
    # verifier.add(get_periodic_constraints(cc, c, v))

    verifier.add(cc.desired_loss_count_bound == 0)
    verifier.add(cc.desired_loss_amount_bound_multiplier == 0)
    verifier.add(cc.desired_loss_amount_bound_alpha == 0)

    optimization_list = [
        Metric(cc.desired_util_f, 0.1, 1, 0.001, True),
        Metric(cc.desired_queue_bound_multiplier, 0, 100, 0.001, False),
        Metric(cc.desired_queue_bound_alpha, 0, 100, 0.001, False),
    ]

    verifier.push()
    for metric in optimization_list:
        if(metric.maximize):
            verifier.add(metric.z3ExprRef == metric.lo)
        else:
            verifier.add(metric.z3ExprRef == metric.hi)

    sat = verifier.check()

    # f = open('tmp/test_verifier_oracle.txt', 'w')
    # f.write(verifier.assertions().sexpr())
    # f.close()

    if(str(sat) == "sat"):
        model = verifier.model()
        print(get_counter_example_str(model))

    # import ipdb; ipdb.set_trace()

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

    # import ipdb; ipdb.set_trace()

    verifier.pop()

    GlobalConfig().logging_levels['cegis'] = logging.INFO
    logger = logging.getLogger('cegis')
    GlobalConfig().default_logger_setup(logger)

    ret = optimize_multi_var(verifier, optimization_list, quick=True)
    df = pd.DataFrame(ret)
    sort_columns = [x.name() for x in optimization_list]
    sort_order = [x.maximize for x in optimization_list]
    df = df.sort_values(by=sort_columns, ascending=sort_order)

    ret = {}
    for x in optimization_list:
        col = x.name()
        ret[col] = df[col].max() if x.maximize else df[col].min()
    # print(ret)
    return ret


# import sys
# check_solution(None)
# sys.exit(0)

all_records = []

ret = check_solution(None)
ret['name'] = 'RoCC'
all_records.append(ret)

all_records.append({
    'name': 'Copa',
    f'{cc.desired_util_f.decl().name()}': 0,
    f'{cc.desired_queue_bound_multiplier.decl().name()}': 3.5,
    f'{cc.desired_queue_bound_alpha.decl().name()}': np.NaN})

all_records.append({
    'name': 'BBR',
    f'{cc.desired_util_f.decl().name()}': 0,
    f'{cc.desired_queue_bound_multiplier.decl().name()}': 2,
    f'{cc.desired_queue_bound_alpha.decl().name()}': np.NaN})

all_records.append({
    'name': 'AIMD',
    f'{cc.desired_util_f.decl().name()}': 1,
    f'{cc.desired_queue_bound_multiplier.decl().name()}': np.inf,
    f'{cc.desired_queue_bound_alpha.decl().name()}': np.NaN})

fpath = 'tmp/hotnets-composeTrue.csv'
f = open(fpath, 'r')
df = pd.read_csv(f)
solutions = df.loc[:, ~df.columns.str.contains('^Unnamed')]
f.close()

for i, solution in solutions.iterrows():
    # if(i != 9):
    #     continue

    solution_str = "c_f[n][t] = "

    rhs = [f"{solution['Gen__const_c_f']}alpha"]
    for shift in range(cc.history):
        rhs.append(f"{solution[f'Gen__coeff_c_f_S_f_{shift}']}S_f[n][t-{shift+1}]")

    solution_str += " + ".join(rhs)
    # print(solution_str)

    ret = check_solution(solution)

    ret['name'] = solution_str
    print(ret)
    # if(ret[f'{cc.desired_util_f}'] < 0.5):
    #     import ipdb; ipdb.set_trace()
    all_records.append(ret)

df = pd.DataFrame(all_records)
df.to_csv('tmp/hotnets-composeTrue-pareto.csv', header=True)
# import ipdb; ipdb.set_trace()
