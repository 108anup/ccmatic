import os
import pickle
import networkx as nx
import pandas as pd
from ccmatic.generator.analyse_assumptions import build_adj_matrix, get_graph, parse_and_create_assumptions, write_draw_graph
from ccmatic.main_cca_assumption_incal import get_solution_str


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Analyse assumptions')
    parser.add_argument('--solution-log-path', type=str,
                        action='store', required=True)
    parser.add_argument('--graph-path', type=str,
                        action='store', required=True)
    parser.add_argument('-o', '--out-dir', type=str,
                        action='store')
    parser.add_argument('--compare-known',
                        action='store_true', default=False)
    args = parser.parse_args()
    return args


args = get_args()

# Read assumptions
f = open(args.solution_log_path, 'r')
df = pd.read_csv(f)
assumption_records = df.loc[:, ~df.columns.str.contains('^Unnamed')]
f.close()

# Read graph
f = open(args.graph_path, 'rb')
g = pickle.load(f)
f.close()

filtered_assumptions = [x for x, d in g.out_degree() if d == 0]

if(args.out_dir):

    def write_assumptions(assumption_ids, filename):
        f = open(os.path.join(args.out_dir, filename), 'w')
        for i, ia in enumerate(assumption_ids):
            assumption_str = get_solution_str(
                assumption_records.iloc[ia], None, None)
            f.write(f"{ia}, {i}.\n{assumption_str}\n")
        f.close()

    # Print assumptions with outdegree 0
    # These assumptions don't imply any other assumption

    write_assumptions(filtered_assumptions, "filtered.txt")

    # For reference, also write all assumptions
    write_assumptions(range(len(assumption_records)), "all_assumptions.txt")
    # import ipdb; ipdb.set_trace()

if(args.compare_known):
    # Check assumptions against known assumption (from CCAC paper)
    # Known assumption
    known_assumption = assumption_records.iloc[0].copy()
    for col in known_assumption.index:
        if('coeff' in col or 'const' in col):
            known_assumption[col] = 0
        else:
            known_assumption[col] = False

    from ccmatic.main_cca_assumption_incal import coeffs, vname2vnum, clauses
    # Ineq 0: A[t] (- L[t]) - S[t] <= 0
    # Ineq 1: W[t] - W[t-1] <= 0
    # Clause 0: 1 or 0

    # Ineq 1:
    known_assumption[coeffs[0][vname2vnum['W']][0].decl().name()] = 1
    known_assumption[coeffs[0][vname2vnum['W']][1].decl().name()] = -1

    # Ineq 0:
    known_assumption[coeffs[1][vname2vnum['A']][0].decl().name()] = 1
    # known_assumption[coeffs[0][vname2vnum['L']][1].decl().name()] = -1
    known_assumption[coeffs[1][vname2vnum['S']][0].decl().name()] = -1

    # Clause 0:
    known_assumption[clauses[0][0].decl().name()] = True
    known_assumption[clauses[0][1].decl().name()] = True


    from ccmatic.main_cca_assumption_incal import lemmas, assumption

    filtered_df = assumption_records.iloc[filtered_assumptions].copy()
    all_assumption_records = filtered_df.append(
        known_assumption, ignore_index=True)
    assumption_assignments, assumption_expressions = \
        parse_and_create_assumptions(all_assumption_records, assumption)
    if(args.out_dir):
        all_assumption_records.to_csv(
            os.path.join(args.out_dir, "seed_assumptions.csv"), header=True)

    adj = build_adj_matrix(
        assumption_assignments, assumption_expressions, lemmas)

    g = get_graph(list(range(len(all_assumption_records))), adj)

    new_to_old = {i: x for i, x in enumerate(filtered_assumptions)}
    new_to_old[len(filtered_assumptions)] = len(assumption_records)
    g = nx.relabel_nodes(g, new_to_old)

    write_draw_graph(g, args.out_dir, "-compare_known")

    import ipdb; ipdb.set_trace()
