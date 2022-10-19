import os
import pickle
import networkx as nx
import pandas as pd
from ccmatic.main_cca_assumption_incal import get_solution_str


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Analyse assumptions')
    parser.add_argument('--solution-log-path', type=str,
                        action='store', required=True)
    parser.add_argument('--graph-path', type=str,
                        action='store', required=True)
    parser.add_argument('-o', '--out-dir', type=str,
                        action='store', required=True)
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


def write_assumptions(assumption_ids, filename):
    f = open(os.path.join(args.out_dir, filename), 'w')
    for i, ia in enumerate(assumption_ids):
        assumption_str = get_solution_str(
            assumption_records.iloc[ia], None, None)
        f.write(f"{ia}, {i}.\n{assumption_str}\n")
    f.close()


# Print assumptions with outdegree 0
# These assumptions don't imply any other assumption
filtered_assumptions = [x for x, d in g.out_degree() if d == 0]
write_assumptions(filtered_assumptions, "filtered.txt")

# For reference, also write all assumptions
write_assumptions(range(len(assumption_records)), "all_assumptions.txt")
# import ipdb; ipdb.set_trace()
