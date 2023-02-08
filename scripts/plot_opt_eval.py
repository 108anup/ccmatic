import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

indir = sys.argv[1]

files = []
for file in os.listdir(indir):
    if(file.endswith('.csv')):
        files.append(file)


def get_file_tag(fname):
    # fname=main_cca_belief_template.py-infinite_buffer=False-finite_buffer=False-dynamic_buffer=True-opt_cegis=True-opt_ve=False-opt_pdt=False-opt_wce=False-opt_feasible=False-opt_ideal=False.csv
    # import ipdb; ipdb.set_trace()
    pdict = {}
    for pv in fname.replace('.csv', '').split('-'):
        param = pv.split('=')[0]
        value = pv.split('=')[1]
        try:
            value = eval(value)
        except:
            pass
        pdict[param] = value

    # opt_ve=False-opt_pdt=False-opt_wce=False-opt_feasible=False-opt_ideal=False
    tag_dict = {
        'opt_ve': 'VE',
        'opt_pdt': 'REL',
        'opt_wce': 'WCE',
        'opt_feasible': 'FR',
        'opt_ideal': 'SV'
    }

    # print(fname)
    # print(pdict)

    if(pdict['opt_cegis'] is False):
        return "Q"

    tag_list = []
    for p, t in tag_dict.items():
        if(pdict[p]):
            tag_list.append(t)
    if(len(tag_list) == 0):
        return "CEGIS"
    return "+" + "+".join(tag_list)


def get_metrics(fname):
    df = pd.read_csv(os.path.join(indir, fname))
    summary = df.iloc[-1].copy()
    summary['total_time'] = df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]
    tag = get_file_tag(file)
    summary['tag'] = tag
    return summary


df_list = []
for file in files:
    sdf = get_metrics(file)
    df_list.append(sdf)

df = pd.DataFrame(df_list).reset_index().drop(columns=['index'])
print(df)
nsolutions = df['solutions'].max()
for i, row in df.iterrows():
    if(row['solutions'] < nsolutions):
        print(f"{row['tag']} did not finish.")
        df['total_time'].iloc[i] = 7 * 86400
df = df.sort_values(by=['total_time'])
print(df)

fig, ax = plt.subplots()
df.plot.bar(x='tag', y='total_time', ax=ax)
ax.get_legend().remove()
ax.set_yscale('log')
ax.set_ylabel('Time (s)')
ax.set_xlabel('Scheme')
ax.tick_params(axis='x', labelrotation=45)
ax.set_title(indir)
fig.set_tight_layout(True)
fig.savefig(os.path.join(indir, 'opt_eval.pdf'), bbox_inches='tight', pad_inches=0.01)
# import ipdb; ipdb.set_trace()

