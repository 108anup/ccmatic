import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

big_size = 24
small_size = 20

update_sizes = {
    'font.size': big_size,
    'axes.titlesize': big_size,
    'axes.labelsize': big_size,
    'legend.fontsize': small_size,
    'legend.title_fontsize': big_size,
    'xtick.labelsize': small_size,
    'ytick.labelsize': small_size,
    'pdf.fonttype': 42,
    'font.family': 'sans-serif',

    'xtick.major.width': 2,
    'ytick.major.width': 2,

    'xtick.minor.width': 1,
    'ytick.minor.width': 1,

    'axes.linewidth': 2,
    'lines.markersize': 20,

    "grid.linewidth": 0.5,
    "grid.linestyle": '--'
}
plt.rcParams.update(update_sizes)

fpath = 'tmp/hotnets-composeTrue-pareto.csv'
f = open(fpath, 'r')
df = pd.read_csv(f)
solutions = df.loc[:, ~df.columns.str.contains('^Unnamed')]
f.close()

markers = {
    'Copa': '^',
    'BBR': 's',
    'AIMD': 'd',
    'RoCC': 'o',
    'Synthesized': 'o'
}

fig, ax = plt.subplots()
xx = []
yy = []
for _, solution in solutions.iterrows():
    name = solution['name']
    x = 100*solution['desired_util_f']
    y = solution['desired_queue_bound_multiplier']
    if(y > 10):
        y = 10
    if(name in markers):
        pass
    else:
        if(x < 0.1):
            import ipdb; ipdb.set_trace()
        xx.append(x)
        yy.append(y)

ax.scatter(xx, yy, label='Synthesized', marker=markers['Synthesized'], alpha=0.5)

for _, solution in solutions.iterrows():
    name = solution['name']
    x = 100*solution['desired_util_f']
    y = solution['desired_queue_bound_multiplier']
    if(y > 10):
        y = 10
    if(name in markers):
        ret = ax.scatter(x, y, label=name, marker=markers[name])
        # import ipdb; ipdb.set_trace()
        if(name == 'Copa'):
            ax.text(x-7, y+1, name)
        else:
            ax.text(x-7, y-2, name)

ax.text(40, 4, 'Synthesized')

ax.set_xlim(-10, 110)
ax.set_ylim(-1, 11)
ax.grid()

ax.set_xlabel('Min Utilization (%)')
ax.set_ylabel('Max Queue Use (BDP)')

# legend = fig.legend(bbox_to_anchor=(0.5, 0.95), loc="lower center", ncol=3)
# # legend = fig.legend()
# legend.set_frame_on(False)

fig.set_tight_layout(True)

fig.canvas.draw()
labels = [item.get_text() for item in ax.get_yticklabels()]
labels[-1] = 'inf'
labels[-2] = '...'
ax.set_yticklabels(labels)
# import ipdb; ipdb.set_trace()

fig.savefig('tmp/scatter-composeTrue.png', dpi=300, bbox_inches='tight')