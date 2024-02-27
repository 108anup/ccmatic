import scipy
import math
import matplotlib.pyplot as plt
from plot_config.figure_type_creator import FigureTypeCreator as FTC

doc = FTC(paper_use_small_font=True).get_figure_type()
ppt = FTC(pub_type='presentation').get_figure_type()

ALPHA = 3
MAX_LINK_RATE = 200
STEP = 10
START = 5
BBUFFER_SEC = 1


def get_growth(f):
    min_c = [START]
    trange = list(range(MAX_LINK_RATE))
    for t in trange:
        min_c.append(min_c[-1] + f(min_c[-1])/BBUFFER_SEC)

    # print(min_c)

    time_to_convergence = scipy.interpolate.interp1d(min_c[:-1], trange)
    # for minc in range(10, 1000, 10):
    #     print(minc, time_to_convergence(minc))
    return time_to_convergence


if(__name__ == "__main__"):
    fdict = {
        # 'log': lambda x: max(ALPHA, math.log(x)) if x > 0 else ALPHA,
        # 'sqrt': lambda x: max(ALPHA, math.sqrt(x)),
        # 'const': lambda x: ALPHA,
        # 'linear': lambda x: max(ALPHA, x),

        # '$\\log(\\mathit{C})$': math.log,
        # '$\\sqrt{\\mathit{C}}$': math.sqrt,
        # 'const': lambda x: ALPHA,
        # '$\\mathit{C}$': lambda x: x,

        '$O(1)$': lambda x: ALPHA,
        '$O(\\log\\mathit{C})$': math.log,
        '$O(\\sqrt{\\mathit{C}})$': math.sqrt,
        '$O(\\mathit{C})$': lambda x: x,
    }

    flist = [
        ('$O(1)$', lambda x: ALPHA, 1),
        ('$O(\\log\\mathit{C})$', math.log, 1),
        ('$O(\\sqrt{\\mathit{C}})$', math.sqrt, 1),
        ('$O(\\mathit{C})$', lambda x: x, 1),
    ]

    # fig, ax = plt.subplots()
    # fig, ax = doc.subfigures(yscale=0.7)
    fig, ax = ppt.subfigures(yscale=0.7, xscale=0.5)
    # ax.set_xlabel("Bandwidth $\\mathit{C}$ [same unit as $\epsilon$]")
    # ax.set_ylabel("Convergence time\n$F^{-1}(\\mathit{C})$ [$\\mathit{RTT}$s]")
    ax.set_xlabel("Bandwidth $\\mathit{C}$")
    ax.set_ylabel("Convergence time [$\\mathit{RTT}$s]")
    for name, f, alpha in flist: # fdict.items():
        print(name)
        convergence_time_func = get_growth(f)
        bw_range = range(START, MAX_LINK_RATE, STEP)
        ctime = [convergence_time_func(bw) for bw in bw_range]
        ax.plot(bw_range, ctime, label=name, alpha=alpha)
    # legend = ax.legend(title="$f(\\mathit{C})$") #, borderpad=0.2)
    # legend = ax.legend(title="Loss") #, borderpad=0.2)
    # frame = legend.get_frame()
    # frame.set_linewidth(0.5)
    # frame.set_edgecolor('black')
    # legend.set_frame_on(False)
    ax.grid(True)
    # fig.set_tight_layout(True)
    fig.savefig("convergence.svg", bbox_inches='tight', pad_inches=0.01)
