import scipy
import math
import matplotlib.pyplot as plt
from plot_config.figure_type_creator import FigureTypeCreator as FTC

doc = FTC(paper_use_small_font=True).get_figure_type()

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

        'log': math.log,
        'sqrt': math.sqrt,
        'const': lambda x: ALPHA,
        'linear': lambda x: x,
    }
    # fig, ax = plt.subplots()
    fig, ax = doc.subfigures()
    ax.set_xlabel("C (Bandwidth) [same unit as $\epsilon$]")
    ax.set_ylabel("$F^{-1}(C)$ (Convergence time) [RTTs]")
    for name, f in fdict.items():
        print(name)
        convergence_time_func = get_growth(f)
        bw_range = range(START, MAX_LINK_RATE, STEP)
        ctime = [convergence_time_func(bw) for bw in bw_range]
        ax.plot(bw_range, ctime, label=name)
    ax.legend(title="$f(C)$")
    ax.grid(True)
    # fig.set_tight_layout(True)
    fig.savefig("tmp/convergence.pdf", bbox_inches='tight', pad_inches=0.01)
