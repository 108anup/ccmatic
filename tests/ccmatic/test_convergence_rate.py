import scipy
import math
import matplotlib.pyplot as plt

ALPHA = 5
MAX_LINK_RATE = 1000
STEP = 10
START = 10


def get_growth(f):
    min_c = [START]
    trange = list(range(MAX_LINK_RATE))
    for t in trange:
        min_c.append(min_c[-1] + f(min_c[-1]))

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
    fig, ax = plt.subplots()
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Convergence time")
    for name, f in fdict.items():
        print(name)
        convergence_time_func = get_growth(f)
        bw_range = range(START, MAX_LINK_RATE, STEP)
        ctime = [convergence_time_func(bw) for bw in bw_range]
        ax.plot(bw_range, ctime, label=name)
    ax.legend()
    ax.grid()
    fig.set_tight_layout(True)
    fig.savefig("tmp/convergence.pdf", bbox_inches='tight')
