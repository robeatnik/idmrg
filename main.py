import os
from idmrg import *
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import matplotlib
label_size = 13
FLAGS = None

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

# mpl.rcParams['xtick.labelsize'] = label_size
# mpl.rcParams['ytick.labelsize'] = label_size
# mpl.rcParams['axes.labelsize'] = label_size
# font = {'family': 'normal', 'weight': 'normal', 'size': label_size}

# matplotlib.rc('font', **font)


# get data
def N_(N, chi, i_theta, n_theta):
    """save the GS energies of each step to see if it is converged"""
    moreN = get_ES(
        chi=chi, n_chi=0, N=N, i_theta=i_theta, n_theta=n_theta, ES=False)
    return (None)


def chi_(chi, n_chi, N, i_theta, n_theta):
    """save the GS energies of different chi to get a most stable case."""
    moreN = get_ES(
        N=N, chi=chi, n_chi=n_chi, i_theta=i_theta, n_theta=n_theta, ES=False)
    return (None)


def theta_(chi, N, n_theta):
    """ output in n_theta sample cases how many ones didn't converge?
    use 60 different chi and N for 60 different case of theta is not practical?
    """
    moreN = 0
    es = []
    for i in range(n_theta):
        print(20 * '*')
        print(20 * '*')
        print('theta=2pi*{}/{}'.format(i, n_theta))
        moreN0, es0 = get_ES(
            N=N, chi=chi, i_theta=i, n_theta=n_theta, n_chi=0, ES=True)
        moreN += moreN0
        es.append(es0)
    print('moreN =', moreN0)
    np.save('./ES/entanglement_spectrum_with_theta_{}.npy'.format(n_theta), es)
    return (None)


# draw the graph
def vs_chi(chi, n_chi, N, i_theta, n_theta, data=False):
    filename = './chi/energy_with_different_chi{}(N{}_theta{}_{}).npy'.format(
        chi + n_chi * 5, N, n_theta, i_theta)
    if data or not os.path.isfile(filename):
        chi_(chi, n_chi, N, i_theta, n_theta)
    # drawing the GS energy VS chi
    energy_with_dffrt_chi = np.load(filename)
    energy_with_dffrt_chi = -energy_with_dffrt_chi
    chi = np.arange(chi, chi + 5 * n_chi + 1)
    chi = chi[::5]
    plt.title(
        "GS energy derived with different $\chi$ in step {}".format(N), y=1.1)
    plt.xticks(chi)
    # plt.yscale("log")
    # plt.ylim(0,)
    plt.ylabel("GS energy")
    plt.xlabel("$\chi$")
    plt.plot(chi, energy_with_dffrt_chi)

    plt.tight_layout()
    plt.savefig("eVSchilog.pdf")
    return (None)


def vs_step(N, chi, i_theta, n_theta, data=False):
    filename = './step/energy_in_every_step(chi{}_theta{}_{}).npy'.format(
        N, chi, n_theta, i_theta)
    if data or not os.path.isfile(filename):
        N_(N, chi, i_theta, n_theta)

    # # drawing the GS energy VS number of steps
    # the case of N>100
    energy_in_every_step = np.load(filename)
    n_step = 20
    step = np.arange(1, 1 + n_step * len(energy_in_every_step), n_step)
    plt.title("GS energy derived in different step with chi 2")
    plt.xticks(step[::5])
    plt.yscale("log")
    plt.ylabel("$|E_\mathrm{idmrg}-E_\mathrm{exact}|$")
    plt.yticks([0, 1e-17, 1e-16, 1e-15, 1e-13, 1e-7, 1e-1])
    plt.xlabel("step")
    plt.plot(step, energy_in_every_step)

    plt.tight_layout()
    plt.savefig("eVSsteps.pdf")
    return (None)


def draw_ES(chi, N, n_theta, data):
    filename = './ES/entanglement_spectrum_with_theta={}(chi={}_N={}).npy'.format(n_theta,chi,N)

    if data or not os.path.isfile(filename):
        theta_(chi=chi, N=N, n_theta=n_theta)

    es = np.load(filename)

    fig, ax = plt.subplots(1)
    props = dict(boxstyle='square', alpha=0.5, facecolor='white')

    for i, v in enumerate(es):
        site = (i + 1) * np.ones([len(v)])
        plt.errorbar(site, v, xerr=0.5, fmt=' ', color='blue')

    xticks = [0, 0.5, 1, 1.5, 2]
    xticks = np.array(xticks)
    plt.xticks(n_theta/xticks[-1] * xticks, xticks)

    ymax = 12
    plt.axvline(x=0.25 * n_theta/xticks[-1] , ymax=ymax)
    plt.axvline(x=0.50 * n_theta/xticks[-1] , ymax=ymax)
    plt.axvline(x=1.25 * n_theta/xticks[-1] , ymax=ymax)
    plt.axvline(x=1.75 * n_theta/xticks[-1] , ymax=ymax)

    plt.xlim(0, 2)
    plt.ylim(0,ymax)

    textstr = '$N = %d$\n$chi=%d$' % (N, chi)
    plt.text(
        0.5 * n_theta,
        0.95 * ymax,
        textstr,
        fontsize=14,
        verticalalignment='top',
        bbox=props)
    # plt.text(1*n_theta,30,textstr)
    plt.xlabel(r'$\theta(\pi)$')
    plt.ylabel('Entanglement Energy')
    plt.tight_layout()
    plt.savefig('es_n_theta{}.pdf'.format(n_theta))
    return (None)


def main():
    data = FLAGS.data
    chi = FLAGS.chi
    n_chi = FLAGS.n_chi
    N = FLAGS.N
    n_theta = FLAGS.n_theta
    i_theta = FLAGS.i_theta

    # when uncommenting this, you should give a nonzero at the same time, or error will occur.
    # vs_chi(chi, n_chi, N, i_theta, n_theta, data)

    # vs_step(N, chi, i_theta, n_theta, data)
    draw_ES(chi, N, n_theta, data)
    return (None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', type=bool, default=False, help='whether to compute the data')
    parser.add_argument('--chi', type=int, default=2, help='bond dimension')
    parser.add_argument('--J', type=float, default=1., help='J')
    parser.add_argument('--n_chi', type=int, default=0, help='n_chi')
    parser.add_argument(
        '--n_theta',
        type=int,
        default=2,
        help='number of theta sampled in a 2pi')
    parser.add_argument(
        '--i_theta',
        type=int,
        default=0,
        help='number of units of 2pi/n_theta in theta')
    parser.add_argument('--N', type=int, default=15, help='number of steps')
    FLAGS = parser.parse_args()
    main()
