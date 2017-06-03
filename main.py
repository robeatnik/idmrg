import os
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
import matplotlib
label_size = 13
FLAGS = None

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# mpl.rcParams['xtick.labelsize'] = label_size
# mpl.rcParams['ytick.labelsize'] = label_size
# mpl.rcParams['axes.labelsize'] = label_size
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : label_size}

matplotlib.rc('font', **font)

def vs_chi():
    # drawing the GS energy VS chi
    exact = -2/3*np.cos(np.arctan(1/3))
    energy_with_dffrt_chi = np.load("./chi/energy_with_different_chi_in_step_15.npy")
    energy_with_dffrt_chi = - energy_with_dffrt_chi
    chi = np.arange(1,1+5*len(energy_with_dffrt_chi))
    chi = chi[::5]
    plt.title("GS energy derived with different $\chi$ in step 15",y=1.1)
    plt.xticks(chi[::5])
    plt.yscale("log")
    # plt.ylim(0,)
    plt.ylabel("GS energy")
    plt.xlabel("$\chi$")
    plt.plot(chi,energy_with_dffrt_chi)
    plt.savefig("eVSchilog.pdf")


def vs_step():
    # drawing the GS energy VS # of steps
    # the case of N>100
    exact = -2/3*np.cos(np.arctan(1/3))
    energy_in_every_step = np.load("./step/energy_in_every_step_with_chi_2.npy")
    energy_in_every_step = np.abs(energy_in_every_step-exact*np.ones([len(energy_in_every_step)]))
    n_step = 20
    step = np.arange(1,1+n_step*len(energy_in_every_step),n_step)
    plt.title("GS energy derived in different step with chi 2")
    plt.xticks(step[::5])
    plt.yscale("log")
    plt.ylabel("$|E_\mathrm{idmrg}-E_\mathrm{exact}|$")
    plt.yticks([0,1e-17,1e-16,1e-15,1e-13,1e-7,1e-1])
    plt.xlabel("step")
    plt.plot(step,energy_in_every_step)

    plt.tight_layout()
    plt.savefig("eVSsteps.pdf")

def draw_ES(n_theta):
    # drawing the energy levels of the exact results of finite system size
    es = []
    for i in range(n_theta):
        theta = 2.*np.pi/n_theta*i
        es.append(np.load("./ES/entanglement_spectrum_with_theta_{}_{}.npy".format(n_theta,i)))
    plt.figure()
    for i,v in enumerate(es):
        site = (i+1)*2*np.ones([len(v)])
        energy = v
        # plt.scatter(site,energy,marker='_')
        plt.errorbar(site, energy, xerr=0.7, fmt=' ',color='blue')
    xticks = [0,0.5,1,1.5,2]
    xticks = np.array(xticks)
    plt.ylim(0,28)
    plt.xlim(0,2)
    plt.xticks(n_theta*xticks, xticks)
    plt.xlabel(r'$\theta(\pi)$')
    plt.ylabel('Entanglement Energy')
    plt.tight_layout()
    plt.savefig('es_n_theta{}.pdf'.format(n_theta))
    return(None)

def N_steps(chi,theta):
    for N in [15,25,35,45,55]:
        os.system('python3 idmrg.py --N={} --n_chi=5'.format(N))
    return(None)

def chi_(N,theta):
    for chi in [10,20,30,40,50]:
        os.system('python3 idmrg.py --chi={}'.format(chi))
    return(None)


def theta_(chi,N,n_theta):
    """
    use 60 different chi and N for 60 different case of theta is not practical?
    """
    for i in range(n_theta):
        print(20*'*')
        print(20*'*')
        print('theta=2pi*{}/{}'.format(i,n_theta))
        os.system('python3 idmrg.py --i_theta={} --n_theta={} --chi=10'.format(i,n_theta))
    return(None)


def main():
    data = FLAGS.data
    chi = 10
    N = 20
    n_theta=FLAGS.n_theta
    if data or not os.path.isfile('./ES/entanglement_spectrum_with_theta_{}_{}.npy'
                                  .format(n_theta,n_theta-1)):
        theta_(chi=chi,N=N,n_theta=n_theta)
    draw_ES(n_theta)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_theta', type=int,default=2,
                        help='number of theta sampled in a 2pi')
    parser.add_argument('--data',type=bool,default=False,
                        help='whether to compute the data')
    FLAGS = parser.parse_args()
    main()

