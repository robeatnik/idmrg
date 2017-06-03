import numpy as np
from numpy import kron
import argparse
import pandas as pd
FLAGS = None

# spin 1/2
sx = 1/2.*np.array([[0,1],[1,0]])
sy = 1/2.*np.array([[0,-1.0j],[1.0j,0]])
sz = 1/2.*np.array([[1,0],[0,-1]])
# spin 1
SX = 1./np.sqrt(2)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
SY = 1./np.sqrt(2)*np.array([[0.,-1.0j,0.],[1.0j,0.,-1.0j],[0.,1.0j,0.]])
SZ = np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]])

def degenerate_(es, k=0):
    es = np.sort(es)
    ES = pd.Series([1],index=[es[0]])
    j = 0;k = k
    for i in es[1:]:
        if np.abs(i-es[j]) <= k:
            ES[es[j]] += 1
        else:
            ES = ES.append(pd.Series([1],index=[i]))
            j += ES[es[j]]
    df = pd.DataFrame({'Energy Spectrum': ES.index,
                       'Degeneracy': ES.data},
                       columns=['Energy Spectrum', 'Degeneracy'])
    return(df)

def kron_(n,spin_dim):
    h = 1
    for i in range(n):
        h = kron(h,np.eye(spin_dim))
    return(h)

def spin_list_position(i, n_site, spin_dim):
    """
    i: the posion of the 1st spin, begin with 0
    """
    if spin_dim == 2:
        spin = [sx, sy, sz]
    elif spin_dim == 3:
        spin = [SX, SY, SZ]
    else:
        print("Wrong spin dimension")
        return(None)

    H = np.zeros([spin_dim**n_site, spin_dim**n_site])
    if i == n_site-1:
        for s in spin:
            h = kron(s,kron_(n_site-2,spin_dim))
            h = kron(h,s)
            np.add(H, h, out=H, casting="unsafe")
    else:
        for s in spin:
            h = kron(kron_(i, spin_dim),s)
            h = kron(h,s)
            h = kron(h,kron_(n_site-1-(i+1),spin_dim))
            np.add(H, h, out=H, casting="unsafe")
    return(H)

def linear_(n_site, spin_dim):
    """
    with periodic boundary condition, now the # of bond is the same as the # of sites.
    """
    H = np.zeros([spin_dim**n_site, spin_dim**n_site])
    for i in range(n_site):
        h = spin_list_position(i, n_site, spin_dim)
        np.add(H, h, out=H, casting="unsafe")
    return(H)

def quandratic_(n_site, spin_dim):
    H = np.zeros([spin_dim**n_site, spin_dim**n_site])
    for i in range(n_site):
        h = spin_list_position(i, n_site, spin_dim)
        h = np.tensordot(h,h,axes=1)
        np.add(H, h, out=H, casting="unsafe")
    return(H)

def heisenberg_(n_site, spin_dim, J):
    H = J*linear_(n_site, spin_dim)
    return(H)

def blb_(n_site, spin_dim, Jl, Jq):
    H = Jl*linear_(n_site, spin_dim)+Jq*quandratic_(n_site, spin_dim)
    return(H)

def main():
    n_site = FLAGS.spin_number
    J = FLAGS.J
    spin_dim = FLAGS.spin_dim
    theta = FLAGS.theta*np.pi
    Jl = np.cos(theta)
    Jq = np.sin(theta)
    H = blb_(n_site, spin_dim, Jl, Jq)
    # H = heisenberg_(n_site, spin_dim, J)
    w, v = np.linalg.eigh(H)
    np.save("n_site_{}".format(n_site),w/n_site)
    print('GS energy density:\n', w[0]/n_site)
    print("GS wavefunction:",v[0])
    # df = degenerate_(w/n_site,0.1)
    # print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
               description='Bilinear-Biquadratic model: Entanglement spectrum.',
               epilog='End of description.')
    parser.add_argument('--spin_number', type=int, default=2,
                         help='number of spins(n_site)(default 2)')
    parser.add_argument('--spin_number_A', type=int, default=3,
                         help='number of spins in subsystem A (default 3)')
    parser.add_argument('--J_bilinear', type=float, default=3.0,
                         help='bilinear coupling strength (default 3.0)')
    parser.add_argument('--J_biquadratic', type=float, default=1.0,
                         help='biquadratic coupling strength (default 1.0)')

    parser.add_argument('--J', type=float, default=1.0,
                         help='J of heisenberg model(default 1.0)')
    parser.add_argument('--spin_dim', type=int, default=3,
                         help='spin dimension(default 3)')
    parser.add_argument('--theta', type=float, default=np.arctan(1/3)/np.pi,
                         help='theta in unit of pi')
    FLAGS = parser.parse_args()
    main()
