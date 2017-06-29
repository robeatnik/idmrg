import numpy as np
from numpy import kron
import pandas as pd

# spin 1/2
sx = 1 / 2. * np.array([[0, 1], [1, 0]])
sy = 1 / 2. * np.array([[0, -1.0j], [1.0j, 0]])
sz = 1 / 2. * np.array([[1, 0], [0, -1]])
# spin 1
SX = 1. / np.sqrt(2) * np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
SY = 1. / np.sqrt(2) * np.array([[0., -1.0j, 0.], [1.0j, 0., -1.0j],
                                 [0., 1.0j, 0.]])
SZ = np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., -1.]])


def degenerate_(es, k=0):
    es = np.sort(es)
    ES = pd.Series([1], index=[es[0]])
    j = 0
    k = k
    for i in es[1:]:
        if np.abs(i - es[j]) <= k:
            ES[es[j]] += 1
        else:
            ES = ES.append(pd.Series([1], index=[i]))
            j += ES[es[j]]
    df = pd.DataFrame(
        {
            'Energy Spectrum': ES.index,
            'Degeneracy': ES.data
        },
        columns=['Energy Spectrum', 'Degeneracy'])
    return (df)


def kron_(n, spin_dim):
    h = 1
    for i in range(n):
        h = kron(h, np.eye(spin_dim))
    return (h)


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
        return (None)

    H = np.zeros([spin_dim**n_site, spin_dim**n_site], dtype=complex)
    if i == n_site - 1:
        for s in spin:
            h = kron(s, kron_(n_site - 2, spin_dim))
            h = kron(h, s)
            # np.add(H, h, out=H, casting="unsafe")
            H += h
    else:
        for s in spin:
            h = kron(kron_(i, spin_dim), s)
            h = kron(h, s)
            h = kron(h, kron_(n_site - 1 - (i + 1), spin_dim))
            # np.add(H, h, out=H, casting="unsafe")
            H += h
    return (H)


def linear_(n_site, spin_dim):
    """
    with periodic boundary condition, now the # of bond is the same as the # of sites.
    """
    H = np.zeros([spin_dim**n_site, spin_dim**n_site], dtype=complex)
    for i in range(n_site):
        h = spin_list_position(i, n_site, spin_dim)
        # np.add(H, h, out=H, casting="unsafe")
        H += h
    return (H)


def quandratic_(n_site, spin_dim):
    H = np.zeros([spin_dim**n_site, spin_dim**n_site],dtype=complex)
    for i in range(n_site):
        h = spin_list_position(i, n_site, spin_dim)
        h = np.tensordot(h, h, axes=1)
        # np.add(H, h, out=H, casting="unsafe")
        H += h
    return (H)


def heisenberg_(n_site, spin_dim, J):
    H = J * linear_(n_site, spin_dim)
    return (H)


def blb_(n_site, spin_dim, Jl, Jq):
    H = Jl * linear_(n_site, spin_dim) + Jq * quandratic_(n_site, spin_dim)
    return (H)


def exact_(i_theta=0, n_theta=2, n_site=2, spin_dim=3):
    theta = i_theta / n_theta * np.pi
    Jl = np.cos(theta)
    Jq = np.sin(theta)
    H = blb_(n_site, spin_dim, Jl, Jq)
    # H = heisenberg_(n_site, spin_dim, J)
    w, v = np.linalg.eigh(H)
    # np.save("n_site_{}".format(n_site), w / n_site)
    # print('GS energy density:\n', w[0] / n_site)
    GS_energy_density = w[0] / n_site
    return GS_energy_density
    # print("GS wavefunction:", v[0])
    # df = degenerate_(w/n_site,0.1)
    # print(df)
