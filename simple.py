""" iDMRG code to find the ground state of
the 1D Ising model on an infinite chain.
The results are compared to the exact results.
Frank Pollmann, frankp@pks.mpg.de"""

# simply used to ensure the correctness of ES got by these code...

import numpy as np
from scipy import integrate
import scipy.sparse.linalg.eigen.arpack as arp
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from exact import degenerate_

from scipy.sparse.linalg import ArpackNoConvergence


# Define the Hamiltonian
class hamiltonian(object):
    def __init__(self, Lp, Rp, w, dtype=float):
        self.Lp = Lp
        self.Rp = Rp
        self.w = w
        self.d = w.shape[3]
        self.chia = Lp.shape[0]
        self.chib = Rp.shape[0]
        self.shape = np.array([
            self.d**2 * self.chia * self.chib,
            self.d**2 * self.chia * self.chib
        ])
        self.dtype = dtype

    def matvec(self, x):
        x = np.reshape(x, (self.d, self.chia, self.d, self.chib))
        x = np.tensordot(self.Lp, x, axes=(0, 1))
        x = np.tensordot(x, self.w, axes=([1, 2], [0, 2]))
        x = np.tensordot(x, self.w, axes=([3, 1], [0, 2]))
        x = np.tensordot(x, self.Rp, axes=([1, 3], [0, 2]))
        x = np.reshape(
            np.transpose(x, (1, 0, 2, 3)),
            ((self.d * self.d) * (self.chia * self.chib)))
        if (self.dtype == float):
            return np.real(x)
        else:
            return (x)


def idmrg(B, s, N, d, Lp, Rp, w, chi, H_bond):
    """
    B,s:the initial guess of the wavefunction in the canonical form.
    N:the number of the steps of the updating
    d:the dimension of the spin
    Lp,Rp:the environments
    w:the MPO
    chi:the bond-dimension
    H_bond:the hamiltonian used to get the final results of the GS energy.
    """
    # Now the iterations
    for step in range(N):
        E = []  # why here an E[]? and everything seems ok if I comment this?
        for i_bond in [0, 1]:
            ia = np.mod(i_bond - 1, 2)
            ib = np.mod(i_bond, 2)
            ic = np.mod(i_bond + 1, 2)
            chia = B[ib].shape[1]
            chic = B[ic].shape[2]

            # Construct theta matrix #
            theta0 = np.tensordot(
                np.diag(s[ia]),
                np.tensordot(B[ib], B[ic], axes=(2, 1)),
                axes=(1, 1))
            theta0 = np.reshape(
                np.transpose(theta0, (1, 0, 2, 3)), ((chia * chic) * (d**2)))

            # Diagonalize Hamiltonian #
            H = hamiltonian(Lp, Rp, w, dtype=float)
            e0, v0 = arp.eigsh(
                H, k=1, which='SA', return_eigenvectors=True, v0=theta0)
            theta = np.reshape(v0.squeeze(), (d * chia, d * chic))

            # Schmidt deomposition #
            X, Y, Z = np.linalg.svd(theta)
            Z = Z.T
            chib = np.min([np.sum(Y > 10.**(-12)), chi])
            X = np.reshape(X[:d * chia, :chib], (d, chia, chib))
            Z = np.transpose(
                np.reshape(Z[:d * chic, :chib], (d, chic, chib)), (0, 2, 1))

            # Update Environment #
            Lp = np.tensordot(Lp, w, axes=(2, 0))
            Lp = np.tensordot(Lp, X, axes=([0, 3], [1, 0]))
            Lp = np.tensordot(Lp, np.conj(X), axes=([0, 2], [1, 0]))
            Lp = np.transpose(Lp, (1, 2, 0))

            Rp = np.tensordot(w, Rp, axes=(1, 2))
            Rp = np.tensordot(np.conj(Z), Rp, axes=([0, 2], [2, 4]))
            Rp = np.tensordot(Z, Rp, axes=([0, 2], [2, 3]))

            # Obtain the new values for B and s #
            s[ib] = Y[:chib] / np.sqrt(sum(Y[:chib]**2))
            B[ib] = np.transpose(
                np.tensordot(np.diag(s[ia]**(-1)), X, axes=(1, 1)), (1, 0, 2))
            B[ib] = np.tensordot(B[ib], np.diag(s[ib]), axes=(2, 1))

            B[ic] = Z
            E = []
            for i_bond in range(2):
                BB = np.tensordot(
                    B[i_bond], B[np.mod(i_bond + 1, 2)], axes=(2, 1))
                sBB = np.tensordot(
                    np.diag(s[np.mod(i_bond - 1, 2)]), BB, axes=(1, 1))
                C = np.tensordot(
                    sBB,
                    np.reshape(H_bond, [d, d, d, d]),
                    axes=([1, 2], [2, 3]))
                E.append(
                    np.squeeze(
                        np.tensordot(
                            np.conj(sBB), C, axes=([0, 3, 1, 2], [0, 1, 2, 3]
                                                   ))).item())
            E_idmrg = np.mean(E)

    return (B, s, theta, E_idmrg)


def get_ES(i_theta, n_theta, N=15, chi=2):
    sx = 1 / 2 * np.array([[0., 1.], [1., 0.]])
    sy = 1 / 2 * np.array([[0, -1.0j], [1.0j, 0]])
    sz = 1 / 2 * np.array([[1., 0.], [0., -1.]])

    SX = 1 / np.sqrt(2) * np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    SY = 1 / np.sqrt(2) * np.array([[0., -1.0j, 0.], [1.0j, 0., -1.0j],
                                    [0., 1.0j, 0.]])
    SZ = np.array([[1., 0., 0.], [0., 0., 0.], [0., 0., -1.]])

    # First define the parameters of the model / simulation
    N = N
    chi = chi

    # Generate the Hamiltonian, MPO, and the environment
    # Bilinear-biquadratic model
    eye3 = np.dot(np.eye(3), np.eye(3))
    i_theta = i_theta
    n_theta = n_theta
    theta = 2. * np.pi / n_theta * i_theta
    d = 3
    Jl = np.cos(theta)
    Jq = np.sin(theta)

    SXX = np.dot(SX, SX)
    SYY = np.dot(SY, SY)
    SZZ = np.dot(SZ, SZ)
    SXY = np.dot(SX, SY)
    SYZ = np.dot(SY, SZ)
    SZX = np.dot(SZ, SX)
    SXZ = np.dot(SX, SZ)
    SYX = np.dot(SY, SX)
    SZY = np.dot(SZ, SY)

    H_bond = Jl*(np.kron(SX,SX)+np.kron(SY,SY)+np.kron(SZ,SZ))\
             +Jq*(np.kron(SXX,SXX)+np.kron(SYY,SYY)+np.kron(SZZ,SZZ)\
             +np.kron(SXY,SXY)+np.kron(SYX,SYX)+np.kron(SYZ,SYZ)\
             +np.kron(SZY,SZY)+np.kron(SZX,SZX)+np.kron(SXZ,SXZ))
    # H_bond = Jl*(np.kron(SX,SX)+np.kron(SY,SY)+np.kron(SZ,SZ))\
    #          +1/3*Jq*(np.kron(SXX,SXX)+np.kron(SYY,SYY)+np.kron(SZZ,SZZ)\
    #                   +np.kron(SXY,SXY)+np.kron(SYX,SYX)+np.kron(SYZ,SYZ)\
    #                   +np.kron(SZY,SZY)+np.kron(SZX,SZX)+np.kron(SXZ,SXZ)\
    #                   +np.kron(SXX,eye3)+np.kron(SYY,eye3)+np.kron(SZZ,eye3)\
    #                   +np.kron(SXY,eye3)+np.kron(SYX,eye3)+np.kron(SYZ,eye3)\
    #                   +np.kron(SZY,eye3)+np.kron(SZX,eye3)+np.kron(SXZ,eye3)\
    #                   +np.kron(eye3,SXX)+np.kron(eye3,SYY)+np.kron(eye3,SZZ)\
    #                   +np.kron(eye3,SXY)+np.kron(eye3,SYX)+np.kron(eye3,SYZ)\
    #                   +np.kron(eye3,SZY)+np.kron(eye3,SZX)+np.kron(eye3,SXZ))
    w = np.zeros((14, 14, d, d), dtype=np.complex)
    w[0, :13] = [
        np.eye(d), SX, SY, SZ, SXX, SYY, SZZ, SXY, SYX, SYZ, SZY, SZX, SXZ
    ]
    w[1:, 13] = [
        Jl * SX, Jl * SY, Jl * SZ, Jq * SXX, Jq * SYY, Jq * SZZ, Jq * SXY,
        Jq * SYX, Jq * SYZ, Jq * SZY, Jq * SZX, Jq * SXZ,
        np.eye(d)
    ]
    dmpo = 14

    # # http://arxiv.org/abs/0910.1811v2 equation(3) spin1 Heisenberg
    # J = n_theta
    # uzz = i_theta
    # dmpo = 5
    # H_bond = J * (np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
    #               ) + 1/2*uzz * (np.kron(SZ, np.eye(3)) + np.kron(np.eye(3), SZ))
    # w = np.zeros((5, 5, d, d), dtype=np.complex)
    # w[0, :] = [np.eye(d), SX, SY, SZ, uzz * SZZ]
    # w[1:, 4] = [SX, SY, SZ, np.eye(d)]


    # environment is commonly used..
    Lp = np.zeros([1, 1, dmpo])
    Lp[0, 0, 0] = 1.
    Rp = np.zeros([1, 1, dmpo])
    Rp[0, 0, dmpo - 1] = 1.

    # get the GS energy
    B = []
    s = []
    for i in range(2):
        B.append(np.zeros([d, 1, 1]))
        B[-1][0, 0, 0] = 1
        s.append(np.ones([1]))
    B, s, Theta, E_idmrg = idmrg(B, s, N, d, Lp, Rp, w, chi, H_bond)
    X, Y, Z = np.linalg.svd(Theta)
    Y = Y[:chi] / np.sqrt(sum(Y[:chi]**2))
    es = -2 * np.log(Y)
    es = np.sort(es)

    return (es)


def main():
    n_theta = 2
    N = 15
    chi = 20
    es = []
    # for i_theta in np.arange(n_theta):
    #     print('i =', i_theta)
    #     try:
    #         es_temp = get_ES(i_theta=i_theta, n_theta=n_theta, N=N, chi=chi)
    #     except ArpackNoConvergence:
    #         continue
    #     es.append(es_temp)
    i = 0
    for i_theta in np.linspace(-1, 2, 14) * n_theta:
        i += 1
        print(i_theta)
        # print('i = ', i)
        try:
            es_temp = get_ES(i_theta=i_theta, n_theta=n_theta, N=N, chi=chi)
        except ArpackNoConvergence:
            continue
        es.append(es_temp)
    print(es)
    fig, ax = plt.subplots(1)
    props = dict(boxstyle='square', alpha=0.5, facecolor='white')

    for i, v in enumerate(es):
        site = (i + 1) * np.ones([v.shape[0]])
        plt.errorbar(site, v, xerr=0.5, fmt=' ', color='blue')

    # xticks = [0, 0.5, 1, 1.5, 2]
    # xticks = np.array(xticks)
    # n = n_theta / xticks[-1]
    # plt.xticks(n_theta / xticks[-1] * xticks, xticks)

    ymax = 12
    # plt.axvline(x=0.25 * n_theta / xticks[-1], ymax=ymax)
    # plt.axvline(x=0.50 * n_theta / xticks[-1], ymax=ymax)
    # plt.axvline(x=1.25 * n_theta / xticks[-1], ymax=ymax)
    # plt.axvline(x=1.75 * n_theta / xticks[-1], ymax=ymax)
    # ylim = ax.get_ylim()
    # ymin, ymax = ylim[:]
    # plt.xlim(0, 2 * n)
    # plt.ylim(0, ymax)
    plt.ylim(0,13)
    # textstr = '$N = %d$\n$chi=%d$' % (N, chi)
    # plt.text(
    #     0.05 * n_theta,
    #     0.95 * ymax,
    #     textstr,
    #     fontsize=14,
    #     verticalalignment='top',
    #     bbox=props)
    # plt.text(1*n_theta,30,textstr)
    # plt.xlabel(r'$\theta(\pi)$')
    # plt.ylabel('Entanglement Energy')
    plt.tight_layout()
    plt.savefig('14893_0_0910_1811_{}.pdf'.format(chi))
    plt.show()
    return None


if __name__ == "__main__":
    main()
