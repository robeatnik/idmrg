""" iDMRG code to find the ground state of
the 1D Ising model on an infinite chain.
The results are compared to the exact results.
Frank Pollmann, frankp@pks.mpg.de"""

import numpy as np
from scipy import integrate
import scipy.sparse.linalg.eigen.arpack as arp
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from exact import degenerate_


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


def save_(i_theta, n_theta, data, type_='ES', para0=None, para=None):
    """
    type_='chi','N' or 'ES'
    """
    if type_ == "chi":
        np.save("./chi/energy_with_different_chi{}(N{}_theta{}_{})".format(
            para0, para, n_theta, i_theta), data)
    elif type_ == "N":
        np.save("./step/energy_in_every_step{}(chi{}_theta{}_{})".format(
            para0, para, n_theta, i_theta), data)
    elif type_ == "ES":
        np.save("./ES/entanglement_spectrum_with_theta_{}_{}".format(
            n_theta, i_theta), data)
    else:
        print("Wrong type_!")

    return (None)


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
    energy_in_every_step = []
    if N > 100:
        n_step = 20
    else:
        n_step = 1
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
        if step % (n_step) == 0:
            energy_in_every_step.append(np.mean(E).real)
            # print("step{}".format(step + 1), np.mean(E).real)

    return (B, s, energy_in_every_step, theta)


def get_ES(i_theta, n_theta, N=15, chi=2, n_chi=0, ES=True):
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
    n_chi = n_chi * 5

    # Generate the Hamiltonian, MPO, and the environment
    # Bilinear-biquadratic model
    eye = np.dot(np.eye(3), np.eye(3))
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
    #                   +np.kron(SXX,eye)+np.kron(SYY,eye)+np.kron(SZZ,eye)\
    #                   +np.kron(SXY,eye)+np.kron(SYX,eye)+np.kron(SYZ,eye)\
    #                   +np.kron(SZY,eye)+np.kron(SZX,eye)+np.kron(SXZ,eye)\
    #                   +np.kron(eye,SXX)+np.kron(eye,SYY)+np.kron(eye,SZZ)\
    #                   +np.kron(eye,SXY)+np.kron(eye,SYX)+np.kron(eye,SYZ)\
    #                   +np.kron(eye,SZY)+np.kron(eye,SZX)+np.kron(eye,SXZ))
    w = np.zeros((14, 14, d, d), dtype=np.complex)
    w[0, :13] = [
        np.eye(d), SX, SY, SZ, SXX, SYY, SZZ, SXY, SYX, SYZ, SZY, SZX, SXZ
    ]
    w[1:, 13] = [
        Jl * SX, Jl * SY, Jl * SZ, Jq * SXX, Jq * SYY, Jq * SZZ, Jq * SXY,
        Jq * SYX, Jq * SYZ, Jq * SZY, Jq * SZX, Jq * SXZ,
        np.eye(d)
    ]
    Lp = np.zeros([1, 1, 14])
    Lp[0, 0, 0] = 1.
    Rp = np.zeros([1, 1, 14])
    Rp[0, 0, 13] = 1.

    # get the GS energy
    energy_with_dffrt_chi = []
    for chi2 in np.arange(chi, chi + n_chi + 1, 5):
        B = []
        s = []
        for i in range(2):
            B.append(np.zeros([d, 1, 1]))
            B[-1][0, 0, 0] = 1
            s.append(np.ones([1]))
        B, s, energy_in_every_step, Theta = idmrg(B, s, N, d, Lp, Rp, w, chi2,
                                                  H_bond)
        # # Get the bond energies
        E = []
        for i_bond in range(2):
            BB = np.tensordot(B[i_bond], B[np.mod(i_bond + 1, 2)], axes=(2, 1))
            sBB = np.tensordot(
                np.diag(s[np.mod(i_bond - 1, 2)]), BB, axes=(1, 1))
            C = np.tensordot(
                sBB, np.reshape(H_bond, [d, d, d, d]), axes=([1, 2], [2, 3]))
            E.append(
                np.squeeze(
                    np.tensordot(
                        np.conj(sBB), C, axes=([0, 3, 1, 2], [0, 1, 2, 3])))
                .item())
        energy_with_dffrt_chi.append(np.mean(E))
        # print("E_iDMRG ={}, chi={}".format(np.mean(E).real, chi2))
        delta = np.abs(np.mean(E) - energy_in_every_step[-2])
        if delta <= 1e-3:
            # print('GOOD')
            moreN = 0
        else:
            # print('BAD')
            moreN = 1
    GS_energy = np.mean(E).real
    if n_chi > 0:
        # if there is more than 1 chi case, it means we are focusing on the energy convergence with
        # different chi
        save_(
            para0=chi + n_chi,
            para=N,
            i_theta=i_theta,
            n_theta=n_theta,
            data=energy_with_dffrt_chi,
            type_='chi')
    else:
        # or we are focusing on the convergence along the step
        save_(
            para0=N,
            para=chi,
            i_theta=i_theta,
            n_theta=n_theta,
            data=energy_in_every_step,
            type_='N')

    # # compute the Entanglement Spectrum
    # print('Bipartition Entanglement Spectrum')
    # using s directly
    # if ES:
    #     es = -np.log(np.mean(np.array(s), axis=0))
    #     es = np.sort(es)
    # save_(
    #     i_theta=i_theta,
    #     n_theta=n_theta,
    #     data=es,
    #     type_='ES',
    #     para=i_theta)
    X, Y, Z = np.linalg.svd(Theta)
    Y = Y[:chi] / np.sqrt(sum(Y[:chi]**2))
    es = -np.log(Y)
    es = np.sort(es)
    # # print the es and its degeneracy
    # df = degenerate_(es)
    # print(df)

    return (GS_energy, moreN, es)
