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
FLAGS = None


# Define the Hamiltonian
class hamiltonian(object):
	def __init__(self,Lp,Rp,w,dtype=float):
		self.Lp = Lp
		self.Rp = Rp
		self.w = w
		self.d = w.shape[3]
		self.chia = Lp.shape[0]
		self.chib = Rp.shape[0]
		self.shape = np.array([self.d**2*self.chia*self.chib,self.d**2*self.chia*self.chib])
		self.dtype = dtype

	def matvec(self,x):
		x=np.reshape(x,(self.d,self.chia,self.d,self.chib))
		x=np.tensordot(self.Lp,x,axes=(0,1))
		x=np.tensordot(x,self.w,axes=([1,2],[0,2]))
		x=np.tensordot(x,self.w,axes=([3,1],[0,2]))
		x=np.tensordot(x,self.Rp,axes=([1,3],[0,2]))
		x=np.reshape(np.transpose(x,(1,0,2,3)),((self.d*self.d)*(self.chia*self.chib)))
		if(self.dtype==float):
			return np.real(x)
		else:
			return(x)

def idmrg(B,s,N,d,Lp,Rp,w,chi,H_bond):
    y = []
    energy = []
    Psi = []
    if N > 100:
        n_step = 20
    else:
        n_step = 1
    # Now the iterations
    for step in range(N):
        E = [] # why here a E[]? and everything seems ok if I comment this?
        for i_bond in [0,1]:
            ia = np.mod(i_bond-1,2); ib = np.mod(i_bond,2); ic = np.mod(i_bond+1,2)
            chia = B[ib].shape[1]; chic = B[ic].shape[2]

            # Construct theta matrix #
            theta0 = np.tensordot(np.diag(s[ia]),np.tensordot(B[ib],B[ic],axes=(2,1)),axes=(1,1))
            theta0 = np.reshape(np.transpose(theta0,(1,0,2,3)),((chia*chic)*(d**2)))
            # print(theta0.shape)

            # Diagonalize Hamiltonian #
            H = hamiltonian(Lp,Rp,w,dtype=float)
            e0,v0 = arp.eigsh(H,k=1,which='SA',return_eigenvectors=True,v0=theta0)
            theta = np.reshape(v0.squeeze(),(d*chia,d*chic));

            # Schmidt deomposition #
            X, Y, Z = np.linalg.svd(theta); Z = Z.T
            chib = np.min([np.sum(Y>10.**(-12)), chi])
            X = np.reshape(X[:d*chia,:chib],(d,chia,chib))
            Z = np.transpose(np.reshape(Z[:d*chic,:chib],(d,chic,chib)),(0,2,1))

            # Update Environment #
            Lp = np.tensordot(Lp, w, axes=(2,0))
            Lp = np.tensordot(Lp, X, axes=([0,3],[1,0]))
            Lp = np.tensordot(Lp, np.conj(X), axes=([0,2],[1,0]))
            Lp = np.transpose(Lp,(1,2,0))

            Rp = np.tensordot(w, Rp, axes=(1,2))
            Rp = np.tensordot(np.conj(Z),Rp, axes=([0,2],[2,4]))
            Rp = np.tensordot(Z,Rp, axes=([0,2],[2,3]))

            # Obtain the new values for B and s #
            s[ib] = Y[:chib]/np.sqrt(sum(Y[:chib]**2))
            B[ib] = np.transpose(np.tensordot(np.diag(s[ia]**(-1)),X,axes=(1,1)),(1,0,2))
            B[ib] = np.tensordot(B[ib], np.diag(s[ib]),axes=(2,1))

            B[ic] = Z
            E=[]
            for i_bond in range(2):
                BB = np.tensordot(B[i_bond],B[np.mod(i_bond+1,2)],axes=(2,1))
                sBB = np.tensordot(np.diag(s[np.mod(i_bond-1,2)]),BB,axes=(1,1))
                C = np.tensordot(sBB,np.reshape(H_bond,[d,d,d,d]),axes=([1,2],[2,3]))
                E.append(np.squeeze(np.tensordot(np.conj(sBB),C,
                                                 axes=([0,3,1,2],[0,1,2,3]))).item())
            if step % n_step == 0:
                energy.append(np.mean(E))
                Psi.append(theta)


    return(B,s,Psi,energy)



def main():
    sx = np.array([[0.,1.],[1.,0.]])
    sz = np.array([[1.,0.],[0.,-1.]])
    sy = np.array([[0,-1j],[1j,0]])

    SX = 1/np.sqrt(2)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
    SY = 1/np.sqrt(2)*np.array([[0.,-1.0j,0.],[1.0j,0.,-1.0j],[0.,1.0j,0.]])
    SZ = np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]])


    # First define the parameters of the model / simulation
    g=0.5; d=2; N=14
    chi = FLAGS.chi
    J = FLAGS.J
    n_chi = FLAGS.n_chi
    theta = FLAGS.theta*np.pi
    d=3; d_of_MPO = 14
    Jl = np.cos(theta)
    Jq = np.sin(theta)

    # Generate the Hamiltonian, MPO, and the environment
    # # Ising model
    # H_bond = np.array( [[J,g/2,g/2,0], [g/2,-J,0,g/2], [g/2,0,-J,g/2], [0,g/2,g/2,J]] )
    # w = np.zeros((3,3,2,2),dtype=np.float)
    # w[0,:2] = [np.eye(2),sz]
    # w[0:,2] = [g*sx, J*sz, np.eye(2)]
    # Lp = np.zeros([1,1,3]); Lp[0,0,0] = 1.
    # Rp = np.zeros([1,1,3]); Rp[0,0,2] = 1.

    # # heisenberg model
    # H_bond = np.array([[J,0,0,0],[0,-J,2*J,0],[0,2*J,-J,0],[0,0,0,J]])
    # w = np.zeros((5,5,d,d),dtype=np.complex)
    # w[0,:4] = [np.eye(2), sx, sy, sz]
    # w[1:,4] = [J*sx, J*sy, J*sz, np.eye(2)]
    # Lp = np.zeros([1,1,5]); Lp[0,0,0] = 1.
    # Rp = np.zeros([1,1,5]); Rp[0,0,4] = 1.

    # Bilinear-biquadratic model
    H_bond = Jl*(np.kron(SX,SX)+np.kron(SY,SY)+np.kron(SZ,SZ))\
             +Jq*(np.kron(SX*SX,SX*SX)+np.kron(SY*SY,SY*SY)+np.kron(SZ*SZ,SZ*SZ)\
             +np.kron(SX*SY,SX*SY)+np.kron(SY*SX,SY*SX)+np.kron(SY*SZ,SY*SZ)\
             +np.kron(SZ*SY,SZ*SY)+np.kron(SZ*SX,SZ*SX)+np.kron(SX*SZ,SX*SZ))
    w = np.zeros((14,14,d,d),dtype=np.complex)
    w[0,:13] = [np.eye(d), SX, SY, SZ, SX*SX, SY*SY, SZ*SZ, SX*SY, SY*SX,
                SY*SZ, SZ*SY, SZ*SX, SX*SZ]
    w[1:,13] = [Jl*SX, Jl*SY, Jl*SZ, Jq*SX*SX, Jq*SY*SY, Jq*SZ*SZ, Jq*SX*SY, Jq*SY*SX,
                Jq*SY*SZ, Jq*SZ*SY, Jq*SZ*SX, Jq*SX*SZ, np.eye(d)]
    Lp = np.zeros([1,1,14]); Lp[0,0,0] = 1.
    Rp = np.zeros([1,1,14]); Rp[0,0,13] = 1.

    Energy_with_dffrt_chi = []
    for chi2 in range(chi,chi+n_chi):
        B=[];s=[]
        for i in range(2):
            B.append(np.zeros([3,1,1])); B[-1][0,0,0]=1
            s.append(np.ones([1]))
        B,s,Psi,energy = idmrg(B,s,N,d,Lp,Rp,w,chi2,H_bond)
        # # Get the bond energies
        E=[];
        for i_bond in range(2):
            BB = np.tensordot(B[i_bond],B[np.mod(i_bond+1,2)],axes=(2,1))
            sBB = np.tensordot(np.diag(s[np.mod(i_bond-1,2)]),BB,axes=(1,1))
            C = np.tensordot(sBB,np.reshape(H_bond,[d,d,d,d]),axes=([1,2],[2,3]))
            E.append(np.squeeze(np.tensordot(np.conj(sBB),C,
                                             axes=([0,3,1,2],[0,1,2,3]))).item())
        Energy_with_dffrt_chi.append(np.mean(E))
        print("chi={},E_iDMRG ={}".format(chi2,np.mean(E)))



    # # exact energy for Ising model
    # f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
    # E0_exact = integrate.quad(f, 0, np.pi, args=(g,))[0]
    # print("E_exact =", E0_exact)

    fig, ax = plt.subplots(1,2)
    ax[0].plot(range(len(energy)),energy)
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('GS energy')
    ax[1].set_xlabel('$\chi$')
    ax[1].plot(range(chi2,chi2+n_chi),Energy_with_dffrt_chi)

    plt.ylim(-1.5,-1)
    plt.tight_layout()
    plt.show()

    print('Bipartition Entanglement Spectrum')
    x,y,z = np.linalg.svd(Psi[-1])
    es = -np.log(y)
    es = np.sort(es)

    ES = pd.Series([1],index=[es[0]])
    j = 0;k = 1
    for i in es[1:]:
        if np.abs(i-es[j]) < k:
            ES[es[j]] += 1
        else:
            ES = ES.append(pd.Series([1],index=[i]))
            j += ES[es[j]]
    print(ES)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chi',type=int, default=15,
                        help='bond dimension')
    parser.add_argument('--J',type=float, default=1.,
                        help='J')
    parser.add_argument('--n_chi',type=int, default=5,
                        help='n_chi')
    parser.add_argument('--theta',type=float, default=0,
                        help='theta in unit of pi')
    FLAGS = parser.parse_args()
    main()
