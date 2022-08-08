#!/usr/bin/env python
# coding: utf-8

# # **Background Theory**:  Numerical Solution of 1D Time Dependent Schrödinger Equation for nuclear evolution on multiple electronic potential energy surfaces by the Multiple Split Operator Fourier Transform (MSOFT) method
# 
# <i class="fa fa-book fa-2x"></i><a href="../msoft.ipynb" style="font-size: 20px"> Go back to the interactive notebook</a>
# 
# **Source code:** https://github.com/osscar-org/quantum-mechanics/blob/master/notebook/quantum-mechanics/theory/theory_msoft.ipynb
# 
# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />

# The MSOFT algorithm is one approach to exactly (to within numerical accuracy) simulate the nonadiabatic dynamical evolution of quantum systems. Nonadiabatic effects must be accounted for when a given quantum mechanical system possesses more than one electronic state and the coupling between the different states is sufficiently large so as to induce transitions between them. For simplicity, in this notebook we shall consider the application of MSOFT to a two level system. It is, however, entirely possible (at least in theory) to use the method in the presence of arbitrarily many electronic states. 
#     
# # **Propagation Scheme**
# 
# <p style="text-align: justify;font-size:15px;width: 90%">
# 
# 
#  Working in the adiabatic basis $\{ |R \rangle | \phi_{\alpha} (R)\rangle \}$ (within which the electronic Hamiltonian $\hat{h}_{\text{el}} ( \hat{R})$ is diagonal), the dynamical equations governing the evolution of the nuclear wavepacket components, $c_1(t)$ and $c_2(t)$, on each of the two electronic states are 
# $i \hbar \frac{\partial }{\partial t} c_1(R,t) = \left\{ - \frac{\hbar^2}{2M} \frac{\partial^2 }{\partial R^2 } + E_1(R)  \right\} c_1(R,t) + \frac{\hbar}{M} D _{12}(R) \frac{\partial }{\partial R} c_2(R,t)$ and
# $i \hbar \frac{\partial }{\partial t} c_2(R,t) = \left\{ - \frac{\hbar^2}{2M} \frac{\partial^2 }{\partial R^2 } + E_2(R)  \right\} c_2(R,t) + \frac{\hbar}{M} D _{21}(R) \frac{\partial }{\partial R} c_1(R,t)$ , where
#  $D _{12}(R) = D _{21}^*(R) = \langle \phi _1(R) | \frac{d}{dR} \phi_2(R) \rangle$.<br/>
# 
#  We may also consider introducing the so-called diabatic basis $\{| R \rangle | \lambda \rangle\}$ in which we can write the state of the system as $|\Psi (t) \rangle = \sum \limits_{\lambda}^{} \int dR \chi _{\lambda}(R,t) |R\rangle | \lambda \rangle$ and for which the kinetic energy operator is now diagonal (but the electronic Hamiltonian is non-diagonal). In this basis the above evolution equations take the form<br/>
# 
# $i \hbar \frac{\partial }{\partial t} \chi_1(R,t) = \left\{ - \frac{\hbar^2}{2M} \frac{\partial^2 }{\partial R^2 } + h_{11}(R)  \right\} \chi_1(R,t) + h _{12}(R)  \chi_2(R,t)$ <br/>
# $i \hbar \frac{\partial }{\partial t} \chi_2(R,t) = \left\{ - \frac{\hbar^2}{2M} \frac{\partial^2 }{\partial R^2 } + h_{22}(R)  \right\} \chi_2(R,t) +  h _{21}(R)  \chi_1(R,t)$
# <br/> 
# <br/> 
# 
# # **Steps of the MSOFT algorithm**
# 
# Starting with the initial state $|\Psi(R,0)\rangle$ of the system and the electronic Hamiltonian $\hat{h}_{\text{el}}$ expressed in the diabatic basis the MSOFT algorithm proceeds as follows:<br/>
#     $\qquad$(INITIALIZATION): Diagonalize $\hat{h}_{ \text{el} }$: $\hat{D}_{e}^T \hat{h}_{ \text{el} } \hat{D_e} = \begin{pmatrix}E_1&0\\0&E_2\end{pmatrix} = \hat{E}$.<br/>
#     $\qquad$(INITIALIZATION): Set up potential and kinetic propagators: $\exp(- \frac{i}{\hbar} \frac{dt}{2} \hat{E} )$ and  $\exp(- \frac{i}{\hbar} \frac{\hat{p}^2}{2M}dt )$<br/>
#     Perform the following loop for a specifies number of timesteps $\text{N}  _{ \text{steps} }$:<br/>
#     $\qquad$(1) Transform to the adiabatic basis: $|\Psi(R,0)\rangle _{ \text{Di} } \to |\Psi(R,0)\rangle _{ \text{Adi} } = \hat{D_e} |\Psi(R,0)\rangle _{ \text{Di} }$<br/>
#     $\qquad$(2) Apply half-step potential propagator:  $|\Phi(R, \frac{dt}{2} )\rangle _{ \text{Adi} }= \exp(- \frac{i}{\hbar} \frac{dt}{2} \hat{E} ) |\Psi(R,0)\rangle _{ \text{Adi} }$.<br/>
#     $\qquad$(3) Transform back to the diabatic basis $|\Phi(R, \frac{dt}{2} )\rangle _{ \text{Adi} } \to |\Phi(R, \frac{dt}{2} )\rangle _{ \text{Di} } = \hat{D_e}^T |\Phi(R, \frac{dt}{2} )\rangle _{ \text{Adi} }$<br/>
#     $\qquad$(4) Apply fast Fourier transform (FFT) to transform state to reciprocal space and render kinetic energy operator diagonal in nuclear part of total tensor product space
#  $\mathcal{H}_N \otimes \mathcal{H} _e$: $|\Phi(R, \frac{dt}{2} )\rangle _{ \text{Di} } \to \text{FFT}[ |\Phi(R, \frac{dt}{2} )\rangle _{ \text{Di} }] =  |\tilde{\Phi}(P, \frac{dt}{2} )\rangle _{ \text{Di} }$<br/>
#  $\qquad$(5) Apply full-timestep kinetic energy propagator: $|\tilde{\Xi}(P, dt )\rangle _{ \text{Di} }=   \exp(- \frac{i}{\hbar} \frac{\hat{P}^2}{2M} dt ) |\tilde{\Phi}(P,dt)\rangle _{ \text{Di} }$.<br/>
# $\qquad$(6) Apply an inverse FFT to transform state back to direct space: $|\tilde{\Xi}(P, dt )\rangle _{ \text{Di} } \to \text{IFFT}[|\tilde{\Xi}(P, dt )\rangle _{ \text{Di}}] = |\Xi(R, dt )\rangle _{ \text{Di}}$.<br/>
#  $\qquad$(7) Transform to the adiabatic basis: $|\Xi(R, dt )\rangle _{ \text{Di}} \to |\Xi(R, dt )\rangle _{ \text{Adi}} = \hat{D_e}^T |\Xi(R, dt )\rangle _{ \text{Di}}|$.<br/>
# $\qquad$(8)  Apply half-step potential propagator:  $|\Psi(R, dt)\rangle _{ \text{Adi} }= \exp(- \frac{i}{\hbar} \frac{dt}{2} \hat{E} ) |\Xi(R,dt)\rangle _{ \text{Adi} }$.<br/>
# $\qquad$This leaves us with the state of the system evolved by one full timestep $dt$. We then repeat steps (2)-(8) $ \text{N} _{steps}-1$ times to achieve a total propagation time of $t= \text{N} _{steps} \cdot dt$ for the system.
# </p>
# 
