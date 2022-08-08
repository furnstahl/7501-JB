#!/usr/bin/env python
# coding: utf-8

# # **Numerical Solution of 1D Time Dependent Schrödinger Equation by Split Operator Fourier Transform (SOFT) Method**
# 
# 
# <i class="fa fa-book fa-2x"></i><a href="../soft.ipynb" style="font-size: 20px"> Go back to the interactive notebook</a>
# 
# **Source code:** https://github.com/osscar-org/quantum-mechanics/blob/master/notebook/quantum-mechanics/theory/theory_soft.ipynb

# ## **Background theory**
# 
# In previous notebooks, we focus on numerical solutions of the time-independent
# Schrödinger equation. Here, we demonstrate the numercial solution of the 
# one-dimensional time dependent Schrödinger equation. The split operator 
# Fourier transform (SOFT) was employed.
# 
# <details open>
# <summary style="font-size: 20px">Propagation operator</summary>
# Let's consider a time-independent Hamiltonian and its associated time-dependent
# Schrödinger equation for a system of one particle in one dimension.
#     
# $$\large i\hbar\frac{d}{dt}|\psi> = \hat{H}|\psi> \quad \text{where} \quad 
# \hat{H} = \frac{\hat{P}^2}{2m} + V(\hat{x})$$
# 
# The time evolution of the eigenstates can be formulated as:
#     
# $$\large \psi_n(x,t) = \psi_n(x)e^{-iE_nt/\hbar}$$
#     
# For a small time $\Delta t$, the evolution of the wavefunction from $t=0$
# to $t=\Delta t$ can be formulated as:
#     
# $$\large \psi(x, \Delta t) = e^{-iH\Delta t/\hbar}\psi(x, 0)
# =\sum_{n=0}^{\infty} \frac{(-1)^n}{n!}\left(\frac{iH\Delta t}{\hbar}\right)^n \psi(x,0)
# =U(\Delta t)\psi(x,0)$$
#     
# and where the $U(\Delta t)$ is called the unitary propagation operator.
# The $U$ is Hermitian, which fulfills the condition:
#     
# $$\large UU^\dagger = e^{-iHt/\hbar}e^{-iHt/\hbar \dagger}
# = e^{-iHt/\hbar}e^{iHt/\hbar} = I$$
#     
# The time-evolution operator is also reversible or symmetric
# in thime:
#     
# $$\large U(-\Delta t)U(\Delta t)|\psi(x,t)> = |\psi(x,t)>$$
# </details>

# <details open>
# <summary style="font-size: 20px">Split operator Fourier transform</summary>
# We know that this equation admits at least a formal solution of the kind
# $|\psi(t)> = \exp\biggl[-\frac{i}{\hbar}\hat{H}t\biggr]|\psi(0)>$
# that projected on the coordinate basis gives the (still formal) solution
# $\psi(x_t,t) = \int dx_0 K(x_t, t; x_0, 0)\psi(x_0,0)$
# where $ K(x_t, t; x_0, 0)= < x_t|\exp\biggl[-\frac{i}{\hbar}\hat{H}t\biggr]|x_0 > $
# Note that $x_t$ and $x_0$ are just labels for the coordinates, as if we had $x$ and $x'$.
# 
# $$\large k(x_t, x_0) =  < x_t|e^{-\frac{i}{\hbar}\hat{H}t} | x_0 > = < x_{N+1} | \underbrace{e^{-\frac{i}{\hbar}t/N} e^{-\frac{i}{\hbar}t/N} ... e^{-\frac{i}{\hbar}t/N}}_\textrm{N} |x_0 >$$
#     
# Let us then focus on the single step propogator.
#     
# $$\large < x_1 |\psi(\epsilon) > = \psi(x_1,\epsilon) = \int dx_0 < x_1 | 
# e^{-\frac{i}{\hbar}\hat{H}\epsilon} |x_0 > \psi(x_0,0)$$
#     
# We can use the Trotter approximation to write:
#     
# $$\large < x_1 |e^{-\frac{i}{\hbar}\hat{H}\epsilon}| x_0 > = < x_1 | e^{-\frac{i}{\hbar}
# [\frac{\hat{P^2}}{2m}+V(\hat{x})]\epsilon} | x_0> \approx < x_1 | e^{-\frac{i}
# {\hbar}V(\hat{x})\epsilon/2}e^{-\frac{i}{\hbar}\frac{\hat{P^2}}{2m}\epsilon}e^{-\frac{i}
# {\hbar}V(\hat{x})\epsilon/2} | x_0 >$$
#     
# $$\large =e^{-\frac{i}{\hbar}V(\hat{x})\epsilon /2} \int dp < x_1 | e^{-\frac{i}{\hbar}\frac{\hat{P^2}}{2m}\epsilon} | p > < p | x_0 > e^{ 
# \frac{i}{\hbar}V(\hat{x})\epsilon/2}$$
#     
# where, $< p | x_0 > = \frac{1}{\sqrt{2\pi\hbar}}e^{-\frac{i}{\hbar}Px_0}$.
#     
# $$\large \psi(x_1,\epsilon)=e^{-\frac{1}{\hbar}V(x_1)\epsilon/2}\int \frac{dp}{\sqrt{2\pi\hbar}}e^{\frac{i}{\hbar}px_1}e^{-\frac{i}{\hbar}\frac{p^2}{2m}\epsilon}\underbrace{\int \frac{dx_0}{\sqrt{2\pi\hbar}}e^{-\frac{i}{\hbar}px_0}\underbrace{e^{-\frac{i}{\hbar}V(x_0)\frac{\epsilon}{2}}\psi(x_0,0)}_{\Phi_{\frac{\epsilon}{2}}(x_0)}}_{\tilde{\Phi}_{\frac{\epsilon}{2}}(p)}$$
#     
# $$\large \psi(x_1,\epsilon)=e^{-\frac{1}{\hbar}V(x_1)\epsilon/2}\underbrace{\int \frac{dp}{\sqrt{2\pi\hbar}}e^{\frac{i}{\hbar}px_1}\underbrace{e^{-\frac{i}{\hbar}\frac{p^2}{2m}\epsilon}\tilde{\Phi}_{\frac{\epsilon}{2}}(p)}_{\tilde{\Phi}(p)}}_{\tilde{\Phi}(x_1)}$$
#     
# By interating N times, we can obtain $\psi(x,t)$. In summary, the split operator
# Fourier transfer algorithm can be conducted into five step as shown below:
# 
# <img src="../images/SOFT_algorithm.png" style="height:250px;">
# </details>
