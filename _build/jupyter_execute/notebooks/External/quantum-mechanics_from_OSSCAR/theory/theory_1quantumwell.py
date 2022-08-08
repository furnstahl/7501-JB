#!/usr/bin/env python
# coding: utf-8

# # **Background Theory**: Numerical Solution of the Schrödinger Equation for a One Dimensional Quantum Well
# 
# <i class="fa fa-book fa-2x"></i><a href="../1quantumwell.ipynb" style="font-size: 20px"> Go back to the interactive notebook</a>
# 
# **Source code:** https://github.com/osscar-org/quantum-mechanics/blob/master/notebook/quantum-mechanics/theory/theory_1quantumwell.ipynb
# 
# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />

# ## **Introduction**
# 
# The Schrödinger equation is the core of quantum mechanics. 
# Solving it, we can obtain chemical and physical properties of
# molecular and crystal systems. However, solving the Schrödinger equation
# for realistic systems is computationally very expensive. The analytical 
# solution is only available for simple systems, such as a hydrogen atom.
# 
# In this notebook, we demonstrate how to solve the (time-independent) Schrödinger 
# equation numerically for a one-dimensional (1D) quantum well. The eigenvalues and 
# eigenfunctions (wavefunctions) are solved by diagonalization of the
# Hamiltonian, written in matrix form on a regular grid along the 1D axis.
# Eigenfunctions and eigenvalues are shown interactively in the figure displayed in the interactive notebook.

# ## **Schrödinger equation**
# 
# The Schrödinger equation is named after the Austrian-Irish physicist Erwin Schrödinger.
# The (time-independent) Schrödinger equation for a one-dimensional system is:
# 
# $$\Large -\frac{\hbar^2}{2m}\frac{d^2}{d x^2} \psi_n(x) + V(x)\psi_n(x) = E_n\psi_n(x)$$
# 
# where $\psi_n(x)$ is the wavefunction of the 1D system, $\hbar$ is the 
# reduced Planck constant, $V(x)$ is the potential energy, $m$ is the mass 
# of the particle, and $E_n$ is the energy of the system.
# By solving this eigenvalue problem, we can obtain 
# the wavefunctions (or eigenfunctions) $\psi_n(x)$ and the corresponding
# eigenenergies $E_n$, labelled by an integer index $n$.

# ## **Numerical method**
#     
# If we define the Hamiltonian operator $\hat H$ as:
#     
# $$\Large \hat H = -\frac{\hbar^2}{2m}\frac{d^2}{d x^2} + \hat V$$
#     
# the Schrödinger equation can be written as:
#     
# $$\Large \hat H \psi_n = E_n \psi_n$$
#     
# In this form, it is clear that the Schrödinger equation is
# a typical eigenvalue equation. 
# If we discretize the $x$ axis on a regular grid of $N$ points
# $(x_0,x_1,x_2,\ldots,x_{N-1})$, then the wavefunction
# $\psi_n(x)$ can be written as a vector:
# $\psi_n(x) = [\psi_n(x_0),\psi_n(x_1),\ldots,\psi_n(x_{N-1})]$.
# In turn, we can discretize the Hamiltonian operator $\hat H$ as a matrix,
# and then solve the equation by numerical matrix diagonalization.
# 
# Let us briefly discuss how to diagonalize $\hat H$, that is the sum of two terms.
# Discretizing the potential term is easy, since the potential energy is local.
# Therefore, the operation $\hat V\psi_n$ can be written as simply multiplying,
# at each grid point $x_i$, the wavefunction by the value of the potential at $x_i$
# (i.e., $V(x_i)$):
# 
#     
# $$\large \hat V\psi = [V(x_0)\psi_n(x_0),V(x_1)\psi_n(x_1),\ldots,V(x_{N-1})\psi_n(x_{N-1})]$$
#     
# that is, the $V$ operator is a diagonal matrix, where the diagonal values are obtained
# by discretizing the potential energy $V$ on the same grid.
#   
# In order to discretize the kinetic-energy term, we need to first understand how
# to discretize a second-derivative operator.
# If we consider a generic (discretized) 1D function $f(x) = [f_0,f_1,\ldots,f_{N-1}]$, 
# we want to write an approximation for its (discretized)
# second derivative $f''(x) = [f''_0,f''_1,\ldots,f''_{N-1}]$.
# We can do this by approximating the derivative:
#     
# $$f''(x) = \lim_{\delta \rightarrow 0} \frac{f'(x+\delta/2)- f'(x-\delta/2))}{\delta} = \lim_{\delta \rightarrow 0} \frac{f(x+\delta) - 2f(x) + f(x-\delta)}{\delta^2} \approx \frac{f(x+\Delta) - 2f(x) + f(x-\Delta)}{\Delta^2}$$
#  
# 
# where $\Delta$ is the distance between to neighboring grid points ($\Delta=x_1 - x_0$).
# Using this result, we can now write the second-derivative operator in matrix form as:
# 
# $$\large
# \begin{pmatrix}f''_0 \\ f''_1 \\ f''_2 \\\vdots \\ f''_{N-1}\end{pmatrix} = \frac{1}{\Delta^2}
# \begin{pmatrix} -2 & 1 & 0 & 0 & \\ 1 & -2 & 1 & 0 & \\ 
# 0& 1 & -2 & 1 &  \\ &  & \ddots & \ddots & \ddots &\\
# &  & & 1 & -2 \end{pmatrix}\begin{pmatrix}f_0 \\ f_1 \\ f_2 \\\vdots \\ f_{N-1}\end{pmatrix}
# $$
# 
# Putting all results together, the Hamiltonian operator $\hat H$ can be thus
# written as:
#     
# $$\large
# \hat H = 
# \begin{pmatrix} -2C+ V_0 & C & 0 & & \\ C & -2C + V_1 & C & & \\ 0 & C & -2C + V_2 & & \\ & & &\ddots & \\ &&&&-2C + V_{N-1}\end{pmatrix}
# $$
# 
# where $C = -\frac{\hbar^2}{2 m \Delta^2}$.
#     
# Using this expression, is now easy to write a numerical form for $\hat H$ for any potential
# term $V$ and use numerical routines to diagonalize the matrix and obtain eigenvalues and
# eigenfunctions. In this example, we use a square-well potential.

# In[ ]:




