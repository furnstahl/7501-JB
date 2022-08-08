#!/usr/bin/env python
# coding: utf-8

# # **Background Theory**: Avoided Crossing in One Dimensional Asymmetric Quantum Well
# 
# <i class="fa fa-book fa-2x"></i><a href="../asymmetricwell.ipynb" style="font-size: 20px"> Go back to the interactive notebook</a>
# 
# **Source code:** https://github.com/osscar-org/quantum-mechanics/blob/master/notebook/quantum-mechanics/theory/theory_asymmetricwell.ipynb
# 
# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />

# ## **Introduction**
# 
# In quantum physics and quantum chemistry, an avoided crossing is the phenomenon 
# where two eigenvalues of a Hermitian matrix representing a quantum observable, 
# and depending on N continuous real parameters, cannot become equal in value 
# ("cross") except on a manifold of N-2 dimensions. Please read the 
# [Wikipedia page](https://en.wikipedia.org/wiki/Avoided_crossing)
# on the avoided crossing for more information.
# 
# In this notebook, we solve the Schrödinger equation for a 1D 
# potential. The formula of the potential is:
# 
# $$\large V(x) = x^4 - 0.6x^2 + \mu x$$
# 
# where, $x$ is the position, and $\mu$ is the potential parameter 
# (which can be tuned using a slider in the interactive notebook) that determines the symmetry 
# of the two quantum wells. 

# <summary style="font-size: 20px">Two-state system</summary>
# Consider a quantum system which has only two states E$_1$ and E$_2$.
# It is a good model to study a variety of physical systems. The two-state
# Hamiltonian H can be formulated in matrix form:
#     
# $$H = \begin{pmatrix} E_1 & 0 \\ 0 & E_2 \end{pmatrix} \quad (1)$$
#     
# When a perturbation is introduced, the Hamiltonian of the perturbed
# system H' is written as:
#     
# $$H' = H + P = \begin{pmatrix} E_1 & 0 \\ 0 & E_2 \end{pmatrix}
# + \begin{pmatrix} 0 & W \\ W^* & 0 \end{pmatrix}
# = \begin{pmatrix} E_1 & W \\ W^* & E_2 \end{pmatrix} \quad (2)$$
#     
# The new eigenvalues are calculated as:
#     
# $$E_+ = \frac{1}{2}(E_1+E_2) + \frac{1}{2}\sqrt{(E_1-E_2)^2+4|W|^2} \quad (4)$$
# $$E_- = \frac{1}{2}(E_1+E_2) - \frac{1}{2}\sqrt{(E_1-E_2)^2+4|W|^2} \quad (5)$$
#     
# One can see that E$_+$ is always bigger than E$_-$.
# The energy difference $E_+ - E_-$ is $\sqrt{(E_1-E_2)^2+4|W|^2}$.
# Hence, the perturbation prevents the two energy states from crossing.
# 
