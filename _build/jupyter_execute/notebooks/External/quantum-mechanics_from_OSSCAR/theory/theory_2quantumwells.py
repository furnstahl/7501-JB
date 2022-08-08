#!/usr/bin/env python
# coding: utf-8

# # **Background Theory**: Numerical Solution of the Schrödinger Equation for the Double Square Well Potential
# 
# <i class="fa fa-book fa-2x"></i><a href="../2quantumwells.ipynb" style="font-size: 20px"> Go back to the interactive notebook</a>
# 
# **Source code:** https://github.com/osscar-org/quantum-mechanics/blob/master/notebook/quantum-mechanics/theory/theory_2quantumwells.ipynb
# 
# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />

# ## **Introduction**
# 
# We implemented a numerical method to solve the Shrödinger equation for the one-dimensional double square well potential in this notebook. The numerical method and its algorithm are explicitly outlined in the  ["Numerical Solution of the Schrödinger Equation for a One Dimensional Quantum Well"](../1quantumwell.ipynb) notebook.

# ## **DSWP model for H$_2$ molecule**
# 
# The double square well potential model is a simple but efficient way to describe 
# real molecular or material systems. For instance, the DSWP model can be used 
# to describe the H$_2$ molecule.
#     
# <div class="container" style="text-align: center; width: 350px;">
#   <img src="../images/DSPW_H2.png" alt="H2 molecule as a DSWP model" class="image">
#   <div class="overlay">A symmetric double square well potential to describe an H$_2$ molecule.</div>
# </div>
#     
# In the H$_2$ molecule, there are two identical protons and two electrons.
# We can simplify the potential as two identical square well potential as
# shown in the figure above. The two lowest energy eigenstates are indicated in the blue and red
# lines in the figure. The red line corresponds to the state that has the lowest energy eigenvalue and an associated symmetric wavefunction. While the blue line has the second
# lowest energy eigenvalue and an antisymmetric wavefunction. One can see the
# double square well potential model is far from the true potential in 
# the H$_2$ molecule system. However, this simple model gives a roughly
# correct description of the wavefunctions.

# ## **DSWP model for ammonia maser**
# 
# <div class="container" style="text-align: center; width: 350px;">
#   <img src="../images/ammonia-maser.png" alt="H2 molecule as a DSWP model" class="image">
#   <div class="overlay">The ammonia maser with the DSWP model.</div>
# </div>
# 
# Another example is using the DSWP model to explain the ammonia maser. 
# As shown in the figure above, the nitrogen in the NH$_3$ molecule can 
# oscillate up and down. Consider starting from an initial state, where the nitrogen is 
# localized on the left quantum well. The wavefunction can be formulated 
# as a combination of the two lowest quantum states.
#     
# $$\large |\psi(t=0)> = \frac{1}{\sqrt{2}}[|\psi_1>+|\psi_2>]$$
# 
# The time-resolved state is written as:
#     
# $$\large |\psi(t)> = \frac{1}{\sqrt{2}}\left[e^{-iE_1t/\hbar}|\psi_1> 
# +e^{-iE_2t/\hbar}|\psi_2> \right]$$
#   
# $$\large = \frac{1}{\sqrt{2}}e^{-i(E_1 + E_2)t/(2\hbar)} 
# \left[e^{i\Omega t/2}|\psi_1> + e^{-i\Omega t/2}|\psi_2> \right]$$
#     
# where the $\Omega$ is defined as:
#     
# $$\Omega = \frac{E_2- E_1}{\hbar}$$
# 
# Therefore, the probability density is given as:
#     
# $$\large |\psi(x,t)|^2 = \frac{1}{2}(\psi_1^2+\psi_2^2) + cos(\Omega t)\psi_1\psi_2$$
#     
# At $t=\frac{\pi}{\Omega}$, the wavefunction becomes localized in the
# right quantum well.
#     
# $$\large |\psi(x, t=\frac{\pi}{\Omega})|^2 = \frac{1}{2}|\psi_1 - \psi_2|^2$$
#     
# In this model, the nitrogen moves from left well to right well and back again
# with a frequency $f = \frac{\Omega}{2\pi}$. As we already know, the energy
# difference $E_1 - E_2$ is very small, which leads to a small frequency of
# the nitrogen inversion. The NH$_3$ molecule emits radiation through this process.

# In[ ]:




