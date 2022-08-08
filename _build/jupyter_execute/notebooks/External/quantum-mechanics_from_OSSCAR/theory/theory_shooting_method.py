#!/usr/bin/env python
# coding: utf-8

# # **Background Theory**: Shooting Method with Numerov Algorithm to Solve the Time Independent Schrödinger Equation for 1D Quantum Well
# 
# <i class="fa fa-book fa-2x"></i><a href="../shooting_method.ipynb" style="font-size: 20px"> Go back to the interactive notebook</a>
# 
# **Source code:** https://github.com/osscar-org/quantum-mechanics/blob/develop/notebook/quantum-mechanics/theory/theory_shooting_method.ipynb
# 
# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />

# ## **Introduction**
# 
# The Schrödinger equation is an ordinary differential equation (ODE) of second order,
# where the 1st-order term does not appear. Numerov algorithm is a numerical method to
# solve this kind of ODEs. One can calculate the wavefunction for any given eigenvalue by
# the Numerov algorithm. However, not all wavefunctions can meet the boundary condition,
# which wavefunctions should converge to zero at both far ends.
# Therefore, one needs to keep one end at zero and evolute the wavefunction to the other
# end by the Numerov algorithm. One can obtain the "trajectories" (the calculated 
# wavefunctions) for different given eigenvalues. The procedure is similar to shooting.
# Only the trajectories which converge to zero at both sides are the true wavefunctions.
# We call this method the "shooting method".

# ## **Numerov algorithm**
# 
# The time independent Schrödinger equation (TISE) is:   
# 
# $$\large \left[
#   -\dfrac{\hslash^2}{2m} \, \dfrac{\partial^2}{\partial x^{2}} + V\right] \psi(x) = E\psi(x) \quad (1)$$
#   
# We can rewrite the TISE as:  
# 
# $$\large  \dfrac{\partial^2}{\partial x^{2}} \psi(x) = -\dfrac{2m}{\hslash^2} \left[E-V\right]\psi(x) \quad (2)$$
# 
# For one dimensional system, the second-derivative can be evaluated numerically. 
# 
# $$\large \psi ''(x_{i})= \dfrac{1}{\delta x^2}\left[ \psi(x_{i+x})-2\psi(x_i)+\psi(x_{i-1}) \right] \quad (3)$$
# 
# One can also include the 4th derivative to increase the accuracy.
# 
# $$\large \psi ''(x_{i})= \dfrac{1}{\delta x^2}\left[ \psi(x_{i+x})-2\psi(x_i)+\psi(x_{i-1}) \right]- \dfrac{\delta x^2}{12} \psi(x_i)^{(4)} \quad (4)$$
# 
# Substituting equation 4 or 3 into equation 2, one can solve the eigenfunctions
# iteratively for any given eigenvalue E. However, the values of the first two 
# starting points are unknown. For the square well potential as shown below, 
# we can assume $\psi(x_0)$ is zero and $\psi(x_1)$ is a very small positive 
# (or negative) number.

# ## **Shooting method**
# 
# "In numerical analysis, the shooting method is a method for solving a boundary 
# value problem by reducing it to the system of an initial value problem. 
# Roughly speaking, we 'shoot' out trajectories in different directions until 
# we find a trajectory that has the desired boundary value." -- cite from <a href="https://en.wikipedia.org/wiki/Shooting_method">Wikipedia</a>.
# 
# When numerically solving the equation from left to right, we need to make 
# sure the boundary condition is also fulfilled at the right edge, which is
# $\psi(x_{+\infty})=0$. As mentioned above, one can try any value for the 
# eigenvalue E. But only the true eigenvalue will lead to the solved wavefunction
# converge to zero at the right side. Through scanning the eigenvalues and 
# monitoring the solved function on the right edge, we can approach the true
# eigenfunction and eigenvalue. This numerical method is called the shooting method.
#     
# <div class="container" style="text-align: center; width: 400px;">
#   <img src="../images/shooting_method.png" alt="Shooting method" class="image">
#   <div class="overlay">
#       Figure adported from book "Introduction to Quantum Mechanics with 
#       Applications to Chemistry (Linus Pauling and E. Bright Wilson)".</div>
# </div>

# In[ ]:




