#!/usr/bin/env python
# coding: utf-8

# # Diagonalizing coordinate space solutions
# 
# In this notebook you will apply a 2nd-derivative operator and a potential as matrices in coordinate space to represent the Schroedinger equation. By diagonalizing it you'll find its eigenvalues (the energy spectrum) and eigenvectors (wave functions). 

# Standard imports plus seaborn (to make plots looks nicer).

# In[ ]:


import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid"); sns.set_context("talk")


# Fill in the ?'s in the  `second_derivative_matrix` function below so that it returns a matrix that implements an approximate second derivative when applied to a vector made up of a function evaluated at the mesh points. The numpy `diag` and `ones` functions are used to create matrices with 1's on particular diagonals, as in this $5\times 5$ example: 
# 
# $$ \frac{1}{(\Delta x)^2}\,\left( 
#     \begin{array}{ccccc}
#     -2 & 1 & 0 & 0 & 0 \\
#     1 & -2 & 1 & 0 & 0 \\
#     0 & 1 & -2 & 1 & 0 \\
#     0 & 0 &1 & -2 & 1 \\
#     0 & 0 & 0 & 1 & -2
#     \end{array}
#    \right) 
#    \left(\begin{array}{c}
#          f_1 \\ f_2 \\ f_3 \\ f_4 \\ f_5
#          \end{array}
#    \right) 
#    \overset{?}{=}
#    \left(\begin{array}{c}
#          ? \\ ? \\ ? \\ ? \\ ?
#          \end{array}
#    \right) 
#  $$  
#  

#  **Replace the ?'s with appropriate numbers in the `second_derivative_matrix` function.**

# In[ ]:


def second_derivative_matrix(N, Delta_x):
    """
    Return an N x N matrix for 2nd derivative of a vector equally spaced by delta_x.
    """
    M_temp = ? * np.diag(np.ones(N-1), +1) + ? * np.diag(np.ones(N-1), -1) + ? * np.diag(np.ones(N), 0)

    return M_temp / (Delta_x**2)


# ## Testing the second derivative
# 
# Check the relative accuracy of the approximate second derivative at a fixed $\Delta x$ by choosing a test function $f(x)$ and a range of $x$. 
# 
# **Choose appropriate values for `N_pts`, `x_min`, and `x_max`. You won't know what to choose at first, so try some values and come back later to update your choices as needed.**

# In[ ]:


N_pts = ?   # number of x points
x_min = ?   # minimum x value (should be negative)
x_max = ?   # maximum x value (should be positive)

Delta_x = (x_max - x_min) / (N_pts - 1)    # calculate Delta x based on the range and number of points
x_mesh = np.linspace(x_min, x_max, N_pts)  # create the grid ("mesh") of x points


# In[ ]:


# Verify that mesh is consistent with Delta_x
print(Delta_x)
print(x_mesh)


# Set up the derivative matrices for the specified mesh.

# In[ ]:


second_deriv = second_derivative_matrix(N_pts, Delta_x)


# ### Set up various test functions
# 
# Only one test function is defined initially, a Gaussian.

# In[ ]:


def f_test_0(x_mesh):
    """
    Return the value of the function e^{-x^2} and its 2nd derivative
    """
    return ( np.exp(-x_mesh**2), np.exp(-x_mesh**2) * (4 * x_mesh**2 - 2) )    


# Pick a test functions and evaluate the function and its derivative on the mesh.
# Then apply the second derivative (`second_deriv`) matrix to the `f_test` vector (using the `@` symbol for matrix-vector, matrix-matrix, and vector-vector multiplication).

# In[ ]:


f_test, f_2nd_deriv_exact = f_test_0(x_mesh)

f_2nd_deriv = second_deriv @ f_test  # create the array of 2nd derivative values on the x mesh


# Make plots comparing the exact to approximate derivative and then the relative errors.

# In[ ]:


def rel_error(x1, x2):
    """
    Calculate the (absolute value of the) relative error between x1 and x2
    """
    return np.abs( (x1 - x2) / ((x1 + x2)/2) )
    #return np.abs( (x1 - x2)  )


# In[ ]:


# Comparison plots
fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(1,2,1)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$df/dx$')

ax1.plot(x_mesh, f_2nd_deriv_exact, color='red', label='exact 2nd derivative')
ax1.plot(x_mesh, f_2nd_deriv, color='blue', label='approx 2nd derivative', linestyle='dashed')

ax1.legend()

ax2 = fig.add_subplot(1,2,2)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'relative error')
ax2.set_xlim(0, x_max)
ax2.set_ylim(1e-6, 2)

# Calculate relative errors
rel_error_2nd_deriv = rel_error(f_2nd_deriv_exact, f_2nd_deriv)

ax2.semilogy(x_mesh, rel_error_2nd_deriv, color='blue', label='2nd derivative', linestyle='dashed')

ax2.legend()

fig.tight_layout()


# ## Harmonic oscillator 
# 
# The Hamiltonian matrix is 
# 
# $$
#  \hat H \doteq  -\frac{\hbar^2}{2m}\frac{d^2}{dx^2} + V(x) ,
# $$
# 
# which we'll implement as a sum of matrices. We'll choose units so that $\hbar^2/2m = 1$ and $\hbar\omega = 1$.

# In[ ]:


def V_SHO_matrix(x_mesh):
    """
    Harmonic oscillator potential matrix (defined as a diagonal matrix)
    """
    k = 1/2       # k is chosen so that hbar*omega = 1 
    V_diag = k * x_mesh**2 / 2  # diagonal matrix elements
    N = len(x_mesh)  # number of x points
    
    return V_diag * np.diag(np.ones(N), 0) 


# In[ ]:


# Combine matrices to make the Hamiltonian matrix
V_SHO = V_SHO_matrix(x_mesh)

Hamiltonian = -second_deriv + V_SHO  


# In[ ]:


# Try diagonalizing using numpy functions
eigvals, eigvecs = np.linalg.eigh(Hamiltonian)


# In[ ]:


print(eigvals[0:10])  # print the first 10 eigenvalues


# **If the eigenvalues are not very close (better than 0.1%, say) to those expected from a harmonic oscillator with $\hbar\omega = 1$, go back and adjust `N_pts`, `x_min`, and `x_max`.**

# In[ ]:


# Extract the first three eigenvectors (wave functions)
wf_0 = eigvecs[:,0]
wf_1 = eigvecs[:,1]
wf_2 = eigvecs[:,2]


# In[ ]:


# Plot the wave functions
fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(1,2,1)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\psi_n(x)$')

ax1.plot(x_mesh, wf_0, color='red', label=r'$n=0$')
ax1.plot(x_mesh, wf_1, color='blue', label=r'$n=1$')
ax1.plot(x_mesh, wf_2, color='green', label=r'$n=2$')

ax1.legend();

