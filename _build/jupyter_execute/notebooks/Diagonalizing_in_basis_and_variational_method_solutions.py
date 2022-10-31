#!/usr/bin/env python
# coding: utf-8

# # Diagonalizing in a basis and the variational method
# 
# In this notebook we cast our Hamiltonian into a discrete basis representation (we'll use harmonic oscillators) and diagonalize it for various basis sizes.
# 
# We'll see that this procedure is an implementation of the *variational* method, through which we get a good estimate of the ground-state energy (and also estimates of excited states) that is always an *upper bound* to the true ground-state energy.

# ## Solving the Schrodinger equation numerically
# 
# The abstract bound-state, time-independent Schrodinger equation,
# 
# $$
#   \hat H| \psi \rangle = E | \psi \rangle ,
# $$
# 
# can be solved in a variety of ways numerically.  Here is a partial laundry list:

# #### 1. Matrix diagonalization in coordinate representation.
# Here we use a finite difference formula for the second derivative in the kinetic energy operator (here with notation appropriate to the radial equation in spherical coordinates and $h \equiv \Delta x$):
# 
# $$
#    \frac{d^2u}{dr^2} = \frac{u(r+h) - 2 u(r) + u(r-h)}{h^2} + {\cal O}(h^2) .
# $$
# 
# You can verify this approximation by expanding $u(r+h)$ and $u(r-h)$ in
#  Taylor series about $u(r)$.
#  Let's suppose we solve this system knowing the \emph{boundary
#  conditions} at $r=0$ and $r = R_{\rm max}$.  The former is $u(0) = 0$, and we'll suppose $R_{\rm max}$
#  is large enough so that $u(R_{\rm max}) \approx 0$ for any bound
#  states. (This also makes the continuum solutions a discrete set of states.)  
#  We'll label the points:
# 
# $$
#    x_i = i\times h\ , \qquad i = 0,1,2,\cdots,N
# $$ 
#  
#  where $N$ is the number of steps and the step size $h$ is given by:
#  
# $$
#    h = \frac{R_{\rm max}}{N} .
# $$
# 
# Thus, $x_0 = 0$, $x_1 = h$, and so on up to $x_{N} = Nh = R_{\rm
#  max}$.  So we can approximate the Schr\"odinger equation at point $x_k$ as
# 
# $$
#   -\frac{\hbar^2}{2M}\frac{u(x_k+h)-2u(x_k) + u(x_k-h)}{h^2} 
#   + V(x_k)u(x_k) = E u(x_k) \ .
# $$
#  
#  If we work in units where $\hbar = 1$ and also $M=1/2$, and if we use the
#  notation:
#  
# $$
#    u_k \equiv u(x_k)\,, \quad u_{k\pm1}\equiv u(x_k\pm h)\,,
#    \quad V_k \equiv V(x_k)\ ,
# $$
#  
#  then the equation at $k$ takes the form
# 
# $$
#    -\frac{u_{k+1}-2u_k + u_{k-1}}{h^2} + V_k u_k = E u_k \;.
#   \label{eq:form}
# $$
#  
#  We know two values, $u_0 = 0$ and $u_{N}=0$.  We can
#  put the rest, $u_1$ to $u_{N-1}$ in a column vector, which
#  then satisfies a matrix eigenvalue problem:
# 
# $$
#    \left(
#      \begin{array}{ccccc}
#      \frac{2}{h^2}+V_1 & -\frac{1}{h^2} & 0 & \cdots & 0 \\
#      -\frac{1}{h^2} & \frac{2}{h^2}+V_2 &  -\frac{1}{h^2}    &        &  \vdots         \\
#      0 &    -\frac{1}{h^2}    & \ddots &        &  \vdots         \\
#      \vdots &        &        & \ddots       &  -\frac{1}{h^2}         \\
#     0 &  \cdots  &  \cdots  & -\frac{1}{h^2}  & \frac{2}{h^2}+V_{N-1}          
#      \end{array}
#    \right)
#    \left(
#     \begin{array}{c}
#       u_1 \\
#       u_2 \\
#       \vdots \\
#       \vdots \\
#       u_{N-1}
#     \end{array}
#    \right) 
#    = E
#    \left(
#     \begin{array}{c}
#       u_1 \\
#       u_2 \\
#       \vdots \\
#       \vdots \\
#       u_{N-1}
#     \end{array}
#    \right) 
# $$
# 
# This is a *tridiagonal* matrix with a simple structure: the only
# non-zero off-diagonal matrix elements are the ones adjacent to the
# diagonal, and they all have the same value, $-1/h^2$.   There are
# special algorithms that can rapidly find the eigenvalues and
# eigenvectors of such a matrix (although we have not taken advantage of these).  
# 
# The eigenvector from `scipy` or `numpy` diagonalization functions is normalized to unity; this means that $\sum_{i=0}^{N-1} |u_i|^2 = 1$. This is *not* the same as $\int_0^\infty \! |u(r)|^2\, dr = 1$, which is the quantum mechanics normalization condition; this is apparent if you plot the $\{u_i\}$'s for different mesh spacings (they will have different heights).  The former *would* be an approximation to the latter if we included a factor of the mesh spacing $h$, which is equivalent to using a simple integration rule.   So we can scale each $u_i$ by $\sqrt{h}$ to approximately normalize the wave function.

# #### 2. Solve as a differential equation in coordinate representation.
# This works when we have a *local* potential.  In one dimension, this means solving the equation:
# 
# $$
#     \left(-\frac{\hbar^2}{2M}\frac{d^2}{dx^2} + V(x) \right) 
#     \Psi_n(x) = E_n \Psi_n(x)
#     \ .
# $$
# 
# If we have a central potential in three dimensions, the potential is purely radial
# 
# $$
#     V({\bf x}) = V(|{\bf x}|) \equiv V(r)
#    \ ,
# $$
# 
# and we can use a partial wave decomposition (which means we 
#     separate the equation in spherical coordinates).  That is, we write 
# 
# $$
#    \Psi_{nlm}({\bf x}) = \frac{u_{nl}(r)}{r} Y_{lm}(\theta,\phi) \ ,
# $$
# 
# where $Y_{lm}$ is a spherical harmonic, and solve the radial one-dimensional Schrodinger equation:
# 
# $$
#    -\frac{\hbar^2}{2M}\frac{d^2u_{nl}(r)}{dr^2}
#    +  
#    \underbrace{\biggl[
#      V(r) + \frac{\hbar^2 l (l+1)}{2M r^2}
#                \biggr]}_{\equiv V_{\rm eff}(r)} 
#     u_{nl}(r)
#   = E_n u_{nl}(r)
#   \ ,
#   \label{eq:radSE}
# $$
# 
# with
# 
# $$
#   u_{nl}(r=0) = 0  \qquad \mbox{and} \qquad
#     \int_0^\infty |u_{nl}(r)|^2\, dr = 1 \ .
#     \label{eq:radbc}
# $$  
# 
# We'll apply this method soon for radial equations.
# 

# #### 3. Solve in momentum space
# 
# Here we can imitate the coordinate-space solution in momentum space with a diagonal kinetic energy and a discretized derivative replacing the functional $x$ or $r$ dependence in the potential.
# 
# A more common approach is to solve an integral equation in momentum representation.(We'll
#    come back to this later!).
# 

# #### 4. Introduce a (truncated) orthonormal basis and expand $u(r)$, then diagonalize the matrix of coefficients.
# 
# This applies similarly in one or three dimensions; we just change the range of integration when calculating matrix elements. (For three dimensions, we'll assume $l=0$ only for now.)
# 
# Imagine we have a set of basis functions:
# 
# $$
#     \{\phi_i(r)\}\,, \quad i=0,1,\cdots,D-1 ,
# $$
# 
# which we've truncated at $D$ states (since we can only use a finite
#   number in the computer), although in principle there are an infinite
#   number.  We can take the $\phi$'s to be real. (\emph{Why?})
#   Orthonormality means that
# 
# $$
#     \int_{-\infty}^\infty \phi_i(x) \phi_j(x)\, dx = \delta_{ij}
#       = \left\{ \begin{array}{ccc}
#                  1 & \mbox{if} & i=j\,, \\
#                  0 & \mbox{if} & i\neq j .
#                \end{array} \right.
# $$
# 
# in one dimension and
# 
# $$
#     \int_0^\infty \phi_i(r) \phi_j(r)\, dr = \delta_{ij}
#       = \left\{ \begin{array}{ccc}
#                  1 & \mbox{if} & i=j\,, \\
#                  0 & \mbox{if} & i\neq j .
#                \end{array} \right.
# $$
#  
# in three dimensions (radial equation). From here on we'll use the 3D notation; just remember for 1D to integrate $x$ from $-\infty$ to $+\infty$.
#  
# Then the expansion and coefficients are:
#   
# $$
#     u_n(r) \approx \sum_{i=0}^{D-1} C_i^{(n)} \phi_i(r)
#       \quad \Longrightarrow \quad
#       C_j^{(n)} = \int_0^\infty \phi_j(r) u_n(r)\, dr .
# $$
#   
#   (Can you derive the expression for $C_j^{(n)}$?)
#   If we substitute the expansion for $u_n(r)$ in the Schrodinger
#   equation, multiply by $\phi_i(r)$ and integrate
#   over $r$,
# 
# $$
#     \sum_{j=0}^{D-1}
#     \underbrace{
#     \int_0^\infty \phi_i(r) \left[ -\frac{\hbar^2}{2M}\frac{d^2}{dr^2} 
#        + V_{\rm eff}(r) \right] \phi_j(r)\, dr
#     }_{\equiv H_{ij}}
#     \cdot C_j^{(n)}
#     = E_n     \sum_{j=0}^{D-1} C_j^{(n)}\int_0^\infty \phi_i(r)\phi_j(r)\, dr
#     = E_n C_i^{(n)}
#      ,
# $$ 
# 
# or
#   
# $$
#     \sum_{j=0}^{D-1} H_{ij} C_j^{(n)} = E_n C_i^{(n)} .
# $$
#   
#   This is simply a matrix eigenvalue problem (take the time to make sure
#   you see that this is true!):
# 
# $$
#    \left(
#      \begin{array}{ccccc}
#      H_{00} & H_{01} & \cdots & \cdots & H_{0 D-1} \\
#      H_{10} & H_{11} &        &        &  \vdots         \\
#      \vdots &        & \ddots &        &  \vdots         \\
#      \vdots &        &        & \ddots       &  \vdots         \\
#      H_{D-1 0} &  \cdots  &  \cdots  & \cdots  & H_{D-1 D-1}          
#      \end{array}
#    \right)
#    \left(
#     \begin{array}{c}
#       C_0^{(n)} \\
#       \vdots \\
#       \vdots \\
#       \vdots \\
#       C_{D-1}^{(n)}
#     \end{array}
#    \right)
#    = E_n
#    \left(
#     \begin{array}{c}
#       C_0^{(n)} \\
#       \vdots \\
#       \vdots \\
#       \vdots \\
#       C_{D-1}^{(n)}
#     \end{array}
#    \right)
# $$
#  which we can give to a library routine (e.g., from `numpy` or `scipy`). 
#  
#  We will use harmonic oscillator radial wave functions as a
#  basis.  The potential for these wave functions is
#  
#  $$
#    V(r) = \frac{1}{2} M \omega^2 r^2 .
#  $$
#  
#  We define the *oscillator parameter* $b$ by
#  
#  $$
#     \hbar\omega = \frac{\hbar^2}{M b^2} ,
#  $$
#  
#  and use units in which $\hbar = 1$.  This means that $b$ sets the
#  length scale and $q \equiv r/b$ is the natural dimensionless
#  coordinate.  The oscillator state $u_{nl}(r)$
#  is specified by the radial quantum
#  number $n$ and the angular momentum quantum number $l$, with normalization
#  
#  $$
#    \int_0^\infty \! dr \, [u_{nl}(r)]^2 = 1 .
#  $$ 
#  
# The diagonalization of a Hamiltonian in a truncated basis can
#    be viewed as a *variational* calculation (we'll discuss this
#    further in future notes).  What are the
#    implications for:
# * What state (ground state or an excited state) is determined best?
# * How should the difference from the exact answer change as the basis size is increased?
#   
# 

# ####  Note on calculating matrix elements in a harmonic oscillator basis.
# 
# The $ij$ matrix element of the Hamiltonian with potential $V(r)$ is given by the integral
# 
# $$
#     H_{ij} = \int_0^\infty \phi_j(r)
#       \left[
#       -\frac{\hbar^2}{2M}\frac{d^2}{dr^2} + V(r)
#       \right]
#       \phi_i(r)\, dr \ ,
# $$
# 
# where $\phi_i$ and $\phi_j$ are harmonic oscillator basis wave
#   functions.  This is not the best thing to calculate numerically, 
#   because we would have to do numerical derivatives.  Instead, we can use
#   the fact that the $\{\phi_i\}$ satisfy a differential equation with
#   a second derivative ($l=0$ is assumed):
# 
# $$
# \left[
#       -\frac{\hbar^2}{2M}\frac{d^2}{dr^2} + \frac12 M\omega^2 r^2
#       \right]
#       \phi_i(r) = \hbar\omega\left(2i+\frac32\right)\phi_i(r) .
# $$
# 
# Thus, we can eliminate the second derivative to obtain
#    
# $$
#     H_{ij} = \int_0^\infty \phi_j(r)
#       \left[
#       \hbar\omega\left(2i+\frac32\right) - \frac12M\omega^2r^2 + V(r)
#       \right]
#       \phi_i(r)\, dr ,
# $$
# 
# which you can implement as a bonus problem.
# 

# ####   Diagonalization of a truncated basis as a variational problem.  
# 
# How might you analyze the eigenvalue program if you
#     didn't know the correct answer for the eigenvalue?  Instead of
#     looking for the lowest error, we could look for the most stable
#     region in $b$ or when the basis gets larger.  Are we guaranteed that
#     the estimate of the energy gets better as the basis size increases?
#     (Be careful:  remember we are doing our calculations on a computer,
#     where round-off errors are always waiting for us!)
#     In fact, the calculation we are doing is equivalent to a
#     *variational* estimate for the ground state. 
#     
# How does a variational calculation work?  If $u_{\rm trial}(r)$
#     is a (real) normalized trial wave function with parameter $b$
#     (e.g., $u_{\rm trial}(r) \propto r e^{-r^2/b^2}$),
#     then the estimate of the energy for that $b$ is:
# 
# $$
#     E(b) \equiv \langle u_{\rm trial} | H | u_{\rm trial} \rangle
#       = \int_0^\infty \! dr \, 
#       u_{\rm trial}(r) 
#       \left[
#       -\frac{\hbar^2}{2M}\frac{d^2}{dr^2} + V(r)
#       \right]
#       u_{\rm trial}(r) .
# $$
# 
# (What do we do if the trial wave function is *not* normalized? Hint: Divide by another integral.)
# The *best* estimate is $b_0$, where 
# 
# $$
#  \left.\frac{dE}{db}\right|_{b_0} = 0   ,
# $$
# 
# and $E(b_0)$ is an $upper bound$ to the true energy (that is,
# the actual energy is always lower, which usually means more
# negative).
#     
# So now suppose our trial wave function is
# a sum of $D$ basis functions with arbitary coefficients;
# 
# $$
# u_{\rm trial}(r) = \sum_{i=0}^{D-1} C_i \phi_i(r)  ,
# $$
# 
# where the $\{\phi_i(r)\}$ are a complete orthonormal basis
#     (e.g., our harmonic oscillator basis).  We want to minimize
#     $\langle u_{\rm trial} | H | u_{\rm trial} \rangle$ subject to
#     the constraint that $| u_{\rm trial} \rangle$ is normalized.
#     The $\{C_i\}$ are the variational parameters.  We use the method
#     of *Lagrange multipliers*.  Then for each $k$, we require
# 
# $$
#    \frac{\partial}{\partial C_k}
#       \left[
#        \langle u_{\rm trial} | H | u_{\rm trial} \rangle
#      - \lambda (\langle u_{\rm trial} | u_{\rm trial}\rangle - 1)       
#       \right]
#       = 0  ,
# $$
# 
# and $\partial/\partial\lambda [\cdots]= 0$, which gives $\sum_i |C_i|^2=1$.
#     Before tackling the general case, let's do the simplest non-trivial
#     special case: two basis states with coefficients $C_0$ and $C_1$.
#     The first condition is:
# 
# $$
# \frac{\partial}{\partial C_0}
#       \left[
#        C_0^2 \langle \phi_0 | H | \phi_0 \rangle +
#        C_0 C_1 \langle \phi_0 | H | \phi_1 \rangle +
#        C_1 C_0 \langle \phi_1 | H | \phi_0 \rangle +
#        C_1^2 \langle \phi_1 | H | \phi_1 \rangle 
#      - \lambda C_0^2 - \lambda C_1^2      
#       \right]
#       = 0 ,
# $$
# 
# or (using the fact that $H$ is Hermitian)
# 
# $$
#    2 C_0 \langle \phi_0 | H | \phi_0 \rangle
#       + 2 C_1  \langle \phi_0 | H | \phi_1 \rangle
#       - 2 \lambda C_0 = 0 , 
# $$
# 
# or switching notation to $\langle \phi_i | H | \phi_j \rangle
#     \equiv H_{ij}$ and dividing by 2:
# 
# $$    
#    C_0 H_{00} + C_1 H_{01} = \lambda C_0
#     \ .
# $$
# 
# The $\partial/\partial C_1$ contribution is similar; when we combine
#     them as a matrix equation, we find
# 
# $$
#     \left(
#        \begin{array}{cc}
#        H_{00} & H_{01} \\
#        H_{10} & H_{11}
#        \end{array}
#       \right)
#       \left(
#        \begin{array}{c}
#        C_{0} \\
#        C_{1}
#        \end{array}
#       \right)
#       = \lambda
#       \left(
#        \begin{array}{c}
#        C_{0}  \\
#        C_{1} 
#        \end{array}
#       \right)  ,
# $$
# 
# which is precisely our eigenvalue equation in the truncated basis! 
# Note that the Lagrange
# multiplier will be given by an energy eigenvalue. 
# 
# More generally, we find that we get the $k^{\rm th}$ row of the
#     eigenvalue matrix equation from 
# 
# 
# $$
#   \frac12\frac{\partial}{\partial C_k}
#       \left(
#       \sum_{ij}C_i C_j H_{ij} - \lambda \sum_{ij} C_i C_j \delta_{ij}
#       \right)
#       =
#       \sum_{j}H_{kj}C_j - \lambda C_k = 0  .
# $$

# ## Implementation of diagonalization in 1D harmonic oscillator basis

# In[1]:


import numpy as np
import scipy.linalg as la
from scipy.special import eval_hermite
from scipy.integrate import quad

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid"); sns.set_context("talk")


# In[2]:


# values of constants (choose natural units)
hbar = 1;
mass = 1/2;  # To match what we used in the 


# In[3]:


def V_linear_matrix(x_mesh):
    """
    Linear potential matrix (defined as a diagonal matrix)
    """
    k = 1/2       # k is chosen for simple units
    V_diag = k * np.abs(x_mesh)  # diagonal matrix elements
    N = len(x_mesh)  # number of x points
    
    return V_diag * np.diag(np.ones(N), 0) 


# In[4]:


def V_linear_potential(x, k=1/2):
    """
    Return the linear potential at point x 
    """
    return k * np.abs(x) 


# In[5]:


def SHO_1d_wf(n, x, omega, hbar=hbar, mass=mass):
    """
    Simple harmonic oscillator wave functions in one dimension.
    n --- principal quantum number (n = 0, 1, 2, ...)
    x --- point to evaluate at
    omega --- oscillator parameter
    """
    
    q = np.sqrt(mass * omega / hbar) * x
    norm = (mass * omega / (np.pi * hbar))**(1/4)
    factor = 1/np.sqrt(2**float(n) * np.math.factorial(n))
    
    return factor * norm * eval_hermite(n, q) * np.exp(-q**2/2)



# In[6]:


def sho_integrand(x, n1, n2, omega):
    """ 
    Integrand to check orthonormality using integration function: <psi_n1 | psi_n2>
    """
    return SHO_1d_wf(n1, x, omega) * SHO_1d_wf(n2, x, omega)


# **Checking whether the basis functions are orthonormal**

# In[7]:


omega = 0.5

print(f'Checking orthonormality for omega = {omega:.3f}')
print('n1 n2  <psi_n1 | psi_n2>')
for n1 in range(5):
    for n2 in range(5):
        psi_n1_psi_n2 = quad(sho_integrand, -np.inf, np.inf, args=(n1, n2, omega))[0]
        print(f' {n1}  {n2}    {psi_n1_psi_n2:.5e}')


# **Set up the Hamiltonian**

# In[8]:


def second_derivative_matrix(N, Delta_x):
    """
    Return an N x N matrix for 2nd derivative of a vector equally spaced by delta_x
    """
    M_temp = np.diag(np.ones(N-1), +1) + np.diag(np.ones(N-1), -1) \
              - 2 * np.diag(np.ones(N), 0)

    return M_temp / (Delta_x**2)


# In[9]:


N_pts = 4001  
x_min = -10.
x_max = 10.
Delta_x = (x_max - x_min) / (N_pts - 1)
x_mesh = np.linspace(x_min, x_max, N_pts)  # create the grid ("mesh") of x points

second_deriv = second_derivative_matrix(N_pts, Delta_x)     


# **Check orthonormality on grid**
# 
# Here we use a simple matrix multiplication (note the factor of $\Delta x$) and then check against a trapezoid rule for integration

# In[10]:


Nb = 8  # basis size
for n1 in range(Nb):
    wf1 = SHO_1d_wf(n1, x_mesh, omega)
    for n2 in range(Nb):
        wf2 = SHO_1d_wf(n2, x_mesh, omega)
        norm_ij = wf1 @ wf2 * Delta_x
        print(f'{norm_ij:.2e} ', end='')        
    print('')

print(' ')
for n1 in range(Nb):
    wf1 = SHO_1d_wf(n1, x_mesh, omega)
    for n2 in range(Nb):
        wf2 = SHO_1d_wf(n2, x_mesh, omega)
        norm_ij_trapz = np.trapz(wf1 * wf2, x_mesh)
        print(f'{norm_ij_trapz:.2e} ', end='')        
    print('')
   


# **Load the Hamiltonian matrix**
# 
# Each matrix element of the Hamiltonian is a calculated as a vector-matrix-vector product. This would be a double integral if written out with continuous (i.e., non-discrete) wave functions in coordinate space.

# In[61]:


N_basis = 1
omega = 2

# Combine matrices to make the Hamiltonian matrix
V_linear = V_linear_matrix(x_mesh)
Hamiltonian = np.zeros((N_basis,N_basis))  # Start with an Nb x Nb matrix of zeros
for n1 in range(N_basis):
    wf1 = SHO_1d_wf(n1, x_mesh, omega)  # this is the bra wave function
    for n2 in range(N_basis):
        wf2 = SHO_1d_wf(n2, x_mesh, omega) # this is the ket wave function
        Hamiltonian_matrix = -hbar**2 / (2 * mass) * second_deriv + V_linear 
        Hamiltonian[n1][n2] = Delta_x * wf1 @ Hamiltonian_matrix @ wf2


# In[62]:


# Try diagonalizing using numpy functions
eigvals, eigvecs = np.linalg.eigh(Hamiltonian)
print(eigvals[0:5])


# **Compare to exact answers**
# 
# These are reproduced following the discussion in S&N, but with more digits for the zeros. 

# In[63]:


ktest = 1/2
airy_zeros = np.array([-2.338, -4.088, -5.521])
airy_deriv_zeros = np.array([-1.018792971647471089017, -3.249, -4.820])
even_states_linear = -airy_deriv_zeros/2**(1/3) * (ktest**2/mass)**(1/3)
odd_states_linear = -airy_zeros/2**(1/3) * (ktest**2/mass)**(1/3)

num_eigs = 5
print(' n    exact   ho basis')
for i, E_approx in enumerate(eigvals[0:num_eigs]):
    if i % 2:  # odd states
        print(f' {i}  {odd_states_linear[int((i-1)/2)]:.6f}  {E_approx:.6f}')        
    else:      # even states
        print(f' {i}  {even_states_linear[int(i/2)]:.6f}  {E_approx:.6f}')
    


# print('  Even eigenvalues     Odd eigenvalues')
# print('  exact   ho basis     exact   ho basis')
# for i, (even, odd) in enumerate(zip(even_states_linear, odd_states_linear)):
#     print(f' {even:.5f}  {eigvals[2*i]:.5f}     {odd:.5f}  {eigvals[2*i+1]:.5f} ')


# ## Extensions

# In[15]:


def H_integrand(x, n1, n2, omega):
    """ 
    Integrand to check orthonormality: <psi_n1 | psi_n2>
    """
    return SHO_1d_wf(n1, x, omega) * SHO_1d_wf(n2, x, omega)

