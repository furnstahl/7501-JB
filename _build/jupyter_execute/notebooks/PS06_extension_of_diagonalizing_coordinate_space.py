#!/usr/bin/env python
# coding: utf-8

# In[10]:


def V_SHO_matrix(x_mesh):
    """
    Harmonic oscillator potential matrix (defined as a diagonal matrix)
    """
    k = 1/2       # k is chosen so that hbar*omega = 1 
    V_diag = k * x_mesh**2 / 2  # diagonal matrix elements
    N = len(x_mesh)  # number of x points
    
    return V_diag * np.diag(np.ones(N), 0) 


# In[11]:


def xsq_matrix(x_mesh):
    """
    matrix for x^2 operator
    """
    N = len(x_mesh)  # number of x points

    return x_mesh**2 * np.diag(np.ones(N), 0) 


# In[12]:


def eikx_matrix(x_mesh, k):
    """
    matrix for e^{ikx} operator
    """
    N = len(x_mesh)  # number of x points

    return np.exp(1j * k * x_mesh) * np.diag(np.ones(N), 0) 


# In[13]:


# Combine matrices to make the Hamiltonian matrix
V_SHO = V_SHO_matrix(x_mesh)

Hamiltonian = -second_deriv + V_SHO  


# In[14]:


# Try diagonalizing using numpy functions
eigvals, eigvecs = np.linalg.eigh(Hamiltonian)


# In[15]:


print(eigvals[0:10])


# Notice that they are all *above* the exact answer. Variational principle!

# In[16]:


wf_0 = eigvecs[:,0]
wf_1 = eigvecs[:,1]
wf_2 = eigvecs[:,2]


# In[18]:


fig_new = plt.figure(figsize=(16,6))

ax1 = fig_new.add_subplot(1,2,1)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\psi_n(x)$')
#ax1.set_xlim(0, x_max)
#ax1.set_ylim(-1., 3)

ax1.plot(x_mesh, wf_0, color='red', label=r'$n=0$')
ax1.plot(x_mesh, wf_1, color='blue', label=r'$n=1$')
ax1.plot(x_mesh, wf_2, color='green', label=r'$n=2$')

ax1.legend();


# In[20]:


wf_0 @ wf_0  # Normalized to 1


# In[21]:


wf_0 @ Hamiltonian @ wf_0  # check that expectation value of H is 0.5


# In[22]:


xsq_exp_val = wf_0 @ xsq_matrix(x_mesh) @ wf_0


# In[23]:


eikx_exp_val = wf_0 @ eikx_matrix(x_mesh, ktest) @ wf_0


# In[26]:


print('  k             <e^{ikx}>          e^{-k^2<x^2>/2}     rel. error ')
for k in np.arange(0, 3.2, .2):
    lhs = wf_0 @ eikx_matrix(x_mesh, k) @ wf_0 
    rhs = np.exp(-k**2 * xsq_exp_val / 2.)
    print(f' {k:.2f}  {lhs:.10f}   {rhs:.10f}      {rel_error(np.real(lhs), rhs):.5e}')

