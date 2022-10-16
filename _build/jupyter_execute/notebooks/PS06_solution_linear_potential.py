#!/usr/bin/env python
# coding: utf-8

# In[11]:


def V_linear_matrix(x_mesh):
    """
    Linear potential matrix (defined as a diagonal matrix)
    """
    k = 1/2       # k is chosen for simple units
    V_diag = k * np.abs(x_mesh)  # diagonal matrix elements
    N = len(x_mesh)  # number of x points
    
    return V_diag * np.diag(np.ones(N), 0) 


# In[15]:


# Combine matrices to make the Hamiltonian matrix
V_linear = V_linear_matrix(x_mesh)

Hamiltonian = -second_deriv + V_linear 


# In[16]:


# Try diagonalizing using numpy functions
eigvals, eigvecs = np.linalg.eigh(Hamiltonian)


# In[17]:


print(eigvals[0:10])


# In[18]:


ktest = 1/2
airy_deriv_zero = -1.018792971647471089017
print(-airy_deriv_zero/2**(1/3) * (2*ktest**2)**(1/3))


# The relative accuracy is about $10^{-6}$ for the current choice of bounds and $\Delta x$.

# In[19]:


wf_0 = eigvecs[:,0]
wf_1 = eigvecs[:,1]
wf_2 = eigvecs[:,2]


# In[31]:


fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(1,2,1)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\psi_n(x)$')
#ax1.set_xlim(0, x_max)
#ax1.set_ylim(-1., 3)

ax1.plot(x_mesh, wf_0, color='red', label=r'$n=0$')
ax1.plot(x_mesh, wf_1, color='blue', label=r'$n=1$')
ax1.plot(x_mesh, wf_2, color='green', label=r'$n=2$')

ax1.legend();


# In[ ]:




