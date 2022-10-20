#!/usr/bin/env python
# coding: utf-8

# In[12]:


def V_quartic_matrix(x_mesh):
    """
    Quartic potential matrix (defined as a diagonal matrix)
    """
    alpha = 4
    k = 1/alpha       # k is chosen for simple units
    V_diag = k * x_mesh**alpha  # diagonal matrix elements
    N = len(x_mesh)  # number of x points
    
    return V_diag * np.diag(np.ones(N), 0) 


# In[16]:


V_quartic = V_quartic_matrix(x_mesh)
Hamiltonian = -second_deriv + V_quartic
eigvals, eigvecs = np.linalg.eigh(Hamiltonian)


# In[18]:


print(eigvals[0:10])


# In[19]:


n_vals = range(1,100)
scaled_E = [eigvals[n-1]/n**(4/3) for n in n_vals]
scaled_E2 = [eigvals[n-1]/n for n in n_vals]
scaled_E3 = [eigvals[n-1]/n**1.5 for n in n_vals]


# In[20]:


fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(1,2,1)
ax1.set_xlabel(r'$n$')
ax1.set_ylabel(r'$E_n / n^{\gamma}$')
#ax1.set_xlim(0, x_max)
#ax1.set_ylim(-1., 3)

ax1.plot(n_vals, scaled_E, color='red', label=r'$\gamma = 4/3$')
ax1.plot(n_vals, scaled_E2, color='blue', label=r'$\gamma = 1.5$')
ax1.plot(n_vals, scaled_E3, color='green', label=r'$\gamma = 1$')
ax1.set_title('Scaling of quartic potential energies')

ax1.legend();

