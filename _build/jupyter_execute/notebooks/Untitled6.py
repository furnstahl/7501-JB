#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Half_V_SHO_matrix(x_mesh):
    """
    Harmonic oscillator potential matrix (defined as a diagonal matrix)
    """
    k = 1/2       # k is chosen so that hbar*omega = 1 
    inf = 1.e10
    #V_diag = k * x_mesh**2 / 2  # diagonal matrix elements
    V_diag = [k * x**2 / 2 if x > 0 else inf for x in x_mesh]  # using a list comprehension
    N = len(x_mesh)  # number of x points
    
    return V_diag * np.diag(np.ones(N), 0) 


# In[ ]:


def xsq_matrix(x_mesh):
    """
    matrix for x^2 operator
    """
    N = len(x_mesh)  # number of x points

    return x_mesh**2 * np.diag(np.ones(N), 0) 


# In[ ]:


# Combine matrices to make the Hamiltonian matrix
V_SHO = V_SHO_matrix(x_mesh)
V_SHO_half = Half_V_SHO_matrix(x_mesh)

Hamiltonian = -second_deriv + V_SHO  
Hamiltonian_half = -second_deriv + V_SHO_half  


# In[ ]:


# Try diagonalizing using numpy functions
eigvals, eigvecs = np.linalg.eigh(Hamiltonian)
eigvals_half, eigvecs_half = np.linalg.eigh(Hamiltonian_half)


# In[ ]:


print(eigvals_half[0:10])


# In[ ]:


wf_0 = eigvecs[:,0]
wf_1 = eigvecs[:,1]
wf_2 = eigvecs[:,2]

wf_half_0 = eigvecs_half[:,0]
wf_half_1 = eigvecs_half[:,1]
wf_half_2 = eigvecs_half[:,2]


# In[ ]:




