#!/usr/bin/env python
# coding: utf-8

# In[29]:


def get_eigvals(ell, beta, num_eigs=5):
    """
    Get the first num_eigs eigenvalues for specified ell and beta 
    """
    hbar = 1
    mass = 1
    a = 1

    V_spherical = Three_D_spherical_box(r_mesh, ell, a, beta, 
                                        mass=mass, hbar=hbar)
    Hamiltonian = -hbar**2/(2*mass) * second_deriv + V_spherical
    # Diagonalize using numpy functions
    eigvals, eigvecs = np.linalg.eigh(Hamiltonian)

    return eigvals[0:num_eigs]


# In[32]:


betas = np.array([4, 10, 25, 100, 1000])
num_eigs = 5

for ell in [0, 1]:
    print(f' orbital angular momentum l = {ell}')
    print(f' beta   {num_eigs} * [eigenvalue   sqrt(eig*2ma^2/hbar^2)]')
    for beta in betas:
        print(f' {beta:4.0f} ', end =" ")
        eigs = get_eigvals(ell, beta, num_eigs)
        for i in range(num_eigs):
            print(f'{eigs[i]:7.3f} ', \
                  f'{np.sqrt(eigs[i]*2*mass*a**2/hbar**2):5.3f}', 
                  end =" ")
        print(' ')    
    print(' ')

