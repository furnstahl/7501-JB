#!/usr/bin/env python
# coding: utf-8

# ## Reproduce some results from Problem Set 2, problem 5 
# 
# In problem 5 on problem set 2 you looked at the matrix representations of two operators $\hat A$ and $\hat B$. Takes the real numbers $a$ and $b$ to be 1 and 2, respectively.
# 
# a. Input the matrices $A$ and $B$.
# b. Show that they commute.
# c. Find eigenvalues and eigenvectors of each. Do the eigenvectors correspond to simultaneous eigenkets?

# In[32]:


# a. Input the matrices A and B
a = 1; b = 2
A = np.array([[a, 0, 0], [0, -a, 0], [0, 0, -a]])
print(A)
print(" ")
B = np.array([[b, 0, 0], [0, 0, -1j*b], [0, 1j*b, 0]])
print(B)


# In[33]:


# b. Show that they commute.
my_commute(A,B)  # calculate [A,B]


# The matrix is all zeros, so they commute.

# In[34]:


# c. Find the eigenvalues and eigenvectors of A and B
eigvals_A, eigvecs_A = np.linalg.eigh(A)
eigvals_B, eigvecs_B = np.linalg.eigh(B)


# In[35]:


# Print the eigenvalues
print(eigvals_A, ' ', eigvals_B)


# In[36]:


# Print the matrices of eigenvectors, with a blank line between. 
#  The eigenvectors are read vertically, in the same order as the eigenvalues.
print(eigvecs_A, '\n\n', eigvecs_B)


# Only one of the eigenvectors is the same, so they are not simultaneous eigenkets automatically (we need to consider linear combinations).

# In[ ]:




