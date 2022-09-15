#!/usr/bin/env python
# coding: utf-8

# # Numerical test of $e^{-x} \approx (1 -\frac{x}{N})^N$ for large $N$
# 
# In this notebook we test:
# 1. For what $x$ does this work?
# 2. How big does $N$ have to be?
# 
# Plan: calculate the *error* for a range of $x$ and $N$.

# Standard Python imports plus seaborn (to make plots looks nicer).

# In[1]:


import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid"); sns.set_context("talk")


# Set up the $x$ and $N$ arrays.

# In[2]:


delta_x = 0.1
x_values = np.arange(0, 2+delta_x, delta_x)  # Mesh points rom 0 to 2 inclusive.
N_values = [10, 100]  # You can add more to this list


# In[3]:


print(x_values)  # just to check they are what we want


# In[4]:


print(N_values)


# Write functions to evaluate the approximation and for relative errors

# In[5]:


def rel_error(x1, x2):
    """
    Calculate the (absolute value of the) relative error between x1 and x2
    """
    return np.abs( (x1 - x2) / ((x1 + x2)/2) )

def exp_approx(z, N):
    """
    Calculate (1 + z/N)^N
    """
    return (1 + z/N)**N


# Step through $x$ array and for each $x$ step through $N$, making a table of results

# In[6]:


print(' x   exp(-x)  N: ', end=" ")   # The end=" " option suppresses a return.
for N in N_values:
   print(f'  {N}     ', end=" ")
print('\n')

for x in x_values:
    f_of_x = np.exp(-x)
    print(f"{x:.1f}   {f_of_x:.3f}   ", end=" ")
    for N in N_values:
        approx = exp_approx(-x, N)
        print(f"  {rel_error(f_of_x, approx):.2e}", end=" ")
    print(" ")


# ## Things to do
# 
# 1. How does the relative error scales with $N$? (Note that you can add additional $N$ values to `N_values`.) Can you explain the observed scaling?
# 2. Investigate how well the approximation works for $x > 2$.

# In[ ]:




