#!/usr/bin/env python
# coding: utf-8

# # Numerical test of $e^{-x} \approx (1 -\frac{x}{N})^N$ for large $N$
# 
# In this notebook we test whether
# 1. For what $x$ does this work?
# 2. How big does $N$ have to be?
# 
# Plan: calculate the *error* for a range of $x$ and $N$.

# Standard imports plus seaborn (to make plots looks nicer).

# In[ ]:


import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style("darkgrid"); sns.set_context("talk")


# Set up the $x$ and $N$ arrays.

# In[ ]:


x_values = np.arange(0, 2, .1)
N_values = [10, 100]


# In[ ]:


print(x_values)


# In[ ]:


print(N_values)


# Write functions to evaluate the approximation and for relative errors

# In[ ]:


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

# In[ ]:


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
    print("\n")


# In[ ]:




