#!/usr/bin/env python
# coding: utf-8

# # Calculate zeros of the Airy function and its derivative
# 
# `scipy` has a built-in Airy function that also returns the derivative (`airy`). 
# 
# There are multiple root solving functions in `scipy`. The general multi-dimensional version is `root`.

# In[1]:


import numpy as np
from scipy.special import airy
from scipy.optimize import root
import matplotlib.pyplot as plt


# In[2]:


def Airy(x, index):
    """
    Airy function. Returns Ai, Ai', Bi, Bi according to index = 0,1,2,3'
    """
    ans = airy(x)
    return ans[index]


# In[3]:


x_mesh = np.arange(-10,10,.01)

fig = plt.figure(figsize=(16,6))

ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$f(x)$')
ax1.set_xlim(-9., 5.)
#ax1.set_ylim(-1., 3)
ax1.axhline(0, color='gray', alpha=0.4)

ax1.plot(x_mesh, Airy(x_mesh, 0), color='red', label=r'Ai(x)')
ax1.plot(x_mesh, Airy(x_mesh, 1), color='blue', label=r"Ai'(x)")
ax1.set_title('Airy function and derivative')
ax1.legend();


# In[4]:


kval = 1/2
const = (2*kval**2/2)**(1/3)

print("guess   Ai zero    odd energy     Ai' zero   even energy")
for x0 in range(0, -10, -1):
    ai_zero_sol = root(Airy, x0, args=0)
    ai_zero = ai_zero_sol.x[0]
    
    ai_prime_zero_sol = root(Airy, x0, args=1)
    ai_prime_zero = ai_prime_zero_sol.x[0]
    print(f' {x0: d}    {ai_zero:.5f}  {-ai_zero*const: .10f}     {ai_prime_zero:.5f}    {-ai_prime_zero*const: .10f}')


# To do: step through and pick out unique energies and throw out negative results (false zeros).

# In[ ]:




