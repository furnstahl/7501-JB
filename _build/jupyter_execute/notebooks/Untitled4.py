#!/usr/bin/env python
# coding: utf-8

# In[2]:


from scipy.special import airy
from scipy.optimize import root


# In[7]:


def Airy(x, index):
    """
    Airy function. Returns Ai, Ai', Bi, Bi'
    """
    ans = airy(x)
    return ans[index]


# In[18]:


z_sol = root(Airy, -1, args=1)
print(f'{z_sol.x[0]:.10f}')


# In[9]:


airy(-1.019)


# In[ ]:




