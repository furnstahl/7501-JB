#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def second_derivative_matrix(N, Delta_x):
    """
    Return an N x N matrix for 2nd derivative of a vector equally spaced by delta_x
    """
    M_temp = np.diag(np.ones(N-1), +1) + np.diag(np.ones(N-1), -1) \
              - 2 * np.diag(np.ones(N), 0)

    return M_temp / (Delta_x**2)


# ## Testing second derivative
# 
# We'll check the relative accuracy of the approximate second derivative at a fixed $\Delta x$ by choosing a test function $f(x)$ and a range of $x$. 
# 
# **Choose values for `N_pts`, `x_min`, and `x_max` 

# In[ ]:


N_pts = 801  
x_min = -8.
x_min = 0.
x_max = 8.
Delta_x = (x_max - x_min) / (N_pts - 1)
x_mesh = np.linspace(x_min, x_max, N_pts)  # create the grid ("mesh") of x points


# In[ ]:




