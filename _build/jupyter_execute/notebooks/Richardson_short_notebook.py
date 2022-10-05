#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def Richardson_derivative_matrix(N, Delta_x):
    """Return an N x N matrix for derivative of an equally spaced vector by delta_x
    """
    sym_matrix_delta_x = symmetric_derivative_matrix(N, Delta_x)
    sym_matrix_2_delta_x = ( np.diag(np.ones(N-2), +2) - np.diag(np.ones(N-2), -2) ) / (4 * Delta_x)
    return (4 * sym_matrix_delta_x - sym_matrix_2_delta_x) / 3


# In[25]:


fig_new = plt.figure(figsize=(12,9))

ax1 = fig_new.add_subplot(1,1,1)
ax1.set_xlabel(r'$\Delta x$')
ax1.set_ylabel(r'relative error')
#ax1.set_xlim(0, x_max)
#ax1.set_ylim(-1., 3)

ax1.loglog(Delta_x_array, rel_error_fd_array, color='red', label='forward derivative')
ax1.loglog(Delta_x_array, rel_error_sd_array, color='blue', label='symmetric derivative')
ax1.loglog(Delta_x_array, rel_error_rd_array, color='green', label='Richardson')

ax1.legend();

