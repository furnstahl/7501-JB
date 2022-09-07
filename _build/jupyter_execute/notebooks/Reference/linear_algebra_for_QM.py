#!/usr/bin/env python
# coding: utf-8

# # Linear algebra for quantum mechanics
# 
# Last revised: 06-Sep-2022 by Dick Furnstahl [furnstahl.1@osu.edu]
# 
# Here we do a selective (i.e., non-comprehensive) introduction to using Python for the type of linear algebra manipulation used in quantum mechanics (QM). We can do both numerical manipulations (with numpy) or symbolic manipulations (with sympy).

# ## If you are new to Jupyter notebooks
# 
# **You can find valuable documentation under the Jupyter notebook Help menu. The "User Interface Tour" and "Keyboard Shortcuts" are useful places to start, but there are also many other links to documentation there.** 
# 
# *Select "User Interface Tour" and use the arrow keys to step through the tour.*

# A Jupyter notebook is displayed on a web browser running on a computer, tablet (e.g., IPad), or even your smartphone.  The notebook is divided into *cells*, of which two types are relevant for us:
# * **Markdown cells:** These have headings, text, and mathematical formulas in $\LaTeX$ using a simple form of HTML called *markdown*.
# * **Code cells:** These have Python code (or other languages, but we'll stick to Python).
# 
# Either type of cell can be selected with your cursor and will be highlighted in color on the left when active.  You evaluate an active cell with shift-return (as with Mathematica) or by pressing `Run` on the notebook toolbar.  Some notes:
# * When a new cell is inserted, by default it is a Code cell and will have `In []:` to the left.  You can type Python expressions or entire programs in a cell.  How you break up code between cells is your choice and you can always put Markdown cells in between.  When you evaluate a cell it advances to the next number, e.g., `In [5]:`.
# * On the notebook menu bar is a pulldown menu that lets you change back and forth between Code and Markdown cells.  Once you evaluate a Markdown cell, it gets formatted (and has a blue border).  To edit the Markdown cell, double click in it. 
# 
# **Try double-clicking on this cell and then shift-return.**  You will see that a bullet list is created just with an asterisk and a space at the beginning of lines (without the space you get *italics* and with two asterisks you get **bold**).  **Double click on the title header above and you'll see it starts with a single #.**  Headings of subsections are made using ## or ###.  See this [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for a quick tour of the Markdown language (including how to add links!).
# 
# **Now try turning the next (empty) cell to a Markdown cell and type:** `Einstein says $E=mc^2$` **and then evaluate it.**  This is $\LaTeX$! (If you forget to convert to Markdown and get `SyntaxError: invalid syntax`, just select the cell and convert to Markdown with the menu.)

# ## A whirlwind tour of Python for linear algebra

# In[52]:


import numpy as np
import scipy.linalg as LA


# ### Inputting vectors and matrices
# 
# Both vectors and matrices will be entered as particular Python list objects known as numpy arrays. Vectors appear in square brackets and matrices in nested square brackets; we create them as arguments to `np.array`.

# In[9]:


x = np.array([0, 1, 2, 3])


# In[10]:


print(x)


# We can pick out elements of the vector (note the square brackets). If there are `n` elements of the vector, they are indexed from `0` to `n-1`. **Why does the last reference fail here?** 

# In[11]:


print(x[0])
print(x[3])
print(x[4])


# In[15]:


ident2 = np.eye(2)   # we can name these whatever we want
my_identity3 = np.eye(3)


# In[20]:


print(ident2)
print('\n')  # skip a line; \n means "newline"
print(my_identity3)


# If we use the `shape` attribute we would get $(3, 3)$ as output, that is verifying that our matrix is a $3\times 3$ matrix. 

# In[21]:


my_identity3.shape


# ### Pauli matrix examples

# In[ ]:


eye2 = np.eye(2)   # 2x2 identify
sig1 = np.array([[0, 1], [1, 0]]) # \sigma_1
sig2 = np.array([[0, j], [j, 0]]) # use j for sqrt(-1)
sig3 = np.array([[1, 0], [0, -1]])


# In[ ]:





# In[ ]:





# ### $3 \times 3$ matrix examples

# In[31]:


Sx = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
Sy = 1/np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])  # note we use "1j" instead of just "j"


# In[30]:


print(Sx)
print('\n')
print(Sy)


# In[33]:


Sx @ Sy - Sy @ Sx


# In[36]:


eigvals, eigvecs = np.linalg.eigh(Sx)
print(eigvals)
print('\n')
print(eigvecs)


# In[69]:


eigvals, eigvecs = np.linalg.eigh(np.sqrt(2)*Sx)
print(eigvals)
print('\n')
print(eigvecs)


# In[68]:


np.set_printoptions(precision=3, suppress=True)

eigvals, eigvecs = np.linalg.eigh(Sy)
print(eigvals)
print('\n')
print(eigvecs)


# In[70]:


np.set_printoptions(precision=3, suppress=True)

eigvals, eigvecs = np.linalg.eigh(np.sqrt(2)*Sy)
print(eigvals)
print('\n')
print(eigvecs)


# The formatting is tricky because it is `0.5 + 0j`.

# In[63]:


eigvals, eigvecs = LA.eigh(Sy)
print(eigvals)
print('\n')
print(eigvecs)


# In[50]:


Sx - Sx.conjugate().T


# In[51]:


Sy - Sy.conjugate().T


# In[74]:


Sz = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])


# In[76]:


1/2 * (Sx@Sx + Sy@Sy + Sz@Sz)


# ## Symbolic linear algebra

# In[22]:


from sympy.physics.matrices import msigma


# In[23]:


msigma(1)


# In[ ]:




