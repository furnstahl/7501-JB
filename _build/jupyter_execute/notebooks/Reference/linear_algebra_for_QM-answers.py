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
# **Try double-clicking on this cell (the one you are reading) and then shift-return.**  You will see that a bullet list is created just with an asterisk and a space at the beginning of lines (without the space you get *italics* and with two asterisks you get **bold**).  **Double click on the title header above and you'll see it starts with a single #.**  Headings of subsections are made using ## or ###.  See this [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for a quick tour of the Markdown language (including how to add links!).
# 
# **Now try turning the next (empty) cell to a Markdown cell and type:** `Einstein says $E=mc^2$` **and then evaluate it.**  This is $\LaTeX$! (If you forget to convert to Markdown and get `SyntaxError: invalid syntax`, just select the cell and convert to Markdown with the menu.)

# In[ ]:





# ## A whirlwind tour of Python for linear algebra

# Start by importing the library modules we'll need, defining standard abbreviations (`np` for numpy and `LA` for scipy.linalg). Google `numpy` or `scipy` with specific topics mentioned (e.g., "arrays") to learn much more.  

# In[1]:


import numpy as np
import scipy.linalg as LA


# ### Inputting vectors and matrices
# 
# Both vectors and matrices will be entered as particular Python list objects known as numpy arrays. Vectors appear in square brackets and matrices in nested square brackets; we create them as arguments to `np.array`.

# In[2]:


x = np.array([0, 1, 2, 3])


# In[3]:


print(x)


# We can pick out elements of the vector (note the square brackets). If there are `n` elements of the vector, they are indexed from `0` to `n-1`. **Why does the last reference fail here with an error?** 

# In[4]:


print(x[0])
print(x[3])
print(x[4])


# Functions are defined to generate matrices such as identity matrices of any dimension (`np.eye`).

# In[5]:


ident2 = np.eye(2)   # we can name these whatever we want
my_identity3 = np.eye(3)


# In[6]:


print(ident2)
print('\n')  # skip a line; \n means "newline"
print(my_identity3)


# If we use the `shape` attribute we would get $(3, 3)$ as output, that is verifying that our matrix is a $3\times 3$ matrix. 

# In[7]:


my_identity3.shape


# ### Pauli matrix examples

# Here we see how to use np.array to input a matrix. Note the use of `1j` for $\sqrt{-1}$.

# In[8]:


eye2 = np.eye(2)   # 2x2 identity
sig1 = np.array([[0, 1], [1, 0]]) # \sigma_1
sig2 = np.array([[0, -1j], [1j, 0]]) # use j for sqrt(-1)
sig3 = np.array([[1, 0], [0, -1]])


# In[9]:


print(eye2, '\n\n', sig1, '\n\n', sig2, '\n\n', sig3)


# Let's do some first matrix multiplications. In all cases we use `@` to multiply matrices and/or vectors. Check some standard results. **You fill in the rest!**

# In[10]:


sig1 @ sig1


# In[11]:


print(sig1 @ sig2)
print(" ")
print(1j * sig3)


# In[12]:


def my_commute(A, B):
    """
    Define our own commutator function.
    """
    return A @ B - B @ A


# In[13]:


my_commute(sig1, sig2)


# Are the results correct?

# ### $3 \times 3$ matrix examples

# Here are the $S_x$ and $S_y$ matrices for spin-1 (in units where $\hbar = 1$).

# In[14]:


Sx = 1/np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
Sy = 1/np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])  # note we use "1j" instead of just "j"


# In[15]:


print(Sx, '\n\n', Sy)   # \n in quotes will start a new line


# Let's do more matrix multipications. In all cases we use `@` to multiply matrices and/or vectors.

# In[16]:


Sx @ Sy - Sy @ Sx   


# Now let's get the eigenvalues (`eigvals` here) and eigenvectors (`eigvecs` here). The function `eigh` is for (complex) Hermitian matrices. The eigenvectors should be read as columns: the first element in the eigenvalue list corresponds to the first column in the eigenvector matrix.

# In[17]:


eigvals, eigvecs = np.linalg.eigh(Sx)
print(eigvals)
print('\n')
print(eigvecs)


# We can manipulate the formatting of the output for greater readability.

# In[18]:


np.set_printoptions(precision=3, suppress=True)

eigvals, eigvecs = np.linalg.eigh(Sy)
print(eigvals)
print('\n')
print(eigvecs)


# The formatting is tricky because it is `0.5 + 0j`, which is easy to misread.

# Let's check that the eigenvectors really are eigenvectors.

# In[19]:


eigvec1 = eigvecs[:,0]  # all columns (that is what : does) in the first row (which is 0)
print(eigvec1)


# In[20]:


print('check Sy @ eigenvector for first eigenvalue: ', Sy @ eigvec1)
print('compare to first eigenvalue * eigenvector:   ', eigvals[0] * eigvec1)


# **Test the other eigenvectors.**

# There are eigensolvers in both `numpy.linalg` and `scipy.linalg`. For now, just pick one.

# In[21]:


eigvals, eigvecs = LA.eigh(Sy)
print(eigvals)
print('\n')
print(eigvecs)


# Check whether $S_x$ (i.e., `Sx`) is Hermitian. We do the complex conjugate (`conjugate()`) and transpose (`T`) by hand here. We can also define our own function to do this.

# In[22]:


Sx - Sx.conjugate().T


# In[23]:


def my_adjoint(Matrix):
    """
    Return the complex-conjugate tranpose of the input Matrix. 
    """
    return Matrix.conjugate().T


# In[24]:


Sy - my_adjoint(Sy)


# In[25]:


Sz = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])


# Check $(\sigma_1^2 + \sigma_2^2 + \sigma_3^2)/2$

# In[26]:


1/2 * (Sx@Sx + Sy@Sy + Sz@Sz)


# ### Inner and outer product
# 
# More useful operations!

# In[27]:


a = np.array([1,2,3])
b = np.array([-1,1,0])
print( a @ b )
print( np.inner(a, b))


# In[28]:


print( np.outer(b, a))


# ### Trace and determinant

# Make a random matrix and take it's trace and determinant.

# In[29]:


M1 = np.random.rand(2,2)
print(M1)


# In[30]:


np.trace(M1)


# In[31]:


np.linalg.det(M1)


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




