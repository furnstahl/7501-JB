#!/usr/bin/env python
# coding: utf-8

# # Shooting Method with Numerov Algorithm
# 
# 
# **Authors:** Dou Du, Taylor James Baird and Giovanni Pizzi 
# 
# <i class="fa fa-home fa-2x"></i><a href="../index.ipynb" style="font-size: 20px"> Go back to index</a>
# 
# **Source code:** https://github.com/osscar-org/quantum-mechanics/blob/master/notebook/quantum-mechancis/shooting_method.ipynb
# 
# This notebook demonstrates the shooting method with the Numerov algorithm to search the 
# eigenfunctions (wavefunctions) and eigenvalues for a one-dimensional quantum well.
# 
# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />

# ## **Goals**
# 
# * Understand the mathematical method to solve the Schrödinger equation 
# numerically and the boundary condition for the 1D quantum well.
# * Understand the Numerov algorithm and how to improve the accuracy with 
# high order correction.
# * Know how to use the shooting method with the Numerov algorithm to get 
# the eigenvalues and eigenfunctions.

# ## **Background theory**
# 
# [More on the background theory.](./theory/theory_shooting_method.ipynb)

# ## **Tasks and exercises**
# 
# 1. Move the sliders for the width and depth of the quantum well. Do you understand
# the concept of quantum confinement? Do you know any numerical method to solve 
# the Schrödinger equation for 1D quantum well?
# 
#     <details>
#     <summary style="color: red">Solution</summary>
#     Please check the previous notebooks for the 
#     <a href="./1quantumwell.ipynb">1D quantum well</a>.
#     In that notebook, the one-dimensional Shrödinger equation was solved
#     by numerical matrix diagonalization.
#     </details>
# 
# 2. With the default width (1.20) and depth (0.20), move the sliders 
# (on the left side) to the targeted energies. Report the energy when the tail 
# of the wavefunction on the right converge to zero (line color turns to green). 
# Is the energy the same as the eigenvalue shown in the right plot? You can also 
# use the "auto search" button to get the eigenvalues, which searches the next 
# solution when increasing the energy (i.e. it searches always upwards).
#     <details>
#     <summary style="color: red">Solution</summary>
#     The 1st eigenvalue is about 0.0092. You may need to click the "Flip
#     eigenfunctions" button to make the comparsion. Check the exact eigenvalue 
#     by clicking on the eigenfunction in the plot.
#     </details>
#     
# 3. Follow the same step to get all the eigenvalues, and make a table to compare 
# the results with the eigenvalues from the figure. Compare the results with 
# and without using the 4th derivative correction (checkbox). Which values 
# should be more accurate and why?
#     <details>
#     <summary style="color: red">Solution</summary>
#     Please check the background theory section for the Numerov algorithm.
#     </details>

# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />
# 
# ## Interactive visualization
# (be patient, it might take a few seconds to load)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'widget')

from numpy import linspace, sqrt, ones, arange, diag, argsort, zeros
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt
from ipywidgets import FloatSlider, jslink, VBox, HBox, Button, Label, Layout, Checkbox, Output
import numpy as np


# In[ ]:


colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f'];
ixx = 0

def singlewell_potential(x, width, depth):
    x1 = ones(len(x))*depth
    for i in range(len(x)):
        if x[i] > - width/2.0 and x[i] < width/2.0:
            x1[i] =0
    return x1


def diagonalization(hquer, L, N, pot=singlewell_potential, width = 0.1, depth = 0.0):
    """Calculated sorted eigenvalues and eigenfunctions. 

       Input:
         hquer: Planck constant
         L: set viewed interval [-L,L] 
         N: number of grid points i.e. size of the matrix 
         pot: potential function of the form pot
         x0: center of the quantum well
         width: the width of the quantum well
         depth: the depth of the quantum well
       Ouput:
         ew: sorted eigenvalues (array of length N)
         ef: sorted eigenfunctions, ef[:,i] (size N*N)
         x:  grid points (arry of length N)
         dx: grid space
         V:  Potential at positions x (array of length N)
    """
    x = linspace(-L, L, N+2)[1:N+1]                 # grid points
    dx = x[1] - x[0]                                # grid spacing
    V = pot(x, width, depth)
    z = hquer**2 /2.0/dx**2                         # second diagonals

    ew, ef = eigh_tridiagonal(V+2.0*z, -z*ones(N-1))
    ew = ew.real                                    # real part of the eigenvalues
    ind = argsort(ew)                               # Indizes f. sort. Array
    ew = ew[ind]                                    # Sort the ew by ind
    ef = ef[:, ind]                                 # Sort the columns 
    ef = ef/sqrt(np.sum(ef[0]*ef[0]*dx))            # Correct standardization 
    return ew, ef, x, dx, V


def plot_eigenfunctions(ax, ew, ef, x, V, width=1, updateTarget=True):
    """Plot of the lowest eigenfunctions 'ef' in the potential 'V (x)'
       at the level of the eigenvalues 'ew' in the plot area 'ax'.
    """
    global lnum, lax1, lspan
    
    fak = sfak.value/(50.0);
    
    try:
        lspan.remove()
    except:
        pass
    
    lspan = ax[0].axhspan(max(V), max(V)+0.05, facecolor='lightgrey')
    
    ax[0].set_xlim([min(x), max(x)])
    ax[0].set_ylim([min(V)-0.05, max(V)+0.05])
    
    ax[0].set_xlabel(r'$x/a$', fontsize = 10)
    ax[0].set_ylabel(r'$V(x)/V_0\ \rm{, Eigenfunctions\ with\ Eigenvalues}$', fontsize = 10)
    
    ax[1].set_xlim([min(x), max(x)])
    ax[1].set_ylim([min(V)-0.05, max(V) + 0.05])
    
    if updateTarget:
        loop1.min = min(V)-0.03
        loop1.min = int(loop1.min*100)/100.0
        loop1.value = loop1.min
        loop2.value = loop2.min
        loop3.value = loop3.min
        loop4.value = loop4.min
    
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    
    ax[1].get_xaxis().set_visible(False)
    #ax[1].set_ylabel(r'$\rm{\ Eigenvalues}$', fontsize = 10)
    
    indmax = sum(ew < max(V))                       
    if not hasattr(width, "__iter__"):           
        width = width*ones(indmax)               
    for i in arange(indmax):                     
        ax[0].plot(x, fak*ef[:, i]+ew[i], linewidth=width[i]+.1, color=colors[i%len(colors)])
        ax[1].plot(x, x*0.0+ew[i], linewidth=width[i]+2.5, color=colors[i%len(colors)])
        
    ax[0].plot(x, V, c='k', linewidth=1.6)
    lnum, = ax[0].plot(x, x*0 + loop1.value,'r--', linewidth=1.0)
    lax1, = ax[1].plot(x, x*0 + loop1.value,'r--', linewidth=1.0)
    


# In[ ]:


mu = 0.06                                            # Potential parameter
L = 1.5                                              # x range [-L,L]
N = 200                                              # Number of grid points
hquer = 0.06                                         # Planck constant
sigma_x = 0.1                                        # Width of the Gaussian function
zeiten = linspace(0.0, 10.0, 400)                    # time
Flip = False                                         # Flip the eigenfunction


swidth = FloatSlider(value = 1.2, min = 0.1, max = 2.0, description = 'Width: ')
sdepth = FloatSlider(value = 0.2, min = 0.05, max = 1.0, step = 0.05, description = 'Depth: ')
sfak = FloatSlider(value = 3.0, min = 1.0, max = 5.0, step = 0.5, description = r'Zoom factor: ')

output = Output()

update = Button(description="Show all")
flip = Button(description="Flip eigenfunction")
search = Button(description="Auto search")

order = Checkbox(value=True, description="incl. 4th derivative", indent=False,
                layout=Layout(width='180px'))

loop1 = FloatSlider(value = -0.03, min = -0.03, max = 0.2,
                    layout=Layout(height='450px', width='30px'), step = 0.01, readout_format=".2f", orientation='vertical')
loop2 = FloatSlider(value = 0, min = 0, max = 99, 
                  layout=Layout(height='450px', width='30px'), step =1.0, readout_format='02d', orientation='vertical')
loop3 = FloatSlider(value = 0, min = 0, max = 99,
                  layout=Layout(height='450px', width='30px'), step =1.0, readout_format='02d', orientation='vertical')
loop4 = FloatSlider(value = 0, min = 0, max = 99,
                  layout=Layout(height='450px', width='30px'), step =1.0, readout_format='02d', orientation='vertical')


Leng = Label('')
Evalue = loop1.value + loop2.value/10000.0 + loop3.value/1000000.0 + loop4.value/100000000.0;
Leng.value = "Current value: " + "{:.8f}".format(Evalue)

width = 1.2
depth = 0.2
fak = 5.0

ew, ef, x, dx, V = diagonalization(hquer, L, N, width = width, depth = depth)
   
with output:
    global fig
    fig, ax = plt.subplots(1, 2, figsize=(6,6), gridspec_kw={'width_ratios': [10, 1]})
    fig.canvas.header_visible = False
    fig.canvas.layout.width = "750px"

    fig.suptitle('Numerical Solution ($\psi$) of the Schrödinger Equation \n for 1D Quantum Well', fontsize = 12)
    plot_eigenfunctions(ax, ew, ef, x, V)
    plt.show()

def Numerov(y, E, Vn, dxx):
    y = zeros(len(y));
    y[0] = 0.0;
    
    if Flip:
        y[1] = -0.00000001
    else:
        y[1] = 0.00000001
    
    k2 = 2.0/(hquer**2)*(E-Vn)*dxx*dxx;
    
    for i in arange(2, len(y)):
        if order.value:
            y[i] = (2*(12.0-5.0*k2[i-1])*y[i-1] - (12+k2[i-2])*y[i-2])/(12+k2[i]);
        else:
            y[i] = 2*y[i-1] - k2[i-1]*y[i-1] - y[i-2]
    return y/(sqrt(np.sum(abs(y)**2*dxx))*50.0)*sfak.value

def plot_numerov(c):
    Nn = 1000
    xx = linspace(-L, L, Nn+2)[1:Nn+1]
    dxx = xx[1] - xx[0];
    Vn = singlewell_potential(xx, width = swidth.value, depth = sdepth.value)
    yy = zeros(len(xx));
    
    Evalue = loop1.value + loop2.value/10000.0 + loop3.value/1000000.0 + loop4.value/100000000.0;
                   
    yy = Numerov(yy, Evalue, Vn, dxx);
    
    if abs(yy[-1]) < 0.001:
        lnum.set_color("green")
        lax1.set_color("green")
    else:
        lnum.set_color("red")
        lax1.set_color("red")

    Leng.value = "Current value: " + "{:.8f}".format(Evalue)    
    lnum.set_data(xx, yy + Evalue)
    lax1.set_data(xx, xx*0 + Evalue)
    
def on_auto_search(b):
    Nn = 1000
    xx = linspace(-L, L, Nn+2)[1:Nn+1]
    dxx = xx[1] - xx[0];
    Vn = singlewell_potential(xx, width = swidth.value, depth = sdepth.value)
    yy = zeros(len(xx));
    
    Evalue = loop1.value + loop2.value/10000.0 + loop3.value/1000000.0 + loop4.value/100000000.0;
    yy = Numerov(yy, Evalue, Vn, dxx);
    
    increment = 0.01
    while abs(yy[-1]) > 0.001:
        tail_old = yy[-1]
        Evalue += increment;
        yy = Numerov(yy, Evalue, Vn, dxx);
        tail_new = yy[-1]
        
        if tail_old*tail_new < 0:
            Evalue -= increment
            increment /= 100.0
            yy = Numerov(yy, Evalue, Vn, dxx);
            
    Leng.value = "Current value: " + "{:.8f}".format(Evalue)        
    lnum.set_data(xx, yy + Evalue)
    lax1.set_data(xx, xx*0 + Evalue)
    
    loop1.value = int(Evalue*100)/100.0;
    loop2.value = int((Evalue-loop1.value)*10000); 
    loop3.value = int((Evalue-loop1.value-loop2.value/10000)*1000000);
    loop4.value = int((Evalue-loop1.value-loop2.value/10000-loop3.value/1000000)*100000000)
    

def on_update_click(b):
    for i in ax[0].lines:
        i.set_alpha(1.0)
    for i in ax[1].lines:
        i.set_alpha(1.0)

    try:
        ann.remove()
        ann1.remove()
    except:
        pass

def on_width_change(change):
    global ew, ef, x, dx, V
    ax[0].lines = []
    ax[1].lines = []
    
    try:
        ann.remove()
        ann1.remove()
    except:
        pass

    ew, ef, x, dx, V = diagonalization(hquer, L, N, width = swidth.value, depth = sdepth.value)
    plot_eigenfunctions(ax, ew, ef, x, V)

def on_depth_change(change):
    global ew, ef, x, dx, V
    ax[0].lines = []
    ax[1].lines = []
    
    try:
        ann.remove()
        ann1.remove()
    except:
        pass

    ew, ef, x, dx, V = diagonalization(hquer, L, N, width = swidth.value, depth = sdepth.value)
    plot_eigenfunctions(ax, ew, ef, x, V)
    loop1.max = max(V)
    
def on_xfak_change(change):
    ax[0].lines = []
    ax[1].lines = []
    
    try:
        ann.remove()
        ann1.remove()
    except:
        pass

    plot_eigenfunctions(ax, ew, ef, x, V, updateTarget=False)
    plot_numerov('test')

def on_press(event):
    global ann, ann1, ixx
    
    ixx = min(enumerate(ew), key = lambda x: abs(x[1]-event.ydata))[0]
    
    for i in range(len(ax[1].lines)-1):
        ax[0].lines[i].set_alpha(0.1)
        ax[1].lines[i].set_alpha(0.1)
        ax[0].lines[i].set_linewidth(1.1)
        
    ax[0].lines[ixx].set_alpha(0.5)
    ax[1].lines[ixx].set_alpha(0.5)
    ax[0].lines[ixx].set_linewidth(2.0)
    
    try:
        ann.remove()
        ann1.remove()
    except:
        pass
    
    ann = ax[0].annotate(s = 'n = ' + str(ixx+1), xy = (0, ew[ixx]), xytext = (-0.15, ew[ixx]), xycoords = 'data', color='k', size=15)
    ann1 = ax[1].annotate(s = str("{:.3f}".format(ew[ixx])), xy = (0, ew[ixx]), xytext = (-1.2, ew[ixx]+0.005), xycoords = 'data', color='k', size=9)

def on_flip_eigenfunctions(b):
    global Flip
    x = lnum.get_xdata();
    y = lnum.get_ydata();
    lnum.set_data(x, -y+2.0*y[0])
    Flip = not Flip

    
cid = fig.canvas.mpl_connect('button_press_event', on_press)

swidth.observe(on_width_change, names = 'value')
sdepth.observe(on_depth_change, names = 'value')
sfak.observe(on_xfak_change, names = 'value')

update.on_click(on_update_click)
flip.on_click(on_flip_eigenfunctions)
search.on_click(on_auto_search)

loop1.observe(plot_numerov, names = 'value')
loop2.observe(plot_numerov, names = 'value')
loop3.observe(plot_numerov, names = 'value')
loop4.observe(plot_numerov, names = 'value')


label1 = Label(value="Targeted eigenvalue")
label2 = Label(value="Click to flip the eigenfunction")
label3 = Label(value="(click on a state to select it)")
label4 = Label(value="(tune to zoom the eigenfunctions)")

display(HBox([VBox([label1, HBox([loop1, loop2, loop3, loop4]), Leng, search, order, label2, flip]), output]))


# Set the **width** and **depth** of the quantum well:

# In[ ]:


display(HBox([swidth, sdepth]), VBox([HBox([sfak, label4]), HBox([update, label3])]))


# <hr style="height:1px;border:none;color:#cccccc;background-color:#cccccc;" />
# 
# # Legend
# 
# (How to use the interactive visualization)
# 
# ## Interactive figures
# 
# In the interative figure, the soild lines show the wavefunctions and their
# corresponding eigenvalues, which are solved by matrix diagonalization. 
# There is a red dash line at the bottom of the figure, which shows the 
# eigenfunction solved by Numerov algorithm. 
# 
# ## Controls
# 
# There are four vertical sliders to control the targeted eigenvalue E. The first 
# slider controls the precision for tenths ($10^{-1}$) and hundredths ($10^{-2}$). 
# The second slider controls thousandths ($10^{-3}$) and ten thousandths decimal ($10^{-4}$). The third slider controls hundred thousandths ($10^{-5}$) and 
# millionths ($10^{-6}$). The last slider controls ten millionths ($10^{-7}$) 
# and hundred millionths ($10^{-8}$). The current value is also displayed under 
# the sliders.
# 
# You need slowly move the 1st slider and observe the tail of the dashed line on 
# the right edge. Once you see the tail change directions (up or down), the true 
# value should be between these two values. You need to go back to a smaller value 
# and start to tune the 2nd slider. Then the same procedure is for the 3rd and 4th
# slider. When the absolute value at the right edge is smaller than 0.001, the 
# dashed red line will turn green. It reaches the desired accuracy for the 
# wavefunction. Then, you can read out the current targeted value, which is the
# corresponding eigenvalue.
# 
# You can also use the `Auto search` button, which finds the closest eigenvalue 
# and eigenfunction (search in the upward direction). In order to make a comparison, 
# you may also need to click the `Flip eigenfunctions` button.

# In[ ]:




