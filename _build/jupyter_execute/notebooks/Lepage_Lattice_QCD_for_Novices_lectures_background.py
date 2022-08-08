#!/usr/bin/env python
# coding: utf-8

# # Path integrals for quantum mechanics with MCMC libraries

# The original reference here is Peter Lepage's lectures entitled *Lattice QCD for Novices*, available as [arXiv:hep-lat/0506036](https://arxiv.org/abs/hep-lat/0506036). The starting point is one-dimensional quantum mechanics and that is where we focus our attention. The aim is to implement the calculations outlined by Lepage (updating some of the included Python code) and then switch to using some of the standard libraries for Markov Chain Monte Carlo (MCMC) designed for Bayesian inference.
# 
# To preview where we are going, the basic idea is that we can think of a discretized path in imaginary time $x(\tau)\rightarrow \{x(\tau_0), x(\tau_1), \ldots, x(\tau_{N-1})\}$ as a set of $N$ parameters and the path integral weighting factor $e^{-S[x(\tau)]}$ as an unnormalized likelihood. We'll combine with a uniform prior to make a posterior. If we sample the posterior with MCMC methods, we are sampling the paths according to the weighting by the action. We can use the samples to take quantum mechanical expectation values. 
# 

# ## Path integral basics
# 

# ### Formal aspects
# 
# We start with a path integral representation for an imaginary-time evolution between position eigenstates:
# 
# $$
#   \langle x_f | e^{-\widehat H(\tau_f - \tau_i)} | x_i \rangle = \int \mathcal{D}x(\tau)\, e^{-S[x(\tau)]}
# $$ 
# 
# where $\tau = it$ and
# 
# $$
#   \mathcal{D}x(\tau) \longrightarrow \mbox{sum over paths } \{x(\tau) \mbox{ where } \tau_i \leq \tau \leq \tau_f\}
#   \mbox{ with } x_i = x(\tau_i),\ x_f = x(\tau_f) .
# $$
# 
# The Hamiltonian is $\widehat H$ and $S[x(\tau)]$ is the classical action evaluated for path $x(\tau)$:
# 
# $$
#   S[x(\tau)] \equiv \int_{\tau_i}^{\tau_f} d\tau\, L(x,\dot x; \tau)
#     = \int_{\tau_i}^{\tau_f} d\tau\, \biggl[\frac{1}{2}m \dot x(\tau)^2 + V(x(\tau))\biggr] .
# $$
# 
# This is the *Euclidean* version of $S$ and the Lagrangian $L$, which is why the relative sign of the kinetic and potential terms is positive (i.e., $L = T_E + V$ rather than $L = T - V$).

# If we insert on the left side a complete set of exact eigenstates of $\widehat H$ (labeled by $n$), namely
# 
# $$
#    \mathbb{1} = \sum_n |E_n\rangle \langle E_n| 
#    \quad\mbox{where}\quad \widehat H |E_n\rangle = E_n |E_n\rangle ,
# $$
# 
# then we can use $e^{-\widehat H (\tau_f - \tau_i)}|E_n\rangle = e^{-E_n(\tau_f-\tau_i)}|E_n\rangle$ (which follows after a power series expansion of the exponential) to obtain
# 
# $$
#   \langle x_f | e^{-\widehat H(\tau_f - \tau_i)} | x_i \rangle 
#      = \sum_n \langle x_f |E_n\rangle e^{-E_n(\tau_f-\tau_i)} \langle E_n | x_i\rangle ,
# $$
# 
# which involves both the wave functions and energy eigenvalues. Because $E_0 < E_{n\neq 0}$, we can take $T \equiv \tau_f - \tau_i \rightarrow \infty$ to isolate the first terms (all the others are suppressed exponentially relative to the first). We have the freedom to take $x_f = x_i \equiv x$, which yields in the large time limit:
# 
# $$
#   \langle x | e^{-\widehat H T} | x \rangle \overset{T\rightarrow\infty}{\longrightarrow}
#     e^{-E_0 T} |\langle x | E_0\rangle|^2 .
# $$
# 
# The wavefunction $\psi_{E_0}(x) = \langle x|E_0\rangle$ is normalized, so
# 
# $$
#   \int_{-\infty}^{\infty} \langle x | e^{-\widehat H T} | x \rangle \overset{T\rightarrow\infty}{\longrightarrow}
#     e^{-E_0 T} 
# $$    
# 
# and we get the energy (then we can go back and divide out this factor to get the wave function squared).
# 
# To derive the path integral formerly, we divide up the Euclidean time interval from $\tau_i$ to $\tau_f$ into little intervals of width $\Delta \tau$ and insert a complete set of $x$ states in each time. This enables us to approximate $e^{-\widehat H\Delta\tau}$ in each interval. In a full derivation we would insert both momentum and coordinate states and evaluate the matrix element of $\widehat H(\hat p,\hat x)$. The small $\Delta\tau$ lets us neglect the commutator between $\hat p$ and $\hat x$ as higher order (proportional to $\Delta\tau^2$) If we then do all the momentum integrals, we get the path integral over $\mathcal{D}x(\tau)$. 

# ### Numerical implementation
# 
# In the formal discussion of path integrals, a path is defined at every continuous point in Euclidan time $\tau$, but to represent this on a computer (and actually make a sensible definition), we discretize the time:
# 
# $$  
#   \tau_i \leq \tau \leq \tau_f \quad\longrightarrow\quad \tau_j = \tau_i + j a \ \mbox{for }j = 0,\ldots,N
# $$
# 
# with grid (mesh) spacing $a$:
# 
# $$
#    a \equiv \frac{\tau_f - \tau_i}{N} .
# $$
# 
# A path is then a list of numbers:
# 
# $$
#    x(\tau)\rightarrow \{x(\tau_0), x(\tau_1), \ldots, x(\tau_{N-1})\}
# $$
# 
# and the integration over all paths becomes a conventional multiple integral over the values of $x_j$ at each $\tau_j$ for $1 \leq j \leq N-1$:
# 
# $$
#    \int \mathcal{D}x(\tau) \longrightarrow A \int_{-\infty}^{\infty} dx_1 \int_{-\infty}^{\infty} dx_2
#      \cdots \int_{-\infty}^{\infty} dx_{N-1} .
# $$
# 
# The endpoints $x_0$ and $x_N$ are determined by the boundary conditions (e.g., $x_0 = x_N = x$ as used above). If we need the normalization $A$ (which we often won't because it will drop out), then for the one-dimensional problems here it is
# 
# $$
#     A = \left(\frac{m}{2\pi a}\right)^{N/2} .
# $$
# 
# We will need an approximation to the action. Here is one choice for $\tau_{j} \leq \tau \leq \tau_{j+1}$:
# 
# $$
#   \int_{\tau_{j}}^{\tau_{j+1}} d\tau\, L \approx
#     a \left[ \frac{m}{2} \left(\frac{x_{j+1}-x_{j}}{a}\right)^2 
#              + \frac{1}{2} \bigl( V(x_{j+1}) + V(x_{j}) \bigr)
#       \right]
# $$
# 
# Now we have
# 
# $$
#   \langle x | e^{-\widehat H \tau} | x \rangle \approx
#     A \int_{-\infty}^{\infty} dx_1 \int_{-\infty}^{\infty} dx_2
#      \cdots \int_{-\infty}^{\infty} dx_{N-1} e^{-S_{\rm lat}[x]}
# $$
# 
# where
# 
# $$
#    S_{\rm lat}[x] \equiv \sum_{j=0}^{N-1} 
#      \left[  
#       \frac{m}{2a}(x_{j+1}-x_{j})^2 + a V(x_j)
#      \right]
# $$
# 
# with $x_0 = x_N = x$ and $a = T/N$.
# 
# Comments:
# * One might worry about the discretization of the derivatives, because of the range of the $x_{j}$'s: adjacent points can be arbitrarily far away (the paths can be arbitrarily "rough"). 

# ### Other matrix elements
# 
# Suppose we wanted a Euclidean expectation value:
# 
# $$
#   \langle\langle x(\tau_2)x(\tau_1) \rangle\rangle
#    \equiv \frac{\int \mathcal{D}x(\tau)\, x(\tau_2)x(\tau_1)\,e^{-S[x]}}{\int \mathcal{D}x(\tau)\, e^{-S[x]}}
# $$
# 
# where there is an integration over $x_i = x_f = x$ as well as over the intermediate $x(t)$'s. Note that we are calculating a weighted average of $x(\tau_2)x(\tau_1)$ over all paths, where the weighting factor is $e^{-S[x]}$. This means that for every path, we take the values of $x$ at $\tau_2$ and $\tau_1$ and multiply them together, with the weight $e^{-S[x]}$ that involves the entire path.
# 
# In the continuum form, the right side is
# 
# $$
#   \int dx\, \langle x| e^{-\widehat H(\tau_f-\tau_2)} \hat x e^{-\widehat H(\tau_2-\tau_1)}
#     \hat x e^{-\widehat H(\tau_1-\tau_i)} | x \rangle ,
# $$
# 
# so with $T = \tau_f - \tau_i$ and $\tau = \tau_2 - \tau_1$, we can insert our complete set of eigenstates to derive
# 
# $$
#     \langle\langle x(\tau_2)x(\tau_1) \rangle\rangle =
#     \frac{\sum_n e^{-E_n T} \langle E_n | \hat x e^{-(\widehat H-E_n)\tau}\hat x | E_n\rangle}{\sum_n e^{-E_n T}}
# $$
# 
# Once again, for large $T$ the ground state will dominate the sums, so
# 
# $$
#   G(\tau) \equiv \langle\langle x(\tau_2)x(\tau_1) \rangle\rangle \longrightarrow
#      \langle E_0 | \hat x e^{-(\widehat H - E_0)\tau} \hat x | E_0 \rangle .
# $$
# 
# But what state propagates in the middle? It could be any created by $\hat x$ acting on the ground state. But this doesn't *include* the ground state because $\hat x$ switches the parity of the state (from even to odd). We can insert a complete set of states again, and if $\tau \ll T$ is large itself, then the first excited state $|E_1\rangle$ will be projected out:
# 
# $$
#   G(\tau) \overset{\tau\ {\rm large}}{\longrightarrow} |\langle E_0 | \hat x | E_1 \rangle|^2
#     e^{-(E_1-E_0)\tau} .
# $$
# 
# Given $G(\tau)$, we can use the $\tau$ dependence of the exponential (and $\tau$ independence of the squared matrix element) to isolate the excitation energy and then the matrix element:
# 
# $$\begin{align}
#   \log(G(\tau)/G(\tau+a) &\overset{\tau\ {\rm large}}{\longrightarrow} (E_1 - E_0)a \\
#   |\langle E_0 | \hat x | E_1 \rangle|^2 &\overset{\tau\ {\rm large}}{\longrightarrow} G(\tau)e^{+(E_1-E_0)\tau} 
# \end{align}$$
# 
# We can, in principle, generalize to an arbitrary function of $x$, call it $\Gamma[x]$, and compute any physical property of the excited states. We can also compute thermal averages at finite $T$:
# 
# $$
#   \langle\langle \Gamma[x] \rangle\rangle = 
#     \frac{\sum_n e^{-E_n T} \langle E_n | \Gamma[\hat x] | E_n\rangle}{\sum_n e^{-E_n T}}
#     \quad\mbox{with } T \rightarrow \beta \equiv \frac{1}{k_B T_{\rm temp}} .
# $$    

# ### Numerical evaluation with MCMC
# 
# The idea of the Monte Carlo evaluation of
# 
# $$
#  \langle\langle \Gamma[x] \rangle\rangle = 
#    \frac{\int\mathcal{D}x(\tau)\, \Gamma[x]e^{-S[x]}}{\int\mathcal{D}x(\tau)\, e^{-S[x]}}
# $$
# 
# is that this is a weighted average of $\Gamma[x]$ over paths, so if we generate a set of paths that are distributed according to $e^{-S[x]}$ (that is, the relative probability of a given path is proportional to this factor), then we can just do a direct average of $\Gamma[x]$ over this set.
# 
# To be explicit, we will generate a large number $N_{\rm cf}$ of random paths ("configurations" or cf)
# 
# $$
#     x^{(\alpha)} \equiv \{ x_0^{(\alpha)}, x_1^{(\alpha)}, \ldots, x_{N-1}^{(\alpha)} \}
#       \quad \alpha = 1,2,\ldots N_{\rm cf}
# $$
# 
# on the time grid such that the probability of getting a particular path $x^{(\alpha)}$ is
# 
# $$
#      P[x^{(\alpha)}] \propto e^{-S[x^{(\alpha)}]} .
# $$
# 
# Then 
# 
# $$
#    \langle\langle \Gamma[x] \rangle\rangle \approx \overline\Gamma \equiv
#        \frac{1}{N_{\rm cf}}\sum_{\alpha=1}^{N_{\rm cf}} \Gamma[x^{(\alpha)}] .
# $$
# 
# This is equivalent to approximating the probability distribution as
# 
# $$
#      P[x] \approx \frac{1}{N_{\rm cf}}\sum_{\alpha=1}^{N_{\rm cf}} \delta(x - x^{(\alpha)}) . 
# $$
# 
# where the delta function is the product of delta functions for each of the $x_j$'s. Note that this representation is automatically normalized simply by including the $1/N_{\rm cf}$ factor.

# How good is the estimate? With a finite number of paths $N_{\rm cf}$ there is clearly a statistical error $\sigma_{\overline\Gamma}$, which he can estimate by calculating the variance using are set of paths:
# 
# $$
#   \sigma_{\overline\Gamma}^2 = \frac{\langle\langle \Gamma^2 \rangle\rangle - \langle\langle  \Gamma\rangle\rangle^2}{N_{\rm cf}}
#   \approx \frac{1}{N_{\rm cf}} \left\{  
#       \frac{1}{N_{\rm cf}} \sum_{\alpha=1}^{N_{\rm cf}} \Gamma^2[x^{(\alpha)}] - \overline\Gamma^2
#   \right\} .
# $$
# 
# The numerator for large $N_{\rm cf}$ is independent of $N_{\rm cf}$ (e.g., it can be calculated directly, in principle, from quantum mechanics), so the error decreases as $\sigma_{\overline\Gamma} \propto 1/\sqrt{N_{\rm cf}}$ with increasing $N_{\rm cf}$.
# 
# We can use Markov Chain Monte Carlo (MCMC) to generate our sample of paths, first by the Metropolis algorithm and then by more sophisticated methods. Starting from an arbitrary path $x^{(0)}$, we generate a series of subsequent paths by considering each lattice site in turn, randomizing the $x_j$'s at these sites according to the algorithm, to generate a new path, and then repeat until we have $N_{\rm cf}$ random paths. The Metropolis algorithm applied to $x_j$ at the $j^{\rm th}$ site is:
# * generate a random number $\zeta \sim U(-\epsilon,\epsilon)$, that is $\zeta$ is uniformly distributed from $-\zeta$ to $+\zeta$, for a constant *step size* $\epsilon$ (see later);
# * replace $x_j \rightarrow x_j + \zeta$ and compute the change $\Delta S$ in the action as a result of this replacement (local Lagrangian means this only requires considering a few terms);
# * if $\Delta S < 0$ (action reduced) then retain new $x_j$ and move to the next site;
# * if $\Delta S > 0$, generate a random number $\eta \sim U(0,1)$ (uniform from 0 to 1) and retain the new $x_j$ if $e^{-\Delta S} > \eta$, else restore the old $x_j$; move to the next site.
# 
# Comments:
# * Whether or not many of the $x_j$'s do not change in successive paths depends a lot on $\epsilon$. If $\epsilon$ is very large, then the changes in $x_j$ are large and many or most will be rejected; if $\epsilon$ is very small, then the changes in $x_j$ are small and most will be accepted, but the path will be close to the same.
# * Neither situation is good: both slow down the exploration of the space of important paths.
# * Claim: usually tune $\epsilon$ so 40% to 60% of the $x_j$'s change on each sweep. This implies that $\epsilon$ is the same order as the typical quantum fluctuations expected.
# * Because the paths take time to get decorrelated, we should only keep every $N_{\rm cor}$ path. The optimal value can be determined empirically; here the optimal $N_{\rm cor}$ depends on $a$ as $1/a^2$.
# * When starting the algorithm, the first configuration is atypical (usually). So we should have a thermalization period where the first paths are discarded. Recommendation: five to ten times $N_{\rm cor}$ should be discarded.

# ### Bootstrap and binning
# 
# 

# ### Higher-order discretization
# 
# Suppose we rewrite the action for one-dimensional quantum mechanics by integrating the kinetic term by parts (assuming $x(\tau_i)=x(\tau_f)=x$:
# 
# $$
#   S[x] \equiv \int_{\tau_i}^{\tau_f} d\tau\, \bigl[-\frac{1}{2}m x(\tau)\ddot x(\tau) + V(x(\tau))\bigr]
# $$
# 
# The discretization is
# 
# $$
#   \ddot x(t) \longrightarrow \Delta^{(2)}x_j \equiv \frac{x_{j+1}-2x_{j}+x_{j-1}}{a^2}
# $$
# 
# where the $x_j$'s wrap around at the ends: $x_0 = x_{N}$, $x_{-1} = x_{N-1}$, and so on. 

# In[ ]:





# In[ ]:




