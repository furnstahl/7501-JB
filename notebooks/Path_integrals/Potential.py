import numpy as np

# 1-dimensional potential class

class Potential:
    """
    A general class for potentials
    
    Parameters
    ----------
    hbar : float
        Planck's constant. Equals 1 by default
    mu : float
        Reduced mass. Equals 1 by default

    Methods
    -------
    V(x)
        Returns the value of the potential at x.
    
    E_gs
        Returns the analytic ground-state energy, if known
        
    wf_gs(x)
        Returns the analytic ground-state wave function, if known
        
    plot_V(ax, x_pts)
        Plots the potential at x_pts on ax
    """
    def __init__(self, hbar=1., mu=1., V_string=''):
        self.hbar = hbar
        self.mu = mu
        self.V_string = V_string
        
    def V(self, x):
        """
        Potential at x
        """
        print('The potential is not defined.') 
        
    def E_gs(self):
        """
        analytic ground state energy, if known
        """
        print('The ground state energy is not known analytically.') 
        
    def wf_gs(self, x):
        """
        analytic ground state wave function, if known
        """
        print('The ground state wave function is not known analytically.')
        
    def plot_V(self, ax, x_pts, V_label=''):
        """
        Plot the potential on the given axis
        """
        ax.plot(x_pts, self.V(x_pts), color='blue', alpha=1, label=V_label)


####################################################################################
### Harmonic Oscillator and related definitions
####################################################################################

# We'll do the one-dimensional harmonic oscillator (ho), by default in units where 
# the basic quantities are all unity.  In these units, the energies should be (n+1/2), 
# where n = 0,1,2,3,...).        

class V_HO(Potential):
    """
    Harmonic oscillator potential (subclass of Potential)

    """
    def __init__(self, k_osc=1, hbar=1, mu=1, V_string='Harmonic oscillator'):
        self.k_osc = 1
        super().__init__(hbar, mu, V_string)

    def V(self, x) :
        """Standard harmonic oscillator potential for particle at x"""
        return self.k_osc * x**2 /2
    
    def E_gs(self):
        """
        1D harmonic oscillator ground-state energy
        """
        (1/2) * self.hbar * np.sqrt(self.k_osc / self.mass)  # ground state energy 
        
    def wf_gs(self, x):
        """
        1D harmonic oscillator ground-state wave function
        """
        return np.exp(-x**2 / 2) / np.pi**(1/4)  # We should  put the units back!


class V_aHO(Potential):
    """
    Subclass of Potential

    Parameters
    ----------

    Methods
    -------

    """
    def __init__(self, k_osc=1, hbar=1, mu=1, V_string='Anharmonic oscillator'):
        self.k_osc = 1
        super().__init__(hbar, mu, V_string)

    def V(self, x) :
        """Anharmonic oscillator potential for particle at x"""
        return self.k_osc * x**4 /2








