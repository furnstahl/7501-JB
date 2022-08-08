# Modules needed for example: emcee is for MCMCsampling, corner for plotting
import numpy as np
from numpy.random import normal, uniform

# Lepage-based class for path integral

class PathIntegral:
    """
    A class for a path integral for 1D quantum mechanics. Associated with an 
    instance of the class is:
      * a potential with the Potential superclass but a particular subclass
      * a lattice definition
      * settings for correlation and thermalization "times" (Monte Carlo steps)
      * method for updating
      * method for averaging
      * list of configurations (paths)
      * choice of by-hand Metropolis updating or using emcee, zeus, or pyMC3
    
    """
    def __init__(self, Delta_T=0.25, N_pts=20, N_config=100, N_corr=20, eps=1.4,
                 V_pot=None):
        self.Delta_T = Delta_T        # DeltaT --> "a" in Lepage
        self.N_pts = N_pts            # N_pts --> "N" in Lepage 
        self.T_max = Delta_T * N_pts  # Tmax --> "T" in Lepage
        
        self.N_config = N_config
        self.N_corr = N_corr
        self.eps = eps
        
        self.V_pot = V_pot  # member of Potential class
        
        #self.x_path = np.zeros(self.N_pts)
        #self.list_of_paths = 
        
        
    def initialize(self, eps=None):
        """
        Initialize a path to zeros if eps=None otherwise to random numbers from 
        a normal distribution with mean zero and standard deviation eps.
        """
        if eps:
            x_path = np.array([normal(loc=0, scale=eps) for i in range(self.N_pts)])
        else:
            x_path = np.zeros(self.N_pts)
        return x_path

    def S_lattice(self, x_path):
        """
        Contribution to the action from path x_path 
        """
        action = 0
        for j in range(0, self.N_pts):        
            j_plus = (j + 1) % self.N_pts
            x_j = x_path[j]
            x_j_plus = x_path[j_plus]
            action = action + self.Delta_T * self.V_pot.V(x_j) \
              + (self.V_pot.mu/(2*self.Delta_T)) * (x_j_plus - x_j)**2
        return action    

    def S_lattice_j(self, x_path, j):
        """
        Function to calculate the contribution to action S from terms with j
        """
        j_plus = (j + 1) % self.N_pts  # next j point, wrapping around if needed
        j_minus = (j - 1) % self.N_pts
        x_j_minus = x_path[j_minus]
        x_j = x_path[j]
        x_j_plus = x_path[j_plus]
        
        return self.Delta_T * self.V_pot.V(x_j) \
               + self.V_pot.mu * x_j * \
                 (x_j - x_j_plus - x_j_minus) / self.Delta_T            

    def update(self, x_path):
        """
           This is a Metropolis update of the passed path.
            * x_path is an array of length N_pts
            * Step through each element and generate a candidate new value
               generated uniformly between -eps and +eps.
            * Check how much the action changes. Keep the new value *unless*
               the action increases *and* e^{-change} < uniform(0,1); that is
               even if the action increases, keep it with probability e^{-change}.    
        """
        for j in range(0, self.N_pts):
            old_x_path = x_path[j] # save original value 
            old_Sj = self.S_lattice_j(x_path, j)
            
            x_path[j] = x_path[j] + uniform(-self.eps, self.eps) # update x_path[j]
            dS = self.S_lattice_j(x_path, j) - old_Sj # change in action 
            if dS > 0 and np.exp(-dS) < uniform(0,1):
                x_path[j] = old_x_path  # restore old value
        return x_path

    def H_lattice_j(self, x_path, j):
        """
        Contribution to the energy from time point j
        """
        j_plus = (j + 1) % self.N_pts
        x_j = x_path[j]
        x_j_plus = x_path[j_plus]
        return self.Delta_T * self.V_pot.V(x_j) \
          + (self.V_pot.mu/(2*self.Delta_T)) * (x_j_plus - x_j)**2
    
    def display_x_path(self, x_path):
        """Print out x_path"""
        print(x_path)
        
    def MC_paths(self):
        """
        Accumulate paths after thermalization, skipping every N_corr
        """
        x_path = self.initialize(self.eps)  # initialize x_path 
        
        # thermalize x_path
        for i in range(5 * self.N_corr):
            x_path = self.update(x_path) 
        list_of_paths = np.array([x_path])
        
        for count in range(self.N_config-1):
            # for every count, skip N_corr paths
            for i in range(self.N_corr):
                x_path = self.update(x_path)
            # add the last one to the list
            list_of_paths = np.append(list_of_paths, [x_path], axis=0)
            
        return np.array(list_of_paths)
    
    def Gamma_avg_over_paths(self, Gamma, n, list_of_paths):
        """
        Calculate the average of Gamma(x_path, n) for tau point n over the
        paths in list_of_paths.
        """
        N_paths = len(list_of_paths)
        #print('Npaths = ', N_paths)
        Gamma_avg = 0.
        for x_path in list_of_paths:
            Gamma_avg = Gamma_avg + Gamma(x_path, n)
        return Gamma_avg / N_paths    
    
    def E_avg_over_paths(self, list_of_paths):
        """
        Average the lattice Hamiltonian over a set of configurations in
        list_of_paths.
        """
        N_paths = len(list_of_paths)
        #print('Npaths = ', N_paths)
        E_avg = 0.
        for x_path in list_of_paths:
            for j in range(self.N_pts):
                E_avg = E_avg + self.H_lattice_j(x_path, j)
        return E_avg / (N_paths * self.N_pts)
    
    def compute_G(self, x_path, n):
        """
        Calculate the correlator << x(t_j) x(t_{j+n}) >> averaged over 
        j = 1, ..., N_pts, where n corresponds to tau, i.e., tau = n * Delta_t.
        """
        N_tau = self.N_pts
        g = 0
        for j in range(0, N_tau):
            g = g + x_path[j] * x_path[(j+n)%N_tau] # wrap around as needed
        return g / N_tau
    
            


####################################################################################
### Other functions from Lepage's lectures
####################################################################################

# These are from Lepage's lectures. Not yet incorporated here.

def bootstrap(G): 
    """
    Do a bootstrap
    """
    N_cf = len(G)
    G_bootstrap = []  # new ensemble
    for i in range(0, N_cf):
        alpha = int(uniform(0,N_cf))  # choose random config
        G_bootstrap.append(G[alpha])  # keep G[alpha] 
    return G_bootstrap

def binning(G, binsize):
    """
    Binning
    """
    G_binned = [] # binned ensemble 
    for i in range(0, len(G), binsize): #loop on bins
        G_avg = 0
        for j in range(0, binsize): # loop on bin elements
            G_avg = G_avg + G[i+j] 
        G_binned.append(G_avg / binsize) # keep bin average
    return G_binned

