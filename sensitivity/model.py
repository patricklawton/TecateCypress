import numpy as np
import json
from itertools import product
import pandas as pd
import sys
import timeit
from scipy.integrate import quad, solve_ivp
from tqdm import tqdm

# Get the average habitat suitability within the Otay Mtn Wilderness area
sdmfn = "../shared_maps/SDM_1995.asc"
sdm = np.loadtxt(sdmfn,skiprows=6)
otay = np.loadtxt("../shared_maps/otayraster.asc", skiprows=6)
sdm_otay = sdm[otay==1] #index "1" indicates the specific part where study was done
h_o = np.mean(sdm_otay[sdm_otay!=0]) #excluding zero, would be better to use SDM w/o threshold
A_o = 0.1 #area of observed sites in Ha

# For sampling from various probability distributions
rng = np.random.default_rng()

class Model:
    def __init__(self, **kwargs):
        # Mortality parameters
        self.alph_m = kwargs['alph_m']; self.beta_m = kwargs['beta_m']; self.gamm_m = kwargs['gamm_m']
        self.sigm_m = kwargs['sigm_m']; self.tau_m = kwargs['tau_m']; self.mu_m = kwargs['mu_m']
        self.alph_nu = kwargs['alph_nu']; self.beta_nu = kwargs['beta_nu']; self.gamm_nu = kwargs['gamm_nu']
        self.K_seedling = kwargs['K_adult']*6 #NEED TO UDPATE FIXED PKL 
        self.kappa = kwargs['kappa']; self.K_adult = kwargs['K_adult']
        # Fecundity parameters
        self.rho_max = kwargs['rho_max']; self.eta_rho = kwargs['eta_rho']; self.a_mature = kwargs['a_mature']
        self.sigm_max = kwargs['sigm_max']; self.eta_sigm = kwargs['eta_sigm']; 
        self.a_sigm_star = self.a_mature
    
    def init_N(self, N_0_1, init_age):
        if hasattr(N_0_1, '__len__'):
            self.N_0_1 = N_0_1
        else:
            self.N_0_1 = np.array([int(N_0_1)])
        if hasattr(init_age, '__len__'):
            self.init_age = init_age
        else:
            self.init_age = np.repeat(int(init_age), len(self.N_0_1))
        assert(len(self.init_age) == len(self.N_0_1))

    def set_effective_area(self, Aeff):
        self.Aeff = Aeff

    def set_t_vec(self, t_vec):
        self.t_vec = t_vec
        self.delta_t = self.t_vec[1] - self.t_vec[0]

    def set_fire_probabilities(self, fire_probs):
        self.fire_probs = fire_probs

    def set_weibull_fire(self, b, c):
        self.weibull_b = b
        self.weibull_c = c

    def generate_fires(self):
        # Get initial age time indices
        init_age_i_vec = [np.nonzero(self.t_vec == a)[0][0] for a in self.init_age]
        # Get timesteps of fire occurances
        if hasattr(self, 'fire_probs'):
            if not hasattr(self.fire_probs, '__len__'):
                self.fire_probs = self.fire_probs * np.ones((len(self.N_0_1), len(self.t_vec)))
            t_fire_vec = rng.binomial(np.ones((len(self.N_0_1), len(self.t_vec))).astype(int), self.fire_probs)
            # Make it deterministic
            #fri = 1/self.fire_probs[0,0]
            #t_fire_vec = np.array([1 if t%fri==0 else 0 for t in self.t_vec])
            #t_fire_vec = np.tile(t_fire_vec, (len(self.N_0_1), 1))
        elif hasattr(self, 'weibull_b') and hasattr(self, 'weibull_c'):
            # Calculate the frequency of fires between timesteps via Weibull hazard
            b_eff = self.weibull_b * self.delta_t
            term1 = 1/b_eff**self.weibull_c
            term2 = (self.t_vec + self.delta_t)**self.weibull_c - self.t_vec**self.weibull_c
            frequency_vec =  term1 * term2 
            t_star_vec = init_age_i_vec.copy() #Time since last fire indices
            t_fire_vec = np.zeros((len(self.N_0_1), len(self.t_vec))).astype(int)
            if self.weibull_b > 0:
                for pop_i in range(len(self.N_0_1)):
                    for t_i in range(len(self.t_vec)):
                        frequency = frequency_vec[t_star_vec[pop_i]]
                        fire = rng.poisson(lam=frequency, size=1)
                        if fire:
                            t_fire_vec[pop_i, t_i] = 1
                            t_star_vec[pop_i] = 0
                        else:
                            if t_star_vec[pop_i] < len(self.t_vec)-1:
                                t_star_vec[pop_i] += 1
        self.t_fire_vec = t_fire_vec.astype(bool)
    
    def _simulate_discrete(self, N_vec, progress):
        # Age-dependent mortality functions
        m_a = self.alph_m * np.exp(-self.beta_m*self.t_vec) + self.gamm_m
        K_a = self.K_seedling * np.exp(-self.kappa*self.t_vec) + self.K_adult
        nu_a = self.alph_nu * np.exp(-self.beta_nu*self.t_vec) + self.gamm_nu
        # Use linear approx to set eta s.t. shape of dens. dep. curve is 
        # the same for arbitrary effective patch size
        #eta_a = 2 / ((nu_a*(1-m_a)) * self.Aeff * self.K_adult)
        #eta_a = np.ones(self.t_vec.size)
        sigm_m_a = self.sigm_m*np.exp(-self.tau_m*self.t_vec)
        epsilon_m_vec = rng.lognormal(np.zeros_like(N_vec)+self.mu_m, np.tile(sigm_m_a, (len(self.N_0_1),1)))
        ## Make it deterministic
        #epsilon_m_mean = np.exp(sigm_m_a**2 / 2) 
        # Age-dependent fecundity functions
        rho_a = self.rho_max / (1+np.exp(-self.eta_rho*(self.t_vec-self.a_mature)))
        #sigm_a = self.sigm_max / (1+np.exp(-self.eta_sigm*(self.t_vec-self.a_sigm_star)))
        sigm_a = np.repeat(self.sigm_max, self.t_vec.size)
        epsilon_rho_vec = rng.lognormal(np.zeros_like(N_vec), np.tile(sigm_a, (len(self.N_0_1),1)))
        ## Make it deterministic
        #epsilon_rho = np.exp(sigm_a**2 / 2)
        #fecundities = rho_a*epsilon_rho

        for t_i, t in enumerate(tqdm(self.t_vec[:-1], disable=(not progress))):
            # Identify extirpated populations (fully zero) and valid ones
            valid_pops = np.sum(N_vec, axis=1) > 0  # Boolean mask for surviving populations
            if not np.any(valid_pops):  # If all populations are extinct, stop early
                break

            # Compute existing age indices for valid populations
            age_mask = N_vec > 0  # Boolean mask for nonzero ages

            # Cap maximum age at len(self.t_vec) - 1
            max_age_mask = np.argmax(age_mask[:, ::-1], axis=1)  # Get rightmost nonzero index
            max_age_mask = N_vec.shape[1] - 1 - max_age_mask
            cap_mask = max_age_mask >= len(self.t_vec) - 1
            N_vec[np.arange(N_vec.shape[0]), max_age_mask - 1] += N_vec[np.arange(N_vec.shape[0]), max_age_mask] * cap_mask
            N_vec[np.arange(N_vec.shape[0]), max_age_mask] *= ~cap_mask  # Zero out if capped

            # Fire event update (vectorized)
            fire_mask = self.t_fire_vec[:, t_i] & valid_pops
            if np.any(fire_mask):
                fecundities = rho_a * epsilon_rho_vec  # Shape: (pop, age)
                num_births = rng.poisson(fecundities * N_vec)  # Shape: (pop, age)
                N_vec[fire_mask, 0] = num_births.sum(axis=1)[fire_mask]  # Seedling recruitment
                N_vec[fire_mask, 1:] = np.roll(N_vec[fire_mask, 1:] * 0.0025, shift=1, axis=1)  # post-fire survival

            # Mortality update (vectorized) - Only applied where fire **does not** occur
            mortality_mask = ~fire_mask & valid_pops  # Populations without fire
            if np.any(mortality_mask):
                survival_probs = np.exp(-m_a * epsilon_m_vec * self.delta_t)  # Shape: (pop, age)
                prob_check = np.all(survival_probs >= 0) and np.all(survival_probs <= 1)
                if not prob_check: print("FUCK")
                num_survivors = rng.binomial(N_vec, survival_probs)  # Binomial survival draw
                N_vec[mortality_mask, :] = np.roll(num_survivors[mortality_mask], shift=1, axis=1)  # Age shift

            # Census update (vectorized)
            if self.t_vec[t_i + 1] in self.census_t:
                self.N_tot_vec[:, t_i + 1] = N_vec.sum(axis=1)

    def _simulate_nint(self, progress):
        # Instantaneous mortality for numerical integration
        def _dNdt(t, N):
            if N < 0:
                return 0.0
            else:
                # Age-dependent mortality functions
                m_t = self.alph_m * np.exp(-self.beta_m*t) + self.gamm_m
                #K_t = self.K_seedling * np.exp(-self.kappa*t) + self.K_adult
                #nu_t = self.alph_nu * np.exp(-self.beta_nu*t) + self.gamm_nu
                #eta_t = 2 / ((nu_t*(1-m_t)) * self.Aeff * self.K_adult)
                #''' Turing off dens dep so set eta_t to 1 or it will blow up'''
                #eta_t = 1
                #dens_dep = ((nu_t)*(1-m_t)) / (1 + np.exp(-eta_t*self.K_adult*(N/K_t - self.Aeff)))
                #m_t_N = m_t + dens_dep
                m_t_N = m_t
                #sigm_m_t = self.sigm_m*np.exp(-self.tau_m*t)
                sigm_m_t = self.sigm_m
                epsilon_m_mean = np.exp(self.mu_m + (sigm_m_t**2 / 2))
                return -m_t_N * N * epsilon_m_mean

        # Number of 1 yr old seedlings following fire for numerical integration
        def _get_num_births(t, N):
            # Age-dependent fecundity functions
            rho_t = self.rho_max / (1+np.exp(-self.eta_rho*(t-self.a_mature)))
            #sigm_t = self.sigm_max / (1+np.exp(-self.eta_sigm*(t-self.a_sigm_star)))
            sigm_t = self.sigm_max
            # Approximate number of births
            epsilon_rho_mean = np.exp(0 + (sigm_t**2 / 2))
            num_births = rho_t*epsilon_rho_mean*N
            return num_births

        for pop_i, t_fire_pop in enumerate(tqdm(self.t_fire_vec, disable=(not progress))):
            fire_indices = np.argwhere(t_fire_pop!=0).flatten()
            # Handle inter-fire intervals
            for fire_num, fire_i in enumerate(fire_indices):
                # Initialize first interval with specified initial abundance
                if fire_i == min(fire_indices):
                    t_eval = np.arange(self.delta_t, fire_i+self.delta_t)
                    t_eval = t_eval + self.init_age[0]
                    init_i = 0
                    N_i = self.N_0_1[pop_i] 
                # Otherwise set initial conditions for a given interval
                else:
                    '''I think sometimes we're starting here instead of above
                       Is it because I'm allowing fires to occur on the first timestep?
                       Do I want to make sure that's always at N_0?
                       .... Maybe I can just roll any effected trajectories up by one, and fill in the initial
                       Also set Aeff in sp to 1
                       But still need to rerun because of forgetting to include Aeff...
                       hopefully doesn't change the shape of each trajectory though'''
                    t_eval = np.arange(self.delta_t, fire_i - fire_indices[fire_num-1] + self.delta_t, self.delta_t)
                    init_i = fire_indices[fire_num-1]
                    if num_births < 1:
                        num_births = 0
                    N_i = num_births
                # Handle cases with nonzero abundance
                if N_i > 0:
                    if len(t_eval) > 1:
                        if fire_i == min(fire_indices):
                            t_bounds = [self.delta_t+self.init_age[0],fire_i+self.init_age[0]]
                            sol = solve_ivp(_dNdt, t_bounds, [N_i], t_eval=t_eval) 
                            # Set any abundances < 1 to zero
                            sol.y[0] = np.where(sol.y[0] > 1, sol.y[0], 0)
                            num_births = _get_num_births(len(t_eval) + self.init_age[0], sol.y[0][-1])
                        else:
                            t_bounds = [self.delta_t, fire_i - fire_indices[fire_num-1] + self.delta_t]
                            sol = solve_ivp(_dNdt, t_bounds, [N_i], t_eval=t_eval) 
                            # Set any abundances < 1 to zero
                            sol.y[0] = np.where(sol.y[0] > 1, sol.y[0], 0)
                            num_births = _get_num_births(len(t_eval), sol.y[0][-1])
                        self.N_tot_vec[pop_i][init_i:fire_i] = sol.y[0]
                    # Handle case of consecutive fires or first fire on timestep 1
                    elif len(t_eval) == 1:
                        # Get num births for first fire on timestep 1 
                        if fire_i == min(fire_indices):
                            num_births = _get_num_births(len(t_eval) + self.init_age[0], N_i)
                        self.N_tot_vec[pop_i][init_i] = num_births
                        # Get num births following consecutive fire
                        num_births = _get_num_births(len(t_eval), num_births)
                    # Handle case where fire occurs on timestep 0
                    elif len(t_eval) == 0:
                        num_births = _get_num_births(1 + self.init_age[0], N_i)
                        if num_births < 1:
                            num_births = 0
                # If pop extirpated, keep abundance at zero
                else:
                    self.N_tot_vec[pop_i][init_i:fire_i+1] = 0.
                    num_births = 0

            # Handle case of no fires
            if len(fire_indices) == 0:
                t_eval = self.t_vec + self.init_age[0]
                t_bounds = np.array([min(self.t_vec), max(self.t_vec)]) + self.init_age[0]
                sol = solve_ivp(_dNdt, t_bounds, [self.N_0_1[pop_i]], t_eval=t_eval)
                sol.y[0] = np.where(sol.y[0] > 1, sol.y[0], 0)
                self.N_tot_vec[pop_i] = sol.y[0]
            # Handle final timesteps without fire
            elif len(self.t_vec) > fire_i+1:
                fire_num += 1
                if num_births < 1:
                    num_births = 0.
                t_eval = np.arange(self.delta_t, len(self.t_vec) - fire_i + self.delta_t)
                sol = solve_ivp(_dNdt, [self.delta_t,len(t_eval)], [num_births], t_eval=t_eval) 
                sol.y[0] = np.where(sol.y[0] > 1, sol.y[0], 0)
                if (len(sol.y)!=0) and (len(sol.y[0]) > 1):
                    self.N_tot_vec[pop_i][fire_i:len(self.t_vec)] = sol.y[0]
                else: 
                    self.N_tot_vec[pop_i][fire_i:len(self.t_vec)] = 0.
            # Handle case where fire occurs on final timestep
            elif len(self.t_vec) == fire_i+1:
                fire_num += 1
                num_births = _get_num_births(len(t_eval) - 1, self.N_tot_vec[pop_i][-2])
                if num_births < 1:
                    num_births = 0.
                self.N_tot_vec[pop_i][-1] = num_births

    def simulate(self, method, census_every=1, progress=False):
        # Get initial age time indices
        init_age_i_vec = [np.nonzero(self.t_vec == a)[0][0] for a in self.init_age]

        # Initialize empty per age abundnace vec updated each timestep in discrete sims
        N_vec = np.ma.array(np.zeros((len(self.N_0_1), len(self.t_vec))))
        for pop_i, N_pop in enumerate(N_vec):
            a_i = init_age_i_vec[pop_i]
            N_pop[a_i] = self.N_0_1[pop_i]
        N_vec = N_vec.astype(int)
        # Initialize empty total abundance array (used in nint and discrete sims)
        self.census_t = self.t_vec[::census_every]
        self.N_tot_vec = np.nan * np.ones((len(self.N_0_1), len(self.census_t)))
        self.N_tot_vec[:,0] = self.N_0_1
        
        if method == "discrete":
            self._simulate_discrete(N_vec, progress)
        elif method == "nint":
            self._simulate_nint(progress)
