import numpy as np
import json
from itertools import product
import pandas as pd
import sys
import timeit
from scipy.integrate import quad, solve_ivp
from tqdm import tqdm

# Get the average habitat suitability within the Otay Mtn Wilderness area
sdmfn = "SDM_1995.asc"
sdm = np.loadtxt(sdmfn,skiprows=6)
otay = np.loadtxt("otayraster.asc", skiprows=6)
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
        self.K_seedling = kwargs['K_seedling']; self.kappa = kwargs['kappa']; self.K_adult = kwargs['K_adult']
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
        self.t_fire_vec = t_fire_vec
    
    def _simulate_discrete(self, N_vec, progress):
        # Age-dependent mortality functions
        m_a = self.alph_m * np.exp(-self.beta_m*self.t_vec) + self.gamm_m
        K_a = self.K_seedling * np.exp(-self.kappa*self.t_vec) + self.K_adult
        nu_a = self.alph_nu * np.exp(-self.beta_nu*self.t_vec) + self.gamm_nu
        # Use linear approx to set eta s.t. shape of dens. dep. curve is 
        # the same for arbitrary effective patch size
        delta, theta = (1.05, 0.050000000000000044) #just hardcoding these in
        eta_a = (theta*2)/((nu_a*(1-m_a)) * (A_o*h_o*self.K_adult) * (delta-1))
        sigm_m_a = self.sigm_m*np.exp(-self.tau_m*self.t_vec)
        epsilon_m_vec = rng.lognormal(np.zeros_like(N_vec)+self.mu_m, np.tile(sigm_m_a, (len(self.N_0_1),1)))
        # Make it deterministic
        epsilon_m_mean = np.exp(sigm_m_a**2 / 2) 
        # Age-dependent fecundity functions
        rho_a = self.rho_max / (1+np.exp(-self.eta_rho*(self.t_vec-self.a_mature)))
        sigm_a = self.sigm_max / (1+np.exp(-self.eta_sigm*(self.t_vec-self.a_sigm_star)))
        # Make it deterministic
        epsilon_rho = np.exp(sigm_a**2 / 2)
        fecundities = rho_a*epsilon_rho

        for t_i, t in enumerate(tqdm(self.t_vec[:-1], disable=(not progress))):
            for pop_i in range(len(N_vec)):
                # If sim invalid or pop extirpated, skip
                if np.all(np.isnan(N_vec)) or (np.sum(N_vec[pop_i]) == 0):
                    continue
                age_i_vec = np.nonzero(N_vec[pop_i])[0]
                # Cap the max age at the length of the time vector minus 1;
                # just because that's all we compute, could calculate on the fly
                if max(age_i_vec) >= len(self.t_vec) - 1:
                    N_vec[pop_i][age_i_vec[-1]-1] += N_vec[pop_i][age_i_vec[-1]]
                    N_vec[pop_i][age_i_vec[-1]] = 0
                    age_i_vec[-1] = len(self.t_vec) - 2
                if len(age_i_vec) > 1:
                    single_age = False
                else:
                    single_age = True
                    age_i = age_i_vec[0]
                    N = N_vec[pop_i][age_i]
                if self.t_fire_vec[pop_i, t_i]:
                    # Update seedlings, kill all adults
                    if not single_age:
                        epsilon_rho = rng.lognormal(np.zeros(len(self.t_vec)), sigm_a)
                        fecundities = rho_a*epsilon_rho
                        num_births = rng.poisson(fecundities*N_vec[pop_i])
                        # Make it deterministic
                        #num_births = fecundities*N_vec[pop_i]
                        N_vec[pop_i,0] = num_births.sum()
                        N_vec[pop_i,1:] = 0
                    else:
                        epsilon_rho = rng.lognormal(0, sigm_a[age_i])
                        fecundities = rho_a[age_i]*epsilon_rho
                        '''Really sloppy way to deal with too large abudances, can get away with this for now because I end up throwing out these simulations during analysis anyways'''
                        try:
                            num_births = rng.poisson(fecundities*N)
                        except ValueError:
                            num_births = rng.poisson(1e18)
                        N_vec[pop_i,0] = num_births
                        N_vec[pop_i,1:] = 0
                else:
                    # Update each pop given mortality rates
                    # Add density dependent term to mortalities
                    if not single_age:
                        dens_dep = ((nu_a)*(1-m_a)) / (1 + np.exp(-eta_a*self.K_adult*(np.sum(N_vec[pop_i]/K_a) - self.Aeff)))
                        m_a_N = m_a + dens_dep
                        survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i] * self.delta_t)
                        # Make it deterministic
                        #survival_probs = np.exp(-m_a_N * epsilon_m_mean * self.delta_t)
                    else:
                        dens_dep = ((nu_a[age_i])*(1-m_a[age_i])) / (1 + np.exp(-eta_a[age_i]*self.K_adult*(N/K_a[age_i] - self.Aeff)))
                        m_a_N = m_a[age_i] + dens_dep
                        survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i][age_i] * self.delta_t)

                    # Ensure survival probs are feasible, otherwise mark sim invalid 
                    prob_check = np.all(survival_probs >= 0) and np.all(survival_probs <= 1)
                    if prob_check:
                        if not single_age:
                            num_survivors = rng.binomial(N_vec[pop_i], survival_probs)
                            # Make it deterministic
                            #num_survivors = N_vec[pop_i]*survival_probs
                            num_survivors = np.roll(num_survivors, 1)
                            # Update abundances
                            N_vec[pop_i] = num_survivors
                        else:
                            num_survivors = rng.binomial(N, survival_probs)
                            # Update abundances
                            N_vec[pop_i][age_i+1] = num_survivors
                            N_vec[pop_i][age_i] = 0
                    else:
                        N_vec = np.nan * np.ones((len(self.N_0_1), len(self.t_vec)))
            if self.t_vec[t_i+1] in self.census_t:
                self.N_tot_vec[:, t_i+1] = N_vec.sum(axis=1)


    def _simulate_nint(self, progress):
        # Instantaneous mortality for numerical integration
        def _dNdt(t, N):
            # Age-dependent mortality functions
            m_t = self.alph_m * np.exp(-self.beta_m*t) + self.gamm_m
            #K_t = K_adult
            K_t = self.K_seedling * np.exp(-self.kappa*t) + self.K_adult
            nu_t = self.alph_nu * np.exp(-self.beta_nu*t) + self.gamm_nu
            delta, theta = (1.05, 0.050000000000000044) #just hardcoding these in
            eta_t = (theta*2)/((nu_t*(1-m_t)) * (A_o*h_o*self.K_adult) * (delta-1))
            dens_dep = ((nu_t)*(1-m_t)) / (1 + np.exp(-eta_t*self.K_adult*(N/K_t - self.Aeff)))
            m_t_N = m_t + dens_dep
            sigm_m_t = self.sigm_m*np.exp(-self.tau_m*t)
            epsilon_m_mean = np.exp(self.mu_m + (sigm_m_t**2 / 2))
            return -m_t_N * N * epsilon_m_mean

        # Number of 1 yr old seedlings following fire for numerical integration
        def _get_num_births(t, N):
            # Age-dependent fecundity functions
            rho_t = self.rho_max / (1+np.exp(-self.eta_rho*(t-self.a_mature)))
            sigm_t = self.sigm_max / (1+np.exp(-self.eta_sigm*(t-self.a_sigm_star)))
            # Approximate number of births
            epsilon_rho_mean = np.exp(0 + (sigm_t**2 / 2))
            num_births = rho_t*epsilon_rho_mean*N
            return num_births

        for pop_i, t_fire_pop in enumerate(tqdm(self.t_fire_vec, disable=(not progress))):
            fire_indices = np.argwhere(t_fire_pop!=0).flatten()
            # Handle inter-fire intervals
            for fire_num, fire_i in enumerate(fire_indices):
                #print(f'fire_num {fire_num}')
                # Initialize first interval with specified initial abundance
                if fire_i == min(fire_indices):
                    t_eval = np.arange(self.delta_t, fire_i+self.delta_t)
                    #print(f"t_eval: {t_eval}")
                    init_i = 0
                    N_i = self.K_adult
                # Otherwise set initial conditions for a given interval
                else:
                    t_eval = np.arange(self.delta_t, fire_i - fire_indices[fire_num-1] + self.delta_t)
                    #print(f"t_eval: {t_eval}")
                    init_i = fire_indices[fire_num-1]
                    if num_births < 1:
                        num_births = 0
                    N_i = num_births
                    #print(f"N_i: {N_i}")
                # Handle cases with nonzero abundance
                if N_i > 0:
                    if len(t_eval) > 1:
                        sol = solve_ivp(_dNdt, [self.delta_t,fire_i], [N_i], t_eval=t_eval) 
                        # Set any abundances < 1 to zero
                        sol.y[0] = np.where(sol.y[0] > 1, sol.y[0], 0)
                        if fire_i == min(fire_indices):
                            num_births = _get_num_births(len(t_eval) + self.init_age[0], sol.y[0][-1])
                        else:
                            num_births = _get_num_births(len(t_eval), sol.y[0][-1])
                        #print(f"solution from timestep {init_i} to {fire_i-1}")
                        #print(sol.y)
                        self.N_tot_vec[pop_i][init_i:fire_i] = sol.y[0]
                    # Handle case of consecutive fires or first fire on timestep 1
                    elif len(t_eval) == 1:
                        # Get num births for first fire on timestep 1 
                        if fire_i == min(fire_indices):
                            num_births = _get_num_births(len(t_eval) + self.init_age[0], N_i)
                        #print(f"solution from timestep {init_i} to {fire_i-1}")
                        #print(num_births)
                        self.N_tot_vec[pop_i][init_i] = num_births
                        # Get num births following consecutive fire
                        num_births = _get_num_births(len(t_eval), num_births)
                        #print(f"num_births after consecutive fire: {num_births}")
                    # Handle case where fire occurs on timestep 0
                    elif len(t_eval) == 0:
                        num_births = _get_num_births(1 + self.init_age[0], N_i)
                        #print(f"num_births following fire on timestep {init_i}: {num_births}")
                        if num_births < 1:
                            num_births = 0
                # If pop extirpated, keep abundance at zero
                else:
                    #print(f"solution from timestep {init_i} to {fire_i-1}")
                    #print("0")
                    self.N_tot_vec[pop_i][init_i:fire_i+1] = 0.
                    num_births = 0
            # Handle final timesteps without fire
            if len(self.t_vec) > fire_i+1:
                fire_num += 1
                if num_births < 1:
                    num_births = 0.
                t_eval = np.arange(self.delta_t, len(self.t_vec) - fire_i + self.delta_t)
                sol = solve_ivp(_dNdt, [self.delta_t,len(t_eval)], [num_births], t_eval=t_eval) 
                if (len(sol.y)!=0) and (len(sol.y[0]) > 1):
                    self.N_tot_vec[pop_i][fire_i:len(self.t_vec)] = sol.y[0]
                else: 
                    self.N_tot_vec[pop_i][fire_i:len(self.t_vec)] = 0.
            # Handle case where fire occurs on final timestep
            elif len(self.t_vec) == fire_i+1:
                fire_num += 1
                num_births = _get_num_births(len(t_eval) - 1, sol.y[0][-1])
                if num_births < 1:
                    num_births = 0.
                self.N_tot_vec[pop_i][-1] = num_births

    def simulate(self, method, census_every=1, progress=False):
        # Get initial age time indices
        init_age_i_vec = [np.nonzero(self.t_vec == a)[0][0] for a in self.init_age]

        N_vec = np.ma.array(np.zeros((len(self.N_0_1), len(self.t_vec))))
        for pop_i, N_pop in enumerate(N_vec):
            a_i = init_age_i_vec[pop_i]
            N_pop[a_i] = self.N_0_1[pop_i]
        N_vec = N_vec.astype(int)
        # Initialize empty abundance array
        self.census_t = self.t_vec[::census_every]
        self.N_tot_vec = np.nan * np.ones((len(self.N_0_1), len(self.census_t)))
        self.N_tot_vec[:,0] = self.N_0_1
        
        if method == "discrete":
            self._simulate_discrete(N_vec, progress)
        elif method == "nint":
            self._simulate_nint(progress)
