import numpy as np
import json
from itertools import product
import pandas as pd
import sys

class Model:
    def __init__(self, **kwargs):
        # Mortality parameters
        self.alph_m = kwargs['alph_m']; self.beta_m = kwargs['beta_m']; self.gamm_m = kwargs['gamm_m']
        self.sigm_m = kwargs['sigm_m']; self.tau_m = kwargs['tau_m']
        self.alph_nu = kwargs['alph_nu']; self.beta_nu = kwargs['beta_nu']; self.gamm_nu = kwargs['gamm_nu']
        self.K_seedling = kwargs['K_seedling']; self.kappa = kwargs['kappa']; self.K_adult = kwargs['K_adult']
        self.eta = kwargs['eta']; self.mu_m = kwargs['mu_m']
        # Fecundity parameters
        self.rho_max = kwargs['rho_max']; self.eta_rho = kwargs['eta_rho']; self.a_mature = kwargs['a_mature']
        self.sigm_max = kwargs['sigm_max']; self.eta_sigm = kwargs['eta_sigm']; 
        self.a_sigm_star = self.a_mature
    
    def init_N(self, N_0_1, init_age):
        if hasattr(N_0_1, '__len__'):
            self.N_0_1 = N_0_1
        else:
            self.N_0_1 = np.array([int(N_0_1)])
        self.init_age = init_age

    def set_area(self, A):
        self.A = A
        # Scale the carrying capacity, in N/ha, with the area
        self.K_adult = self.K_adult * self.A

    def set_fire_probabilities(self, p):
        if not hasattr(p, '__len__'):
            self.p = p * np.ones((len(self.N_0_1), len(t_vec)))

    def simulate(self, t_vec=np.arange(1,100), census_every=1, store=['mortality', 'fecundity'],
                 fire_probs=0.0):
        delta_t = t_vec[1] - t_vec[0]
        # For sampling from various probability distributions
        rng = np.random.default_rng()

        # Get timesteps of fire occurances
        if not hasattr(fire_probs, '__len__'):
            fire_probs = fire_probs * np.ones((len(self.N_0_1), len(t_vec)))
        #t_fire_vec = rng.binomial(np.ones((len(self.N_0_1), len(t_vec))).astype(int), fire_probs)
        # Make it deterministic
        fri = 1/fire_probs[0,0]
        t_fire_vec = np.array([1 if t%fri==0 else 0 for t in t_vec])
        t_fire_vec = np.tile(t_fire_vec, (len(self.N_0_1), 1))

        N_vec = np.ma.array(np.zeros((len(self.N_0_1), len(t_vec))))
        init_age_i = np.nonzero(t_vec == self.init_age)[0][0]
        N_vec[:,init_age_i] = self.N_0_1
        N_vec = N_vec.astype(int)
        # Initialize empty abundance array
        self.census_t = t_vec[::census_every]
        self.N_tot_vec = np.nan * np.ones((len(self.N_0_1), len(self.census_t)))
        self.N_tot_vec[:,0] = self.N_0_1
        
        # Age-dependent mortality functions
        m_a = self.alph_m * np.exp(-self.beta_m*t_vec) + self.gamm_m
        K_a = self.K_adult
        nu_a = self.alph_nu * np.exp(-self.beta_nu*t_vec) + self.gamm_nu
        sigm_m_a = self.sigm_m*np.exp(-self.tau_m*t_vec)
        epsilon_m_vec = rng.lognormal(np.zeros_like(N_vec)+self.mu_m, np.tile(sigm_m_a, (len(self.N_0_1),1)))
        # Make it deterministic
        epsilon_m_mean = np.exp(sigm_m_a**2 / 2) 
        # Age-dependent fecundity functions
        rho_a = self.rho_max / (1+np.exp(-self.eta_rho*(t_vec-self.a_mature)))
        sigm_a = self.sigm_max / (1+np.exp(-self.eta_sigm*(t_vec-self.a_sigm_star)))
        # Make it deterministic
        epsilon_rho = np.exp(sigm_a**2 / 2)
        fecundities = rho_a*epsilon_rho

        for t_i, t in enumerate(t_vec[:-1]):
            for pop_i, N_pop in enumerate(N_vec):
                # If sim invalid or pop extirpated, skip
                if np.all(np.isnan(N_vec)) or (np.sum(N_vec[pop_i]) == 0):
                    continue
                if t_fire_vec[pop_i, t_i] == True:
                    # Update seedlings, kill all adults
                    epsilon_rho = rng.lognormal(np.zeros(len(t_vec)), sigm_a)
                    fecundities = rho_a*epsilon_rho
                    num_births = rng.poisson(fecundities*N_vec[pop_i])
                    # Make it deterministic
                    #num_births = fecundities*N_vec[pop_i]
                    N_vec[pop_i,0] = num_births.sum()
                    N_vec[pop_i,1:] = 0
                else:
                    # Update each pop given mortality rates
                    # Add density dependent term to mortalities
                    dens_dep = ((nu_a)*(1-m_a)) / (1 + np.exp(-self.eta*self.K_adult*(np.sum(N_pop/K_a) - 1)))
                    m_a_N = m_a + dens_dep
                    # Try correcting for discretization issue
                    age_i = np.nonzero(N_pop)[0][0]
                    num_draws = 1000
                    epsilon_draws = rng.lognormal(np.zeros(num_draws), np.repeat(sigm_m_a[age_i], num_draws))
                    m_eff = 1 - np.mean(np.exp(-m_a_N[age_i]*epsilon_draws*delta_t))
                    m_nint = m_a_N[age_i]*epsilon_m_mean[age_i]*delta_t
                    z = m_nint - m_eff 
                    m_a_N[age_i] = m_a_N[age_i] + z
                    # Make it deterministic
                    #epsilon_m = epsilon_m_mean
                    survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i] * delta_t)
                    try:
                        # Ensure survival probs are feasible, otherwise mark sim invalid 
                        assert(np.all(survival_probs >= 0) and np.all(survival_probs <= 1))
                        num_survivors = rng.binomial(N_pop, survival_probs)
                        # Make it deterministic
                        #num_survivors = N_pop*survival_probs
                        num_survivors = np.roll(num_survivors, 1)
                        # Update abundances
                        N_vec[pop_i] = num_survivors
                    except:
                        N_vec = np.nan * np.ones((len(self.N_0_1), len(t_vec)))
            if t_vec[t_i+1] in self.census_t:
               self.N_tot_vec[:, t_i+1] = N_vec.sum(axis=1)
