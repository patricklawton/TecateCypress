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
        if hasattr(init_age, '__len__'):
            self.init_age = init_age
        else:
            self.init_age = np.repeat(int(init_age), len(self.N_0_1))
        assert(len(self.init_age) == len(self.N_0_1))

    def set_area(self, A):
        self.A = A
        # Scale the carrying capacity, in N/ha, with the area
        self.K_adult = self.K_adult * self.A
        self.K_seedling = self.K_seedling * self.A

    def set_fire_probabilities(self, fire_probs):
        #if not hasattr(fire_probs, '__len__'):
        #    self.fire_probs = fire_probs * np.ones((len(self.N_0_1), len(t_vec)))
        self.fire_probs = fire_probs

    def set_weibull_fire(self, b, c):
        self.weibull_b = b
        self.weibull_c = c

    def simulate(self, t_vec=np.arange(1,100), census_every=1, store=['mortality', 'fecundity']):
        delta_t = t_vec[1] - t_vec[0]
        # For sampling from various probability distributions
        rng = np.random.default_rng()
        # Get initial age time indices
        init_age_i_vec = [np.nonzero(t_vec == a)[0][0] for a in self.init_age]

        # Get timesteps of fire occurances
        if hasattr(self, 'fire_probs'):
            if not hasattr(self.fire_probs, '__len__'):
                self.fire_probs = self.fire_probs * np.ones((len(self.N_0_1), len(t_vec)))
            t_fire_vec = rng.binomial(np.ones((len(self.N_0_1), len(t_vec))).astype(int), self.fire_probs)
            # Make it deterministic
            #fri = 1/self.fire_probs[0,0]
            #t_fire_vec = np.array([1 if t%fri==0 else 0 for t in t_vec])
            #t_fire_vec = np.tile(t_fire_vec, (len(self.N_0_1), 1))
        elif hasattr(self, 'weibull_b') and hasattr(self, 'weibull_c'):
            # Calculate the frequency of fires between timesteps via Weibull hazard
            b_eff = self.weibull_b * delta_t
            term1 = 1/b_eff**self.weibull_c
            term2 = (t_vec + delta_t)**self.weibull_c - t_vec**self.weibull_c
            frequency_vec =  term1 * term2 
            t_star_vec = init_age_i_vec.copy() #Time since last fire indices
            t_fire_vec = np.zeros((len(self.N_0_1), len(t_vec))).astype(int)
            if self.weibull_b > 0:
                for pop_i in range(len(self.N_0_1)):
                    for t_i in range(len(t_vec)):
                        frequency = frequency_vec[t_star_vec[pop_i]]
                        fire = rng.poisson(lam=frequency, size=1)
                        if fire:
                            t_fire_vec[pop_i, t_i] = 1
                            t_star_vec[pop_i] = 0
                        else:
                            if t_star_vec[pop_i] < len(t_vec)-1:
                                t_star_vec[pop_i] += 1

        N_vec = np.ma.array(np.zeros((len(self.N_0_1), len(t_vec))))
        for pop_i, N_pop in enumerate(N_vec):
            a_i = init_age_i_vec[pop_i]
            N_pop[a_i] = self.N_0_1[pop_i]
        N_vec = N_vec.astype(int)
        # Initialize empty abundance array
        self.census_t = t_vec[::census_every]
        self.N_tot_vec = np.nan * np.ones((len(self.N_0_1), len(self.census_t)))
        self.N_tot_vec[:,0] = self.N_0_1
        
        # Age-dependent mortality functions
        m_a = self.alph_m * np.exp(-self.beta_m*t_vec) + self.gamm_m
        K_a = self.K_seedling * np.exp(-self.kappa*t_vec) + self.K_adult
        #K_a = np.repeat(self.K_adult, len(t_vec))
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
            for pop_i in range(len(N_vec)):
                # If sim invalid or pop extirpated, skip
                if np.all(np.isnan(N_vec)) or (np.sum(N_vec[pop_i]) == 0):
                    continue
                age_i_vec = np.nonzero(N_vec[pop_i])[0]
                # Cap the max age at the length of the time vector minus 1;
                # just because that's all we compute, could calculate on the fly
                if max(age_i_vec) >= len(t_vec) - 1:
                    N_vec[pop_i][age_i_vec[-1]-1] += N_vec[pop_i][age_i_vec[-1]]
                    N_vec[pop_i][age_i_vec[-1]] = 0
                    age_i_vec[-1] = len(t_vec) - 2
                if len(age_i_vec) > 1:
                    single_age = False
                else:
                    single_age = True
                    age_i = age_i_vec[0]
                    N = N_vec[pop_i][age_i]
                if t_fire_vec[pop_i, t_i]:
                    # Update seedlings, kill all adults
                    if not single_age:
                        epsilon_rho = rng.lognormal(np.zeros(len(t_vec)), sigm_a)
                        fecundities = rho_a*epsilon_rho
                        num_births = rng.poisson(fecundities*N_vec[pop_i])
                        # Make it deterministic
                        #num_births = fecundities*N_vec[pop_i]
                        N_vec[pop_i,0] = num_births.sum()
                        N_vec[pop_i,1:] = 0
                    else:
                        epsilon_rho = rng.lognormal(0, sigm_a[age_i])
                        fecundities = rho_a[age_i]*epsilon_rho
                        num_births = rng.poisson(fecundities*N)
                        N_vec[pop_i,0] = num_births
                        N_vec[pop_i,1:] = 0
                else:
                    # Update each pop given mortality rates
                    # Add density dependent term to mortalities
                    if not single_age:
                        dens_dep = ((nu_a)*(1-m_a)) / (1 + np.exp(-self.eta*self.K_adult*(np.sum(N_vec[pop_i]/K_a) - 1)))
                        m_a_N = m_a + dens_dep
                        survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i] * delta_t)
                        # Make it deterministic
                        #survival_probs = np.exp(-m_a_N * epsilon_m_mean * delta_t)
                    else:
                        dens_dep = ((nu_a[age_i])*(1-m_a[age_i])) / (1 + np.exp(-self.eta*self.K_adult*(N/K_a[age_i] - 1)))
                        m_a_N = m_a[age_i] + dens_dep
                        survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i][age_i] * delta_t)

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
                        N_vec = np.nan * np.ones((len(self.N_0_1), len(t_vec)))
            if t_vec[t_i+1] in self.census_t:
                self.N_tot_vec[:, t_i+1] = N_vec.sum(axis=1)
