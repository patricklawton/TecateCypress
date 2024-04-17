from scipy.integrate import quad, solve_ivp
import json
import numpy as np
from matplotlib import pyplot as plt

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle))

#def m_t_N(t, N):
def dNdt(t, N):
    # Mortality parameters
    alph_m = params['alph_m']; beta_m = params['beta_m']; gamm_m = params['gamm_m']
    sigm_m = params['sigm_m']; tau_m = params['tau_m']
    alph_nu = params['alph_nu']; beta_nu = params['beta_nu']; gamm_nu = params['gamm_nu']
    K_seedling = params['K_seedling']; kappa = params['kappa']; K_adult = params['K_adult']
    eta = params['eta']; mu_m = params['mu_m']

    # Age-dependent mortality functions
    m_t = alph_m * np.exp(-beta_m*t) + gamm_m
    K_t = K_adult
    nu_t = alph_nu * np.exp(-beta_nu*t) + gamm_nu
    dens_dep = ((nu_t)*(1-m_t)) / (1 + np.exp(-eta*(N - K_t)))
    m_t_N = m_t + dens_dep
    rng = np.random.default_rng()
    #epsilon_m = rng.lognormal(mu_m, sigm_m*np.exp(-tau_m*t))
    sigm_m_t = sigm_m*np.exp(-tau_m*t)
    epsilon_m_mean = np.exp(mu_m + (sigm_m_t**2 / 2))

    #return m_t_N
    return -m_t_N * N * epsilon_m_mean

# Import rest sim replicas to compare
N_tot_vec = np.load('N_tot_vec.npy')
census_yrs = np.load('census_yrs.npy')

sol = solve_ivp(dNdt, [1,30], [0.9*params['K_adult']], t_eval=census_yrs)

fig, axs = plt.subplots(1, 1, figsize=(7,5))
axs.plot(sol.t, sol.y[0], c='k')
axs.plot(census_yrs, N_tot_vec.mean(axis=0), c='g')
axs.set_ylim(0, params['K_adult'])
fig.savefig('nint_test.png', bbox_inches='tight')
