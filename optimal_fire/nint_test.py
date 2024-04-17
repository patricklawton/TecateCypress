from scipy.integrate import quad, solve_ivp
import json
import numpy as np
from matplotlib import pyplot as plt

fri = 30

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

def get_num_births(t, N):
    # Fecundity parameters
    rho_max = params['rho_max']; eta_rho = params['eta_rho']; a_mature = params['a_mature']
    sigm_max = params['sigm_max']; eta_sigm = params['eta_sigm'];
    a_sigm_star = a_mature
    # Age-dependent fecundity functions
    rho_t = rho_max / (1+np.exp(-eta_rho*(t-a_mature)))
    sigm_t = sigm_max / (1+np.exp(-eta_sigm*(t-a_sigm_star)))
    # Approximate number of births
    epsilon_rho_mean = np.exp(0 + (sigm_t**2 / 2))
    num_births = rho_t*epsilon_rho_mean*N
    return num_births

# Import rest sim replicas to compare
N_tot_vec = np.load('N_tot_vec.npy')
census_yrs = np.load('census_yrs.npy')
#print(census_yrs)

#sol = solve_ivp(dNdt, [1,fri], [0.9*params['K_adult']], t_eval=census_yrs)

num_intervals = 5
nint_res = np.ones(fri*num_intervals)*np.nan
t_full = np.arange(1, fri*num_intervals+1)
t_eval = np.arange(1, fri+1)
for i in range(1, num_intervals+1):
    if i == 1:
        sol = solve_ivp(dNdt, [1,fri], [0.9*params['K_adult']], t_eval=t_eval)
    else:
        sol = solve_ivp(dNdt, [1,fri], [num_births], t_eval=t_eval)
    num_births = get_num_births(fri, sol.y[0][-1])
    nint_res[(i-1)*fri:i*fri] = sol.y[0]

fig, axs = plt.subplots(2, 1, figsize=(7,10))
#axs.plot(sol.t, sol.y[0], c='k', label='numerical integration')
axs[0].plot(t_full, nint_res, c='k', label='numerical integration')
axs[0].plot(census_yrs, N_tot_vec.mean(axis=0), c='g', label='simulation')
axs[0].set_ylim(0, params['K_adult'])
axs[0].legend()
axs[1].plot(t_full, nint_res, c='k', label='numerical integration')
axs[1].plot(census_yrs, N_tot_vec.mean(axis=0), c='g', label='simulation')
fig.savefig('nint_test.png', bbox_inches='tight')
