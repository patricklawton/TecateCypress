from scipy.integrate import quad, solve_ivp
import json
import numpy as np
from matplotlib import pyplot as plt
import timeit

fri = 40

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle))

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
    sigm_m_t = sigm_m*np.exp(-tau_m*t)
    epsilon_m_mean = np.exp(mu_m + (sigm_m_t**2 / 2))
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
census_t = np.load('census_t.npy')
delta_t = census_t[1] - census_t[0]
N_tot_vec_mean = N_tot_vec.mean(axis=0)
#print(census_t)
sim_mort = np.ones_like(N_tot_vec_mean) * np.nan
for i in range(len(N_tot_vec_mean)-1):
    diff_per_N = (N_tot_vec_mean[i] - N_tot_vec_mean[i+1]) / N_tot_vec_mean[i]
    mort = diff_per_N / delta_t
    if mort > 0:
        sim_mort[i] = mort

#sol = solve_ivp(dNdt, [1,fri], [0.9*params['K_adult']], t_eval=census_t)
start_time = timeit.default_timer()
num_intervals = 2
interval_steps = round(fri/delta_t)
nint_res = np.ones(interval_steps*num_intervals)*np.nan
t_full = np.arange(delta_t, round(fri*num_intervals)+delta_t, delta_t)
t_eval = np.arange(delta_t, fri+delta_t, delta_t)
for i in range(1, num_intervals+1):
    if i == 1:
        sol = solve_ivp(dNdt, [delta_t,fri], [0.9*params['K_adult']], t_eval=t_eval)
    else:
        sol = solve_ivp(dNdt, [delta_t,fri], [num_births], t_eval=t_eval)
    num_births = get_num_births(fri, sol.y[0][-1])
    nint_res[(i-1)*interval_steps:i*interval_steps] = sol.y[0]
nint_mort = np.ones_like(nint_res) * np.nan
for i in range(len(nint_res)-1):
    prop_diff = (nint_res[i] - nint_res[i+1]) / nint_res[i]
    mort = prop_diff / delta_t
    if mort > 0:
        nint_mort[i] = mort
elapsed = timeit.default_timer() - start_time
print(elapsed)

fig, axs = plt.subplots(3, 1, figsize=(8,15))
#axs.plot(sol.t, sol.y[0], c='k', label='numerical integration')
axs[0].plot(census_t, N_tot_vec_mean, c='g', label='simulation')
axs[0].plot(t_full, nint_res, c='k', label='numerical integration')
axs[0].set_ylim(0, params['K_adult'])
axs[0].legend()
axs[1].plot(t_full, nint_res, c='k', label='numerical integration')
axs[1].plot(census_t, N_tot_vec_mean, c='g', label='simulation')
#axs[1].set_ylim(0, 60*params['K_adult'])
#axs[2].plot(census_t, N_tot_vec_mean, c='g', label='simulation')
#axs[2].plot(t_full, nint_res, c='k', label='numerical integration')
axs[2].plot(census_t, sim_mort, c='g', label='simulation')
axs[2].plot(t_full, nint_mort, c='k', label='numerical integration')
#axs[2].set_ylim(0.01, 0.25); axs[2].set_xlim(0,20)
fig.savefig('nint_test.png', bbox_inches='tight')
