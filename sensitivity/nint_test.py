from model import Model
from scipy.integrate import quad, solve_ivp
import os
import json
import numpy as np
# For sampling from various probability distributions
rng = np.random.default_rng()
from matplotlib import pyplot as plt
import timeit
from scipy.special import gamma
from scipy.stats import weibull_min

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle))

# Constants
overwrite_discrete = True
Aeff = 7.29
fri = 50
c = 1.42
b = fri / gamma(1+1/c)
#t_final = 15
t_final = 1000
# Get the average habitat suitability within the Otay Mtn Wilderness area
sdmfn = "SDM_1995.asc"
sdm = np.loadtxt(sdmfn,skiprows=6)
otay = np.loadtxt("otayraster.asc", skiprows=6)
sdm_otay = sdm[otay==1] #index "1" indicates the specific part where study was done
h_o = np.mean(sdm_otay[sdm_otay!=0]) #excluding zero, would be better to use SDM w/o threshold
A_o = 0.1 #area of observed sites in Ha
delta_t = 1
num_reps = 10000
N_0_1 = Aeff*params['K_adult']
N_0_1_vec = np.repeat(N_0_1, num_reps)
init_age = round(params['a_mature']) + 20
t_vec = np.arange(delta_t, t_final+delta_t, delta_t)

# Run discrete simulation
start_time = timeit.default_timer()
discrete_fn = 'nint_data/N_tot_mean_discrete.npy'
t_fire_vec_fn = 'nint_data/t_fire_vec.npy'
if (os.path.isfile(discrete_fn)==False) or (overwrite_discrete):
    model = Model(**params)
    model.set_effective_area(Aeff)
    model.init_N(N_0_1_vec, init_age)
    model.set_weibull_fire(b=b, c=c)
    model.simulate(t_vec=t_vec, census_every=1)
    # Store some results
    N_tot_mean_disc = model.N_tot_vec.mean(axis=0)
    t_fire_vec = model.t_fire_vec
    np.save(discrete_fn, N_tot_mean_disc)
    np.save(t_fire_vec_fn, t_fire_vec)
else:
    N_tot_mean_disc = np.load(discrete_fn)
    t_fire_vec = np.load(t_fire_vec_fn)
#t_fire_vec = np.array([[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0]])
elapsed = timeit.default_timer() - start_time
print('{} seconds'.format(elapsed))

# Set up functions for instantaneous change rates
def dNdt(t, N):
    # Mortality parameters
    alph_m = params['alph_m']; beta_m = params['beta_m']; gamm_m = params['gamm_m']
    sigm_m = params['sigm_m']; tau_m = params['tau_m']; mu_m = params['mu_m']
    alph_nu = params['alph_nu']; beta_nu = params['beta_nu']; gamm_nu = params['gamm_nu']
    K_seedling = params['K_seedling']; kappa = params['kappa']; K_adult = params['K_adult'] 

    # Age-dependent mortality functions
    m_t = alph_m * np.exp(-beta_m*t) + gamm_m
    #K_t = K_adult
    K_t = K_seedling * np.exp(-kappa*t) + K_adult
    nu_t = alph_nu * np.exp(-beta_nu*t) + gamm_nu
    delta, theta = (1.05, 0.050000000000000044) #just hardcoding these in
    eta_t = (theta*2)/((nu_t*(1-m_t)) * (A_o*h_o*K_adult) * (delta-1))
    dens_dep = ((nu_t)*(1-m_t)) / (1 + np.exp(-eta_t*K_adult*(N/K_t - Aeff)))
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

start_time = timeit.default_timer()
N_tot_vec = np.empty((t_fire_vec.shape[0], len(t_vec)))
for pop_i, t_fire_pop in enumerate(t_fire_vec):
    fire_indices = np.argwhere(t_fire_pop!=0).flatten()
    #print(fire_indices)
    for fire_num, fire_i in enumerate(fire_indices):
        #print(f'fire_num {fire_num}')
        if fire_i == min(fire_indices):
            t_eval = np.arange(delta_t, fire_i+delta_t)
            #print(f"t_eval: {t_eval}")
            init_i = 0
            sol = solve_ivp(dNdt, [delta_t,fire_i], [params['K_adult']], t_eval=t_eval) 
        else:
            t_eval = np.arange(delta_t, fire_i-fire_indices[fire_num-1] + delta_t)
            #print(f"t_eval: {t_eval}")
            init_i = fire_indices[fire_num-1]
            sol = solve_ivp(dNdt, [delta_t,fire_i], [num_births], t_eval=t_eval) 
        '''Lazy and need to fix this but handling interfire periods lt 1 this way'''
        if (len(sol.y)!=0) and (len(sol.y[0]) > 1):
            # Set any abundances < 1 to zero
            sol.y[0] = np.where(sol.y[0] > 1, sol.y[0], 0)
            num_births = get_num_births(len(t_eval), sol.y[0][-1])
            #print(f"solution from timestep {init_i} to {fire_i-1}")
            #print(sol.y)
            N_tot_vec[pop_i][init_i:fire_i] = sol.y[0]
        else:
            N_tot_vec[pop_i][init_i:fire_i+1] = 0.
    if len(t_vec) > fire_i+1:
        fire_num += 1
        #print(f'fire_num {fire_num}')
        t_eval = np.arange(delta_t, len(t_vec) - 1 - fire_i + delta_t)
        #print(f"t_eval: {t_eval}")
        sol = solve_ivp(dNdt, [delta_t,len(t_eval)], [num_births], t_eval=t_eval) 
        #print(f"solution from timestep {fire_i+1} to {len(t_vec)-1}")
        #print(sol.y)
        if (len(sol.y)!=0) and (len(sol.y[0]) > 1):
            N_tot_vec[pop_i][fire_i+1:len(t_vec)] = sol.y[0]
        else: 
            N_tot_vec[pop_i][fire_i+1:len(t_vec)] = 0.
N_tot_mean_nint = N_tot_vec.mean(axis=0)
nint_fn = 'nint_data/N_tot_mean_nint.npy'
np.save(nint_fn, N_tot_mean_nint)
elapsed = timeit.default_timer() - start_time
print('{} seconds'.format(elapsed))
