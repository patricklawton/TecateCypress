import numpy as np
import pandas as pd
from scipy.stats import moment
import pickle

# Get the average habitat suitability within the Otay Mtn Wilderness area
#sdmfn = "mortality/SDM_1995.asc"
sdmfn = "../shared_maps/SDM_1995.asc"
sdm = np.loadtxt(sdmfn,skiprows=6)
#otay = np.loadtxt("mortality/otayraster.asc", skiprows=6)
otay = np.loadtxt("../shared_maps/otayraster.asc", skiprows=6)
sdm_otay = sdm[otay==1] #index "1" indicates the specific part where study was done
h_o = np.mean(sdm_otay[sdm_otay!=0]) #excluding zero, would be better to use SDM w/o threshold

fixed = {'tau_m': 0.0, 'mu_m': 0.0, 
         'alph_nu': 0.0, 'beta_nu': 0.0, 'gamm_nu': 0.0, 
         'kappa': 0.0, 'K_adult': 1_000.0, 'K_seedling': 0.0}
         #'K_seedling': 60_000/h_o, 'K_adult': 10_000/h_o}
with open('mortality/fixed.pkl', 'wb') as handle:
    pickle.dump(fixed, handle)

def simulator(params):
    # Assign parameter labels
    alph_m = params[0]; beta_m = params[1]; gamm_m = params[2]
    sigm_m = params[3]; tau_m = fixed['tau_m']; mu_m = fixed['mu_m']
    alph_nu = fixed['alph_nu']; beta_nu = fixed['beta_nu']; gamm_nu = fixed['gamm_nu']
    K_seedling = fixed['K_seedling']; kappa = fixed['kappa']; K_adult = fixed['K_adult']; 

    # For generating env stochasticity multipliers
    rng = np.random.default_rng()

    # Initialize empty results array
    census_yrs = np.array([1,2,6,8,11,14])
    res_len = len(census_yrs) - 1
    results = np.empty(res_len * 2)
    res_i = 0

    t_vec = np.arange(1,15)
    fn = 'mortality/observations/density.csv'
    densities_o = pd.read_csv(fn, header=None)
    densities_o[0] = [round(v) for v in densities_o[0]]
    A_o = 0.1 #area of observed sites in Ha
    N_0_1 = densities_o[densities_o[0] == 1][1].to_numpy() * A_o
    N_vec = np.ma.array(np.zeros((len(N_0_1), len(t_vec))))
    N_vec[:,0] = N_0_1
    N_vec = N_vec.astype(int)
    census_init = N_vec.sum(axis=1)
    census_yr_init = t_vec[0]

    m_a = alph_m * np.exp(-beta_m*t_vec) + gamm_m
    K_a = K_seedling * np.exp(-kappa*t_vec) + K_adult
    #K_a = np.repeat(K_adult, len(t_vec))
    nu_a = alph_nu * np.exp(-beta_nu*t_vec) + gamm_nu
    # Use linear approx to set eta s.t. shape of dens. dep. curve is 
    # the same for arbitrary effective patch size
    #eta_a = 2 / ((nu_a*(1-m_a)) * (A_o*h_o) * K_adult)
    eta_a = np.repeat(1, len(m_a))
    sigm_m_a = sigm_m*np.exp(-tau_m*t_vec)
    epsilon_m_vec = rng.lognormal(np.zeros_like(N_vec)+mu_m, np.tile(sigm_m_a, (len(N_0_1),1)))

    for age_i, t in enumerate(t_vec[:-1]):
        for pop_i, N_pop in enumerate(N_vec):
            # If pop already extirpated, skip
            if (np.ma.is_masked(N_vec)) and (np.ma.getmask(N_vec)[pop_i, 0]):
                continue
            age_i_vec = np.nonzero(N_vec[pop_i])[0]
            if len(age_i_vec) > 1:
                single_age = False
            else:
                single_age = True
                age_i = age_i_vec[0]
                N = N_vec[pop_i][age_i]
            if not single_age:
                dens_dep = ((nu_a)*(1-m_a)) / (1 + np.exp(-eta_a*K_adult*(np.sum(N_vec[pop_i]/K_a) - A_o*h_o)))
                m_a_N = m_a + dens_dep
                survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i])
                # Make it deterministic
                #survival_probs = np.exp(-m_a_N * epsilon_m_mean)
            else:
                dens_dep = ((nu_a[age_i])*(1-m_a[age_i])) / (1 + np.exp(-eta_a[age_i]*K_adult*(N/K_a[age_i] - A_o*h_o)))
                m_a_N = m_a[age_i] + dens_dep
                survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i][age_i])
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
                    # Note if population was extirpated
                    if np.sum(num_survivors) == 0:
                        N_vec[pop_i, :] = np.ma.masked
                        census_init[pop_i] = np.ma.masked
            else:
                N_vec[:, :] = np.ma.masked
        # If enough populations extirpated, consider parameter set invalid
        if (np.ma.is_masked(N_vec)) and (sum(np.ma.getmask(N_vec)[:,0]) > 3):
            results[0:res_len] = np.ones(len(census_yrs)-1)*np.nan
            results[res_len:res_len*2] = np.ones(len(census_yrs)-1)*np.nan
            #results[res_len*2:res_len*3] = np.ones(len(census_yrs)-1)*np.nan
            break
        elif t+1 in census_yrs:
            # Calculate and store mortality stats
            delta_t = (t+1) - census_yr_init
            census_final = N_vec.sum(axis=1)
            mortality = ((census_init - census_final) / census_init) / delta_t
            # Use the first three moments
            results[res_i] = np.mean(mortality)
            results[res_len + res_i] = moment(mortality, moment=2)
            #results[res_len*2 + res_i] = moment(mortality, moment=3)
            # Reset for next census
            res_i += 1
            census_init = census_final
            census_yr_init = t+1
    return results
