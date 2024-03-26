import numpy as np
import pandas as pd
from scipy.stats import moment

fixed = {'gamm_m': 0.01, 'tau_m': 0.01, 'beta_nu': 0.1, 'gamm_nu': 0.1,
         'K_seedling': 60000, 'kappa': 0.4, 'K_adult': 16000, 'eta': 0.02, 'mu_m': 0.0}

def simulator(params):
    # Assign parameter labels
    alph_m = params[0]; beta_m = params[1]; gamm_m = fixed['gamm_m']
    sigm_m = params[2]; tau_m = fixed['tau_m']
    alph_nu = params[3]; beta_nu = fixed['beta_nu']; gamm_nu = fixed['gamm_nu']
    K_seedling = fixed['K_seedling']; kappa = fixed['kappa']; K_adult = fixed['K_adult']
    eta = fixed['eta']; mu_m = fixed['mu_m']

    # For generating env stochasticity multipliers
    rng = np.random.default_rng()

    # Initialize empty results array
    census_yrs = np.array([1,2,6,8,11,14])
    res_len = len(census_yrs) - 1
    results = np.empty(res_len * 3)
    res_i = 0

    t_vec = np.arange(1,15)
    fn = 'observations/density.csv'
    densities_o = pd.read_csv(fn, header=None)
    densities_o[0] = [round(v) for v in densities_o[0]]
    N_0_1 = densities_o[densities_o[0] == 1][1].to_numpy()
    N_vec = np.ma.array(np.zeros((len(N_0_1), len(t_vec))))
    N_vec[:,0] = N_0_1
    N_vec = N_vec.astype(int)
    census_init = N_vec.sum(axis=1)
    census_yr_init = t_vec[0]

    m_a = alph_m * np.exp(-beta_m*t_vec) + gamm_m
    #K_a = K_seedling * np.exp(-kappa*t_vec) + K_adult
    K_a = K_adult
    nu_a = alph_nu * np.exp(-beta_nu*t_vec) + gamm_nu

    for age_i, t in enumerate(t_vec[:-1]):
        for pop_i, N_pop in enumerate(N_vec):
            # If pop already extirpated, skip
            if (np.ma.is_masked(N_vec)) and (np.ma.getmask(N_vec)[pop_i, 0]):
                continue
            # Add density dependent term to mortalities
            dens_dep = ((nu_a)*(1-m_a)) / (1 + np.exp(-eta*K_adult*(np.sum(N_pop/K_a) - 1)))
            m_a_N = m_a + dens_dep
            # Draw env. stoch. terms and combine for final survival prob.
            epsilon_m = rng.lognormal(np.zeros_like(t_vec)+mu_m, sigm_m*np.exp(-tau_m*t_vec))
            survival_probs = np.exp(-m_a_N * epsilon_m)
            num_survivors = rng.binomial(N_pop, survival_probs)
            num_survivors = np.roll(num_survivors, 1)
            # Finally, update abundances
            N_vec[pop_i] = num_survivors
            # Note if population was extirpated
            if np.sum(num_survivors) == 0:
                #print('pop {} extirpated'.format(pop_i))
                N_vec[pop_i, :] = np.ma.masked
                census_init[pop_i] = np.ma.masked
                #print(N_vec)
        # If enough populations extirpated, consider parameter set invalid
        if (np.ma.is_masked(N_vec)) and (sum(np.ma.getmask(N_vec)[:,0]) > 3):
            results[0:res_len] = np.ones(len(census_yrs)-1)*10
            results[res_len:res_len*2] = np.ones(len(census_yrs)-1)*-1
            results[res_len:res_len*2] = np.ones(len(census_yrs)-1)*20
            #results[res_len*2:res_len*3] = np.ones(len(census_yrs)-1)*20
            #results[res_len:res_len*2 + 1] = np.ones(len(census_yrs))*20
            break
        elif t+1 in census_yrs:
            # Calculate and store mortality stats
            delta_t = (t+1) - census_yr_init
            census_final = N_vec.sum(axis=1)
            mortality = ((census_init - census_final) / census_init) / delta_t
            results[res_i] = np.mean(mortality)
            ### min/max
            #results[res_len + res_i] = np.min(mortality)
            #results[res_len*2 + res_i] = np.max(mortality)
            ### moments
            results[res_len + res_i] = moment(mortality, moment=2)
            results[res_len*2 + res_i] = moment(mortality, moment=3)
            #if (t+1) == 2:
            #    skew_t2 = moment(mortality, moment=3)   
            ### percentiles
            #nan_mort = np.ma.filled(mortality, np.nan)
            #results[res_len + res_i] = np.nanpercentile(nan_mort, 20)
            #results[res_len*2 + res_i] = np.nanpercentile(nan_mort, 80)
            # Reset for next census
            res_i += 1
            census_init = census_final
            census_yr_init = t+1
    #results[res_len*2] = skew_t2
    return results
