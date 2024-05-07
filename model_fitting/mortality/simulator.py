import numpy as np
import pandas as pd
from scipy.stats import moment

#fixed = {'gamm_m': 0.01, 'tau_m': 0.01, 'beta_nu': 0.1, 'gamm_nu': 0.1,
#         'K_seedling': 160000, 'kappa': 0.4, 'K_adult': 16000, 'eta': 0.02, 'mu_m': 0.0}
fixed = {'gamm_m': 0.01, 'tau_m': 0.01, 'beta_nu': 0.1, 'gamm_nu': 0.1,
         'kappa': 0.4, 'eta': 0.02, 'mu_m': 0.0}

def simulator(params):
    # Assign parameter labels
    alph_m = params[0]; beta_m = params[1]; gamm_m = fixed['gamm_m']
    sigm_m = params[2]; tau_m = fixed['tau_m']
    alph_nu = params[3]; beta_nu = fixed['beta_nu']; gamm_nu = fixed['gamm_nu']
    kappa = fixed['kappa']; K_adult = params[4]
    eta = fixed['eta']; mu_m = fixed['mu_m']

    # For generating env stochasticity multipliers
    rng = np.random.default_rng()

    # Initialize empty results array
    census_yrs = np.array([1,2,6,8,11,14])
    res_len = len(census_yrs) - 1
    results = np.empty(res_len * 3)
    res_i = 0

    t_vec = np.arange(1,15)
    fn = 'mortality/observations/density.csv'
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
    K_a = np.repeat(K_adult, len(t_vec))
    nu_a = alph_nu * np.exp(-beta_nu*t_vec) + gamm_nu
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
                dens_dep = ((nu_a)*(1-m_a)) / (1 + np.exp(-eta*K_adult*(np.sum(N_vec[pop_i]/K_a) - 1)))
                m_a_N = m_a + dens_dep
                survival_probs = np.exp(-m_a_N * epsilon_m_vec[pop_i])
                # Make it deterministic
                #survival_probs = np.exp(-m_a_N * epsilon_m_mean)
            else:
                dens_dep = ((nu_a[age_i])*(1-m_a[age_i])) / (1 + np.exp(-eta*K_adult*(N/K_a[age_i] - 1)))
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
            results[res_len*2:res_len*3] = np.ones(len(census_yrs)-1)*np.nan
            break
        elif t+1 in census_yrs:
            # Calculate and store mortality stats
            delta_t = (t+1) - census_yr_init
            census_final = N_vec.sum(axis=1)
            mortality = ((census_init - census_final) / census_init) / delta_t
            # Use the first three moments
            results[res_i] = np.mean(mortality)
            results[res_len + res_i] = moment(mortality, moment=2)
            results[res_len*2 + res_i] = moment(mortality, moment=3)
            #if (t+1) == 2:
            #    skew_t2 = moment(mortality, moment=3)   
            # Reset for next census
            res_i += 1
            census_init = census_final
            census_yr_init = t+1
    #results[res_len*2] = skew_t2
    return results
