import numpy as np

# Brennan 2019
num_sites = np.array([2,4,4,4,2])
stand_age_bren = np.array([7,24,33,42,53])
prefire_density = np.array([2205,48,3430,233,400])
prefire_density_se = np.array([35,14,308,136,240])
prefire_density_sd = prefire_density_se * np.sqrt(num_sites)
rec_density = np.array([0,10000,18500,29000,182000])
rec_density_se = np.array([0,2345,9385,18443,92000])
rec_density_sd = rec_density_se * np.sqrt(num_sites)
fecundity_bren = rec_density / prefire_density
fecundity_err_bren = np.zeros(len(fecundity_bren))
for i in range(len(fecundity_err_bren)):
    if (prefire_density[i] == 0) or (rec_density[i] == 0):
        fecundity_err = 0
    else:
        fecundity_err = fecundity_bren[i] * np.sqrt((prefire_density_se[i]/prefire_density[i])**2 + (rec_density_se[i]/rec_density[i])**2)
        #fecundity_err = fecundity_bren[i] * np.sqrt((prefire_density_sd[i]/prefire_density[i])**2 + (rec_density_sd[i]/rec_density[i])**2)
    fecundity_err_bren[i] = fecundity_err

# Dunn 1986
stand_age_dunn = np.array([10,19,20,20,30,36,39,63])
fecundity_dunn = (np.array([0,0.1,2.9,26.5,15.7,1206.5,1387.3,1400]) / 100) + 1
# Assume something ad hoc for CV on Dunn's measurements
fecundity_err_dunn = np.repeat(0.75, len(fecundity_dunn)) * fecundity_dunn

stand_age_all = np.concatenate((stand_age_bren, stand_age_dunn))
sort = np.argsort(stand_age_all)
stand_age_all = stand_age_all[sort]
fecundity_all = np.concatenate((fecundity_bren, fecundity_dunn))[sort]
fecundity_err_all = np.concatenate((fecundity_err_bren,fecundity_err_dunn))[sort]
weights = np.concatenate((num_sites, np.ones(len(stand_age_dunn))))[sort]
binwidth = 22
numbins = max(stand_age_all) // binwidth + (max(stand_age_all) % binwidth > 0)
age_cntrs = np.zeros(numbins)
mean_fecundities = []
bin_errors = []
for i in range(numbins):
    filt = ((stand_age_all > binwidth*i) & (stand_age_all <= binwidth*(i+1)))
    avg = np.average(fecundity_all[filt], weights=weights[filt])
    mean_fecundities.append(avg)
    age_cntr = binwidth*i + binwidth/2
    age_cntrs[i] = age_cntr
    err = (1/sum(weights[filt]))*np.sqrt(sum((fecundity_err_all[filt]*weights[filt])**2))
    bin_errors.append(err)

def save_observations():
    #observations = np.concatenate((mean_fecundities, bin_errors))
    observations = np.concatenate((mean_fecundities, bin_errors, [0.2]))
    np.save('fecundity/observations/observations.npy', observations)

def simulator(params):
    rho_max = params[0]; eta_rho = params[1]; a_mature = params[2]
    sigm_max = params[3]; eta_sigm = params[4]; a_sigm_star = params[5]

    # Read this in from file(s) in the actual script
    #a_vec = np.concatenate((np.repeat(stand_age_bren, num_sites), stand_age_dunn))
    a_vec = np.concatenate((
                            np.repeat(stand_age_bren, num_sites), 
                            stand_age_dunn,
                            [numbins*binwidth + 1, (numbins+1)*binwidth]
                           ))
    a_vec.sort()

    rho_a = rho_max / (1+np.exp(-eta_rho*(a_vec-a_mature)))
    rng = np.random.default_rng()
    sigm_a = sigm_max / (1+np.exp(-eta_sigm*(a_vec-a_sigm_star)))
    epsilon_rho = rng.lognormal(np.zeros_like(a_vec), sigm_a)
    fecundities = rng.poisson(rho_a*epsilon_rho)

    mean_fecundities = np.zeros(numbins)
    bin_stdevs = np.zeros(numbins)
    results = np.empty(numbins*2 + 1)
    for i in range(numbins):
        filt = ((a_vec > binwidth*i) & (a_vec <= binwidth*(i+1)))
        fecundities_bin = fecundities[filt]
        mean_fecundities[i] = fecundities_bin.mean()
        bin_stdevs[i] = np.std(fecundities_bin, ddof=1)
    results[0:numbins] = mean_fecundities
    results[numbins:numbins*2] = bin_stdevs

    # Fecundity should level out by end of last bin
    # check using mean epsilon value
    mean_diff = rho_a[-1]*np.exp(sigm_a[-1]**2 / 2) - rho_a[-2]*np.exp(sigm_a[-2]**2 / 2)
    results[-1] = mean_diff
    return results
