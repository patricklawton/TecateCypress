import numpy as np
import signac as sg
import os
import pickle
import sys
root_splt = os.getcwd().split('/')[:-1]
root_splt.append('model_fitting')
sbi_path = '/'.join(root_splt)
sys.path.insert(1, sbi_path) 
import sbi

demographic_samples = 1
mort_labels = ['alph_m', 'beta_m', 'sigm_m', 'gamm_nu', 'kappa']
fec_labels = ['rho_max', 'eta_rho', 'a_mature', 'sigm_max', 'eta_sigm']
with open('../model_fitting/mortality/posterior.pkl', 'rb') as handle:
    mort_posterior = pickle.load(handle)
mort_samples = mort_posterior.sample(sample_shape=(demographic_samples,))
with open('../model_fitting/mortality/fixed.pkl', 'rb') as handle:
    mort_fixed = pickle.load(handle)
with open('../model_fitting/fecundity/posterior.pkl', 'rb') as handle:
    fec_posterior = pickle.load(handle)
fec_samples = fec_posterior.sample(sample_shape=(demographic_samples,))

b_vec = [22, 66]
q_vec = np.arange(0.1, 1.1, 0.1)
for b in b_vec:
    for q in q_vec:
        r_vec = []
        for i in range(demographic_samples):
            params = {}
            for p_i, p in enumerate(mort_samples[i]):
                params.update({mort_labels[p_i]: float(p)})
            for p_i, p in enumerate(fec_samples[i]):
                params.update({fec_labels[p_i]: float(p)})
            params.update(mort_fixed)
            params['K_adult'] = q*params['K_adult']
            params['K_seedling'] = q*params['K_seedling']
        break
    break

