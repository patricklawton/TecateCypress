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
from itertools import product

try:
    project = sg.get_project()
except:
    project = sg.init_project()

overwrite_bvec = True
sd_fn = project.fn('shared_data.h5')
if not os.path.isfile(sd_fn) or overwrite_bvec:
    #b_vec = np.arange(0, 140, 4)
    b_vec = np.concatenate(([0], np.arange(2, 140, 8)))
    with sg.H5Store(sd_fn).open(mode='w') as sd:
        sd['b_vec'] = b_vec

A_cell = 270**2 / 10_000 #Ha
sdm_min = 0.32649827003479004 
sdm_mean = 0.51
#Aeff_vec = np.array([np.round(A_cell*sdm_min, 2)])
Aeff_vec = np.array([np.round(A_cell, 2)])
#Aeff_vec = np.array([np.round((4*A_cell)*sdm_mean, 2)])
#Aeff_vec = np.array([1.0])
t_final_vec = np.array([300])
demographic_samples_vec = np.array([5_000])
method_vec = ["discrete"]

for Aeff, t_final, demographic_samples, method in product(Aeff_vec, t_final_vec, 
                                                          demographic_samples_vec, 
                                                          method_vec):
    existing_samples = project.find_jobs({'Aeff': float(Aeff), 't_final': int(t_final), 'method': str(method)})
    demographic_samples -= len(existing_samples) #len(project)
    mort_labels = ['alph_m', 'beta_m', 'gamm_m', 'sigm_m', 'gamm_nu', 'kappa', 'K_adult']
    fec_labels = ['rho_max', 'eta_rho', 'a_mature', 'sigm_max', 'eta_sigm']

    if demographic_samples > 0:
        with open('../model_fitting/mortality/posterior.pkl', 'rb') as handle:
            mort_posterior = pickle.load(handle)
        mort_samples = mort_posterior.sample(sample_shape=(demographic_samples,))
        with open('../model_fitting/fecundity/posterior.pkl', 'rb') as handle:
            fec_posterior = pickle.load(handle)
        fec_samples = fec_posterior.sample(sample_shape=(demographic_samples,))

    #params = mort_fixed
    for i in range(demographic_samples):
        sp = {'params': {}, 'Aeff': Aeff, 't_final': t_final, 'method': method, 
              'demographic_index': len(existing_samples) + i + 1}
        for p_i, p in enumerate(mort_samples[i]):
            sp['params'].update({mort_labels[p_i]: float(p)})
        for p_i, p in enumerate(fec_samples[i]):
            sp['params'].update({fec_labels[p_i]: float(p)})
        job = project.open_job(sp)
        job.init()
