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

project = sg.init_project()

sd_fn = project.fn('shared_data.h5')
if not os.path.isfile(sd_fn):
    b_vec = np.arange(0, 92, 2)
    with sg.H5Store(sd_fn).open(mode='w') as sd:
        sd['b_vec'] = b_vec

A_vec = np.array([1,0.5,5,10,100])
demographic_samples_vec = np.concatenate(([2000], np.repeat(1000, 4)))
for A, demographic_samples in zip(A_vec, demographic_samples_vec):
    existing_samples = project.find_jobs({'A': A})
    demographic_samples -= len(existing_samples) #len(project)
    mort_labels = ['alph_m', 'beta_m', 'sigm_m', 'gamm_nu', 'kappa']
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
        sp = {'params': {}, 'A': A}
        for p_i, p in enumerate(mort_samples[i]):
            sp['params'].update({mort_labels[p_i]: float(p)})
        for p_i, p in enumerate(fec_samples[i]):
            sp['params'].update({fec_labels[p_i]: float(p)})
        job = project.open_job(sp)
        job.init()
