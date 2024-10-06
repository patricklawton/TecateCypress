'''
explicitly set constants in a dictionary
save that dictionary if desired
initialize instance of phase with this dictionary

call method for data preprocessing/reading on rank 0
broadcast to all ranks

'''
import numpy as np
import signac as sg
import sys
from project_new import Phase
from itertools import product
import timeit

constants = {}
constants['progress'] = False
constants['overwrite_metrics'] = False
constants['metrics'] = ['lambda_s']
constants['metric_thresh'] = 0.98
constants['c'] = 1.42
constants['Aeff'] = 7.29
constants['t_final'] = 600
constants['sim_method'] = 'nint'
constants['ul_coord'] = [1500, 2800]
constants['lr_coord'] = [2723, 3905]
constants['final_max_tau'] = np.nan
constants['min_tau'] = 3
constants['A_cell'] = 270**2 / 1e6 #km^2
constants['tau_bw_ratio'] = 50 #For binning initial tau (with uncertainty)
constants['tauc_baseline'] = 200 #years, max(tauc) possible at min(ncell) given C
constants['metric_integrand_ratio'] = 800
constants['ncell_samples'] = 10
constants['slice_samples'] = 10
constants['baseline_A_min'] = 10 #km^2
constants['baseline_A_max'] = 160
constants['baseline_A_samples'] = 3
constants['delta_tau_min'] = -10.0
constants['delta_tau_max'] = 10.0
constants['delta_tau_samples'] = 3
constants['root'] = 0 #For mpi

# Init a phase processor based on above constants
#for tauc_method in ["flat", "scaledtoinit"]:
for tauc_method in ["flat"]:
    constants.update({'tauc_method': tauc_method})
    pproc = Phase(**constants) 
    #for item in pproc.__dict__.items():
    #    print(item)
    pproc.initialize()

    # Loop over metrics and initial tau uncertainty
    #for metric, (delta_tau_i, delta_tau) in product(pproc.metrics, enumerate(pproc.delta_tau_vec)): 
    for metric in pproc.metrics:
        for delta_tau_i, delta_tau in enumerate(pproc.delta_tau_vec):
            start_time = timeit.default_timer()

            # Apply uncertainty to <tau> distribution
            '''change to pproc.set_tau_expect(delta_tau)'''
            pproc.tau_expect = pproc.tau_flat + delta_tau

            # Initialize data for visualizing slices of phase at delta tau and C values 
            phase_deltatau_C = np.ones((len(pproc.C_vec), len(pproc.tau_bin_edges), len(pproc.ncell_vec))) * np.nan
            # Do the same for excess resources; only do this once
            if metric == pproc.metrics[0]:
                phase_xs_deltatau_C = np.ones((len(pproc.C_vec), len(pproc.tau_bin_edges), len(pproc.ncell_vec))) * np.nan

            # Loop over resource constraint values and alteration slice sizes
            for (C_i, C), (ncell_i, ncell) in product(enumerate(pproc.C_vec), enumerate(pproc.ncell_vec)):
                #print(C, ncell)
                '''
                pproc.prep_rank_samples(ncell)
                pproc.process_samples()
                    within process_samples:
                        change_tau(delta_tau, C, ncell)
                        compute_metric_expect()
                pproc.collect_and_store_samples()
        save data for this metric
                '''
                pproc.prep_rank_samples(metric, ncell)
                pproc.process_samples(metric, delta_tau, C, ncell)
