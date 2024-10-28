import numpy as np
import signac as sg
import sys
from project import Phase
from itertools import product
import timeit
import os
import json

constants = {}
constants['progress'] = False
constants['c'] = 1.42
constants['Aeff'] = 1.0
constants['t_final'] = 600
constants['sim_method'] = 'nint'
constants['ul_coord'] = [1500, 2800]
constants['lr_coord'] = [2723, 3905]
constants['min_tau'] = 2
constants['A_cell'] = 270**2 / 1e6 #km^2
constants['plotting_tau_bw_ratio'] = 30 #For binning initial tau (with uncertainty) in phase slice plots
constants['tauc_baseline'] = 200 #years, max(tauc) possible at min(ncell) given C
constants['ncell_samples'] = 6
#constants['slice_samples'] = 30
constants['slice_samples'] = 30
constants['baseline_A_min'] = 10 #km^2
constants['baseline_A_max'] = 160 * 2.0043963553530753
#constants['baseline_A_samples'] = 20
constants['baseline_A_samples'] = 2
#constants['delta_tau_min'] = 0.0
#constants['delta_tau_max'] = 0.0
#constants['delta_tau_samples'] = 1
#constants['delta_tau_min'] = -10.0
#constants['delta_tau_max'] = 10.0
#constants['delta_tau_samples'] = 81
constants['root'] = 0 #For mpi
constants.update({'final_max_tau': np.nan})
constants['overwrite_results'] = True

# Define metrics and tauc methods to run analysis on
#metrics = ["lambda_s", "mu_s", "r"]
metrics = ["P_s"]
#tauc_methods = ["flat", "initlinear", "initinverse"]
tauc_methods = ["flat"]

# Define uncertainty axes (and save under metric folder later)
mu_tau_vec = np.linspace(-10, 0, 3)
sigm_tau_vec = np.linspace(0, 0, 1)
tauc_eps_samples = 1
mu_tauc_vec = np.linspace(0, 0, tauc_eps_samples)
sigm_tauc_vec = np.linspace(0, 0, tauc_eps_samples)

for metric in metrics:
    constants.update({'metric': metric})
    constants.update({'overwrite_metrics': False}) #Set True to overwrite metrics (only once per metric)
    constants.update({'overwrite_scaleparams': False}) 
    for (tauc_method_i, tauc_method) in enumerate(tauc_methods):
        if tauc_method_i > 0:
            constants.update({'overwrite_metrics': False})
        constants.update({'tauc_method': tauc_method})

        # Init a phase processor based on above constants
        pproc = Phase(**constants) 
        if pproc.rank == 0: print(f"on {tauc_method} tauc_method")
        pproc.initialize()

        # Process data over uncertainty space samples
        #for delta_tau_i, delta_tau in enumerate(pproc.delta_tau_vec):
        for mu_tau, sigm_tau in product(mu_tau_vec, sigm_tau_vec):
            if pproc.rank == pproc.root: print(f"on (mu_tau, sigm_tau)={mu_tau, sigm_tau}")
            pproc.generate_eps_tau(mu_tau, sigm_tau)
            # Calculate <metric> at sampled resource constraint values and alteration slice sizes
            if pproc.rank == pproc.root: start_time = timeit.default_timer()
            for (C_i, C), (ncell_i, ncell) in product(enumerate(pproc.C_vec), enumerate(pproc.ncell_vec)):
                pproc.prep_rank_samples(ncell)
                pproc.process_samples(C, ncell)
            if pproc.rank == pproc.root:
                elapsed = timeit.default_timer() - start_time
                print('{} seconds to run metric {}'.format(elapsed, metric))
            # Plot slices of phase matricies
            if pproc.rank == pproc.root:
                for C in pproc.C_vec:
                    pproc.plot_phase_slice(C)
                    pproc.plot_phase_slice(C, xs=True)
            # Save phase matrix at this uncertainty parameterization
            pproc.store_phase()
