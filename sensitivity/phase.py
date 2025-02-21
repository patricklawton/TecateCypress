import numpy as np
import signac as sg
import sys
from project import Phase
from itertools import product
import timeit
import os
import h5py
from tqdm import tqdm

constants = {}
constants['progress'] = False
constants['c'] = 1.42
#constants['Aeff'] = 1.0
constants['Aeff'] = 7.29
#constants['t_final'] = 600
constants['t_final'] = 300
constants['sim_method'] = 'nint'
constants['ul_coord'] = [1500, 2800]
constants['lr_coord'] = [2723, 3905]
'''Should just set this to min tau_vec I think, which it basically already is'''
constants['min_tau'] = 2
constants['A_cell'] = 270**2 / 1e6 #km^2
constants['plotting_tau_bw_ratio'] = 30 #For binning initial tau (with uncertainty) in phase slice plots
constants['tauc_baseline'] = 200 #years, max(tauc) possible at min(ncell) given C
#constants['ncell_samples'] = 30
constants['ncell_samples'] = 15
#constants['slice_samples'] = 30
constants['slice_samples'] = 30
constants['baseline_A_min'] = 10 #km^2
#constants['baseline_A_max'] = 160 * 2.0043963553530753
constants['baseline_A_max'] = (160 * 2.0043963553530753) * 2
#constants['baseline_A_samples'] = 20
constants['baseline_A_samples'] = 5
constants['root'] = 0 #For mpi
constants.update({'final_max_tau': np.nan})
constants['overwrite_results'] = True
#constants['meta_metric'] = 'distribution_avg'
constants['meta_metric'] = 'gte_thresh'

# Define metrics and tauc methods to run analysis on
#metrics = ["lambda_s", "mu_s", "r"]
metrics = ["P_s"]
#tauc_methods = ["flat", "initlinear", "initinverse"]
tauc_methods = ["flat"]

# Define uncertainty axes (and save under metric folder later)
mu_tau_vec = np.linspace(-10, 0, 6)
sigm_tau_vec = np.linspace(0, 10, 5)
mu_tauc_vec = np.linspace(-10, 0, 6)
sigm_tauc_vec = np.linspace(0, 10, 5)
#mu_tau_vec = np.linspace(0, 0, 1)
#sigm_tau_vec = np.linspace(0, 0, 1)
#mu_tauc_vec = np.linspace(0, 0, 1)
#sigm_tauc_vec = np.linspace(0, 0, 1)

total_computations = len(metrics) * len(tauc_methods) * len(mu_tau_vec) * len(sigm_tau_vec)
total_computations = total_computations * len(mu_tauc_vec) * len(sigm_tauc_vec)
num_computations_finished = 0
with tqdm(total=total_computations) as pbar:
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
            if pproc.rank == pproc.root: print(f"on {tauc_method} tauc_method")
            pproc.initialize()

            # Save uncertainty axes to file
            if (pproc.rank == pproc.root):
                fn = pproc.data_dir + "/eps_axes.h5"
                if (not os.path.isfile(fn)) or pproc.overwrite_results:
                    with h5py.File(fn, "w") as handle:
                        handle['mu_tau'] = mu_tau_vec
                        handle['sigm_tau'] = sigm_tau_vec
                        handle['mu_tauc'] = mu_tauc_vec
                        handle['sigm_tauc'] = sigm_tauc_vec

            # Process data over uncertainty space samples
            for mu_tau, sigm_tau in product(mu_tau_vec, sigm_tau_vec):
                # Generate uncertainties on initial tau values
                pproc.generate_eps_tau(mu_tau, sigm_tau)
                for mu_tauc, sigm_tauc in product(mu_tauc_vec, sigm_tauc_vec):
                    if pproc.rank == pproc.root: print(f"on (mu_tau, sigm_tau, mu_tauc, sigm_tauc)={mu_tau, sigm_tau, mu_tauc, sigm_tauc}")
                    # Generate uncertainties on alterations to tau values
                    pproc.generate_eps_tauc(mu_tauc, sigm_tauc)
                    # Calculate <metric> at sampled resource constraint values and alteration slice sizes
                    if pproc.rank == pproc.root: start_time = timeit.default_timer()
                    for (C_i, C), (ncell_i, ncell) in product(enumerate(pproc.C_vec), enumerate(pproc.ncell_vec)):
                        pproc.prep_rank_samples(ncell)
                        pproc.process_samples(C, ncell)
                    if pproc.rank == pproc.root:
                        elapsed = timeit.default_timer() - start_time
                        print('{} seconds to run metric {}'.format(elapsed, metric))
                        pbar.update(1)
                    # Save phase matrix at this uncertainty parameterization
                    pproc.store_phase()
                    ## Plot slices of phase matricies
                    #for C in pproc.C_vec:
                    #    pproc.plot_phase_slice(C)
                    #    pproc.plot_phase_slice(C, xs=True)
