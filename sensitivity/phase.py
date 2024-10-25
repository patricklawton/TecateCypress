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
constants['metric_thresh'] = 0.98
constants['c'] = 1.42
constants['Aeff'] = 1.0
constants['t_final'] = 600
constants['sim_method'] = 'nint'
constants['ul_coord'] = [1500, 2800]
constants['lr_coord'] = [2723, 3905]
constants['min_tau'] = 3
constants['A_cell'] = 270**2 / 1e6 #km^2
constants['plotting_tau_bw_ratio'] = 30 #For binning initial tau (with uncertainty) in phase slice plots
constants['tauc_baseline'] = 200 #years, max(tauc) possible at min(ncell) given C
constants['ncell_samples'] = 20
#constants['ncell_samples'] = 20
#constants['slice_samples'] = 30
constants['slice_samples'] = 100
constants['baseline_A_min'] = 10 #km^2
constants['baseline_A_max'] = 160 * 2.0043963553530753
constants['baseline_A_samples'] = 20
#constants['baseline_A_samples'] = 3
#constants['delta_tau_min'] = 0.0
#constants['delta_tau_max'] = 0.0
#constants['delta_tau_samples'] = 1
constants['delta_tau_min'] = -10.0
constants['delta_tau_max'] = 10.0
constants['delta_tau_samples'] = 81
constants['root'] = 0 #For mpi
constants.update({'final_max_tau': np.nan})

#metrics = ["lambda_s", "mu_s", "r"]
metrics = ["mu_s"]
tauc_methods = ["flat", "initlinear", "initinverse"]
#tauc_methods = ["flat"]
# Loop over tauc_methods and metrics 
for metric in metrics:
    constants.update({'metric': metric})
    # Set True to overwrite metrics (only once per metric)
    constants.update({'overwrite_metrics': False}) 
    if metric == metrics[0]:
        # Set True to overwrite scaling params (only once per tauc_method)
        constants.update({'overwrite_scaleparams': False})
    else:
        constants.update({'overwrite_scaleparams': False}) 

    for (tauc_method_i, tauc_method) in enumerate(tauc_methods):
        if tauc_method_i > 0:
            constants.update({'overwrite_metrics': False})
        constants.update({'tauc_method': tauc_method})

        # Init a phase processor based on above constants
        pproc = Phase(**constants) 
        if pproc.rank == 0: print(f"on {tauc_method} tauc_method")
        pproc.initialize()

        # Loop over initial tau uncertainty
        for delta_tau_i, delta_tau in enumerate(pproc.delta_tau_vec):
            if pproc.rank == pproc.root: print(f"on delta_tau={delta_tau}")
            start_time = timeit.default_timer()

            # Apply uncertainty to <tau> distribution
            '''change to pproc.set_tau_expect(delta_tau)...is this line even neccessary?'''
            pproc.tau_expect = pproc.tau_flat + delta_tau

            # Calculate <metric> at sampled resource constraint values and alteration slice sizes
            for (C_i, C), (ncell_i, ncell) in product(enumerate(pproc.C_vec), enumerate(pproc.ncell_vec)):
                pproc.prep_rank_samples(metric, ncell)
                pproc.process_samples(metric, delta_tau, C, ncell)

            if pproc.rank == pproc.root:
                elapsed = timeit.default_timer() - start_time
                print('{} seconds to run metric {}'.format(elapsed, metric))

            # Plot slices of phase matricies
            if pproc.rank == pproc.root:
                for C in pproc.C_vec:
                    pproc.plot_phase_slice(delta_tau, C)
                    pproc.plot_phase_slice(delta_tau, C, xs=True)
        #if pproc.rank == pproc.root:
        #    fn = f"data/Aeff_{pproc.Aeff}/tfinal_{pproc.t_final}/metric_{metric}/phase_{pproc.tauc_method}.npy"
        #    np.save(fn, pproc.phase)
