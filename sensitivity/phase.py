import numpy as np
import signac as sg
import sys
from project import Phase
from itertools import product
import timeit
import os
import json
from global_functions import plot_phase

constants = {}
constants['progress'] = False
constants['overwrite_metrics'] = False
constants['overwrite_scaleparams'] = False
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
#constants['ncell_samples'] = 10
constants['ncell_samples'] = 5
constants['slice_samples'] = 75
#constants['slice_samples'] = 5
constants['baseline_A_min'] = 10 #km^2
constants['baseline_A_max'] = 160
#constants['baseline_A_samples'] = 10
constants['baseline_A_samples'] = 5
#constants['delta_tau_min'] = 0.0
#constants['delta_tau_max'] = 0.0
#constants['delta_tau_samples'] = 1
constants['delta_tau_min'] = -10.0
constants['delta_tau_max'] = 10.0
constants['delta_tau_samples'] = 21
constants['root'] = 0 #For mpi

# Init a phase processor based on above constants
for tauc_method in ["flat", "initlinear", "initinverse"]:
#for tauc_method in ["initlinear", "initinverse"]:
    constants.update({'tauc_method': tauc_method})
    pproc = Phase(**constants) 
    if pproc.rank == 0: print(f"on {tauc_method} tauc_method")
    pproc.initialize()
    #print(pproc.phase.shape)
    #sys.exit()

    # Loop over metrics and initial tau uncertainty
    for metric in pproc.metrics:
        for delta_tau_i, delta_tau in enumerate(pproc.delta_tau_vec):
            if pproc.rank == pproc.root: print(f"on delta_tau={delta_tau}")
            start_time = timeit.default_timer()

            # Apply uncertainty to <tau> distribution
            '''change to pproc.set_tau_expect(delta_tau)...is this line even neccessary?'''
            pproc.tau_expect = pproc.tau_flat + delta_tau

            # Initialize data for visualizing slices of phase at delta tau and C values 
            pproc.phase_deltatau_C = np.ones((len(pproc.C_vec), len(pproc.tau_bin_edges), len(pproc.ncell_vec))) * np.nan
            # Do the same for excess resources; only do this once
            if metric == pproc.metrics[0]:
                pproc.phase_xs_deltatau_C = np.ones((len(pproc.C_vec), len(pproc.tau_bin_edges), len(pproc.ncell_vec))) * np.nan

            # Loop over resource constraint values and alteration slice sizes
            for (C_i, C), (ncell_i, ncell) in product(enumerate(pproc.C_vec), enumerate(pproc.ncell_vec)):
                pproc.prep_rank_samples(metric, ncell)
                pproc.process_samples(metric, delta_tau, C, ncell)

            if pproc.rank == pproc.root:
                elapsed = timeit.default_timer() - start_time
                print('{} seconds to run metric {}'.format(elapsed, metric))

            # Save and plot phase slice matricies
            if pproc.rank == pproc.root:
                for C_i, C in enumerate(pproc.C_vec):
                    phase_slice = pproc.phase_deltatau_C[C_i]
                    if metric == pproc.metrics[0]:
                        phase_xs_slice = pproc.phase_xs_deltatau_C[C_i]
                    data_dir = f"data/Aeff_{pproc.Aeff}/tfinal_{pproc.t_final}/metric_{metric}/deltatau_{delta_tau}/C_{C}/"
                    if not os.path.isdir(data_dir):
                        os.makedirs(data_dir)
                    fn = data_dir + f"/phase_{tauc_method}.npy"
                    with open(fn, 'wb') as handle:
                        np.save(handle, phase_slice)

                    if tauc_method == "flat":
                        tauc_vec = C / pproc.ncell_vec
                    else:
                        tauc_vec = None
                    nochange_dir = f"data/Aeff_{pproc.Aeff}/tfinal_{pproc.t_final}/metric_{metric}/deltatau_{delta_tau}/"
                    fn = nochange_dir + f"nochange_{pproc.tauc_method}.json"
                    with open(fn, 'r') as handle:
                        nochange = json.load(handle)[f'{metric}_expect_nochange']
                    figs_dir = f"figs/Aeff_{pproc.Aeff}/tfinal_{pproc.t_final}/metric_{metric}/deltatau_{delta_tau}/C_{C}/"
                    if not os.path.isdir(figs_dir):
                        os.makedirs(figs_dir)
                    fn = figs_dir + f"/phase_slice_{tauc_method}.png"
                    plot_phase(phase_slice, metric, nochange, pproc.tau_bin_cntrs, pproc.ncell_vec, fn, tauc_vec)
                    if metric == pproc.metrics[0]:
                        fn = figs_dir + f"/phase_xs_slice_{tauc_method}.png"
                        plot_phase(phase_xs_slice, 'xs', 0, pproc.tau_bin_cntrs, pproc.ncell_vec, fn, tauc_vec)
        if pproc.rank == pproc.root:
            fn = f"data/Aeff_{pproc.Aeff}/tfinal_{pproc.t_final}/metric_{metric}/phase_{pproc.tauc_method}.npy"
            np.save(fn, pproc.phase)
