import numpy as np
from project import Phase
import torch
import sys
from matplotlib import pyplot as plt
from mpi4py import MPI
from tqdm import tqdm
import timeit
from itertools import product
import os
import h5py

rng = np.random.default_rng()

constants = {}
constants['progress'] = False
constants['c'] = 1.42
constants['Aeff'] = 7.29
constants['t_final'] = 300
constants['sim_method'] = 'discrete'
constants['ul_coord'] = [1500, 2800]
constants['lr_coord'] = [2723, 3905]
'''Should just set this to min tau_vec I think, which it basically already is'''
constants['min_tau'] = 2
constants['A_cell'] = 270**2 / 1e6 #km^2
constants['ncell_min'] = 2_500
constants['root'] = 0 #For mpi
constants.update({'final_max_tau': np.nan})
constants['meta_metric'] = 'gte_thresh'
constants.update({'metric': 'lambda_s'})
constants.update({'tauc_method': 'flat'})
constants.update({'overwrite_metrics': False}) 
constants['overwrite_results'] = True

constants['plotting_tau_bw_ratio'] = 30 #For binning initial tau (with uncertainty) in phase slice plots

# Get list of samples for each parameter
constants['tauc_min_samples'] = np.array([9.0])
constants['ncell_samples'] = 15
constants['slice_samples'] = 30
#constants['ncell_samples'] = 5
#constants['slice_samples'] = 10
mu_tau_vec = np.arange(-9, 6, 3).astype(float)
sigm_tau_vec = np.linspace(0, 10, 3)
mu_tauc_vec = np.arange(-9, 6, 3).astype(float)
sigm_tauc_vec = np.linspace(0, 10, 3)

# Start timer to track runtime
start_time = timeit.default_timer()

# Define ordered list of parameter keys
param_keys = ['C', 'ncell', 'slice_left',
              'mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc']

# Initialize "Phase" instance for processing samples
pproc = Phase(**constants)
pproc.initialize()
pproc.init_strategy_variables()
assert pproc.metric_exp_spl(pproc.min_tau) > 0

if pproc.rank != pproc.root:
    x_all = None
else:
    # Theoretical (or ad-hoc) maxima/minima for parameters
    minima = {
        # 'C': 5.*pproc.ncell_tot,
        # 'ncell': int(0.02*pproc.ncell_tot),
        # 'slice_left': int(0.* pproc.ncell_tot),
        'mu_tau': -9.,
        'sigm_tau': 0.,
        'mu_tauc': -9.,
        'sigm_tauc': 0.
    }
    maxima = {
        # 'C': 5.*pproc.ncell_tot,
        # 'ncell': int(1. * pproc.slice_right_max),
        # 'slice_left': int(1.*pproc.ncell_tot),
        'mu_tau': 3.,
        'sigm_tau': 7.,
        'mu_tauc': 0.,
        'sigm_tauc': 3.
    }

    ### Initialize strategy combinations ### 
    num_strategy_combs = pproc.C_vec.size * pproc.ncell_vec.size * pproc.slice_left_all.size

    strategy_combs = product(pproc.C_vec,
                    pproc.ncell_vec,
                    pproc.slice_left_all)

    # Initialize strategy parameter combinations
    x_strategy = np.full((num_strategy_combs, 3), np.nan)

    # Filter out any invalid param samples
    for strategy_comb_i, strategy_comb in enumerate(strategy_combs):
        # Check that slice is within allow range
        if strategy_comb[2] > (pproc.slice_right_max - strategy_comb[1]): continue
        x_strategy[strategy_comb_i, :] = strategy_comb
    nan_filt = np.any(np.isnan(x_strategy), axis=1)
    x_strategy = x_strategy[~nan_filt, :]

    # Recompute number of strategy combinations
    num_strategy_combs = x_strategy.shape[0]

    ### Add on samples of uncertain samples ###
    # First define the number of eps samples per strategy combination
    num_eps_combs = 225
    #num_eps_combs = 500
    #num_eps_combs = 100
    num_combs_tot = num_strategy_combs * num_eps_combs

    # Initialize x_all with repeats of strategy combinations
    x_all = np.tile(x_strategy, (num_eps_combs, 1))

    # Fill columns with samples of uncertain parameters
    uncertain_params = ['mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc']

    # Add columns for uncertain param samples
    x_all = np.hstack((
                          x_all, 
                          np.full((x_all.shape[0], len(uncertain_params)), np.nan)
                      ))

    # Loop over each uncertain param to populate in x_all
    for i, uncertain_param in enumerate(uncertain_params):
        uncertain_param_i = i + 3
        # First insert some zeros bc we want to reference these in our analysis
        '''Reserve the demo sample index 0 for mean lambda(tau)'''
        x_all[:num_strategy_combs, uncertain_param_i] = np.zeros(num_strategy_combs)

        # Get min, max and type for this param
        _min = minima[uncertain_param]
        _max = maxima[uncertain_param]
        assert (type(_min)==type(_min))
        _type = type(_min)

        # Generate and store nonzero samples
        if _type == float:
            samples = rng.uniform(_min, _max, num_combs_tot-num_strategy_combs)
        elif _type == int:
            samples = rng.integers(_min, _max, num_combs_tot-num_strategy_combs)
        x_all[num_strategy_combs:, uncertain_param_i] = samples

    # Shuffle to give all procs ~ the same amount of work
    pproc.rng.shuffle(x_all)

    # Save to file for reference in postprocessing
    np.save(pproc.data_dir + '/x_all.npy', x_all)

    # Save uncertainty axes to file
    fn = pproc.data_dir + "/eps_axes.h5"
    if (not os.path.isfile(fn)) or pproc.overwrite_results:
        with h5py.File(fn, "w") as handle:
            handle['mu_tau'] = mu_tau_vec
            handle['sigm_tau'] = sigm_tau_vec
            handle['mu_tauc'] = mu_tauc_vec
            handle['sigm_tauc'] = sigm_tauc_vec

# Broadcast samples to all ranks
x_all = pproc.comm.bcast(x_all, root=pproc.root)

# Set number of samples as instance attribute
pproc.total_samples = x_all.shape[0]

# Initialze stuff for parallel processing
pproc.prep_rank_samples()
if pproc.rank == pproc.root:
    pbar = tqdm(total=pproc.rank_samples, position=0, dynamic_ncols=True, file=sys.stderr)

# Generate metric values for training
for rank_sample_i, x_i in enumerate(range(pproc.rank_start, pproc.rank_start + pproc.rank_samples)):
    x = x_all[x_i] # Retrieve parameter sample
    pproc.global_sample_i = x_i # Referenced in change_tau_expect

    for i, param in enumerate(param_keys):
        # Assign parameter values for this sample
        if i in [1,2]:
            _type = int
        else:
            _type = float
        setattr(pproc, param, x[i].astype(_type))

    # Reset tau values to baseline
    pproc.tau_expect = pproc.tau_flat

    # Add in uncertainty on baseline tau values
    pproc.generate_eps_tau()
    pproc.tau_expect = pproc.tau_flat + pproc.eps_tau

    # Shift selected tau values (including uncertainty)
    pproc.change_tau_expect(pproc.C, pproc.ncell, pproc.slice_left)

    # Compute and store metric value
    if pproc.meta_metric == 'gte_thresh':
        pproc.calculate_metric_gte()
    pproc.metric_expect_rank[rank_sample_i] = pproc.metric_expect

    # Update progress (root only)
    if (pproc.rank == pproc.root) and (rank_sample_i % 500 == 0):
        pbar.update(500); print()

# Collect data across ranks
# Initialize data to store sample means across all ranks
sendcounts = np.array(pproc.comm.gather(len(pproc.metric_expect_rank), root=pproc.root))
if pproc.rank == pproc.root:
    sampled_metric_expect = np.empty(sum(sendcounts))
    sampled_xs_means = np.ones(sum(sendcounts)) * np.nan
else:
    sampled_metric_expect = None
    sampled_xs_means = None
# Now gather data
pproc.comm.Gatherv(pproc.metric_expect_rank, sampled_metric_expect, root=pproc.root)
pproc.comm.Gatherv(pproc.xs_means_rank, sampled_xs_means, root=pproc.root)

if pproc.rank == pproc.root:
    pbar.close()
    print(f"{timeit.default_timer() - start_time} seconds")
    meta_metric_all = sampled_metric_expect[:,None]
    np.save(pproc.data_dir + '/meta_metric_all.npy', meta_metric_all)
    #plt.hist(meta_metric_all);
    #plt.show()

    # Compute meta metric under no change for reference later
    pproc.tau_expect = pproc.tau_flat
    if pproc.meta_metric == 'gte_thresh':
        pproc.calculate_metric_gte()
    np.save(pproc.data_dir + '/meta_metric_nochange.npy', pproc.metric_expect)
