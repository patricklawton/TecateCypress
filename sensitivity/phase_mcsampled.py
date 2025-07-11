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
import pickle

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
metric_thresh = 0.975 # Threshold of pop metric value used for calculating meta metric

# Get list of samples for each parameter
constants['tauc_min_samples'] = np.array([9.0])
#constants['tauc_min_samples'] = np.arange(1, 17, 4)
#constants['ncell_samples'] = 20
#constants['slice_samples'] = 40
#constants['ncell_samples'] = 50
#constants['slice_samples'] = 75
constants['ncell_samples'] = 35
constants['slice_samples'] = 65

# Start timer to track runtime
start_time = timeit.default_timer()

# Define ordered list of parameter keys
param_keys = ['C', 'ncell', 'slice_left',
              'mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'demographic_index']

# Initialize "Phase" instance for processing samples
pproc = Phase(**constants)
pproc.initialize()
pproc.init_strategy_variables()

# Read in all splined interpolations of metric(tau)
with open(pproc.data_dir + "/metric_spl_all.pkl", "rb") as handle:
    metric_spl_all = pickle.load(handle)

# Population full list of parameter combinations 'x_all'
if pproc.rank != pproc.root:
    x_all = None
    fixed_metric_mask = None
else:
    # Store indices of demographic samples that will have a fixed value of the metapop metric
    # This occurs for the gte_thresh metric when metric(tau) < thresh for all tau
    fixed_metric_mask = np.full(len(metric_spl_all), False)
    tau_test = np.linspace(pproc.min_tau, pproc.final_max_tau, 1000)
    for demographic_index, metric_spl in metric_spl_all.items():
        # Check that spline lower bound makes sense
        assert metric_spl(pproc.min_tau) > 0
        if np.all(tau_test < metric_thresh) and (pproc.metric == 'gte_thresh'):
            fixed_metric_mask[demographic_index] = True

    # Theoretical (or ad-hoc) maxima/minima for parameters
    minima = {
        # 'C': 5.*pproc.ncell_tot,
        # 'ncell': int(0.02*pproc.ncell_tot),
        # 'slice_left': int(0.* pproc.ncell_tot),
        'mu_tau': -10.,
        'sigm_tau': 0.,
        'mu_tauc': -1.0,
        'sigm_tauc': 0.,
        'demographic_index': 1
    }
    maxima = {
        # 'C': 5.*pproc.ncell_tot,
        # 'ncell': int(1. * pproc.slice_right_max),
        # 'slice_left': int(1.*pproc.ncell_tot),
        'mu_tau': 0.,
        'sigm_tau': 10.,
        'mu_tauc': 0.0,
        'sigm_tauc': 0.,
        'demographic_index': len(metric_spl_all) - 1 
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
    #num_eps_combs = 225
    #num_eps_combs = 500
    #num_eps_combs = 1000
    num_eps_combs = 5000
    np.save(pproc.data_dir + '/num_eps_combs.npy', num_eps_combs)
    num_combs_tot = num_strategy_combs * num_eps_combs

    # Initialize x_all with repeats of strategy combinations
    x_all = np.tile(x_strategy, (num_eps_combs, 1))

    # Fill columns with samples of uncertain parameters
    uncertain_params = ['mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'demographic_index']

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
            samples = rng.integers(_min, _max, num_combs_tot-num_strategy_combs, endpoint=True)
        x_all[num_strategy_combs:, uncertain_param_i] = samples

    # Shuffle to give all procs ~ the same amount of work
    pproc.rng.shuffle(x_all)

    # Save to file for reference in postprocessing
    np.save(pproc.data_dir + '/x_all.npy', x_all)

# Broadcast samples to all ranks
x_all = pproc.comm.bcast(x_all, root=pproc.root)
fixed_metric_mask = pproc.comm.bcast(fixed_metric_mask, root=pproc.root)

# Set number of samples as instance attribute
pproc.total_samples = x_all.shape[0]

# Initialze stuff for parallel processing
pproc.prep_rank_samples()
if pproc.rank == pproc.root:
    pbar = tqdm(total=pproc.rank_samples, position=0, dynamic_ncols=True, file=sys.stderr)
    pbar_step = int(pproc.rank_samples/100)

# Generate metric values for training
for rank_sample_i, x_i in enumerate(range(pproc.rank_start, pproc.rank_start + pproc.rank_samples)):
    x = x_all[x_i] # Retrieve parameter sample
    pproc.global_sample_i = x_i # Referenced in change_tau_expect

    for i, param in enumerate(param_keys):
        # Assign parameter values for this sample
        if i in [1,2]:
            #_type = int
            param_val = int(x[i])
        elif param == 'demographic_index':
            # Retrieve the spline function for this demographic sample
            demographic_index = int(x[i])
            param_val = metric_spl_all[demographic_index]
            param = 'metric_spl'
        else:
            #_type = float
            param_val = float(x[i])
        setattr(pproc, param, param_val)

    ## Skip parameter comb if meta metric doesn't change; fill these in later
    #if fixed_metric_mask[demographic_index]: continue

    # Reset tau values to baseline
    pproc.tau_expect = pproc.tau_flat

    # Add in uncertainty on baseline tau values
    '''Note that at this stage tau values may become negative; 
       they are resticted to positive in change_tau_expect'''
    pproc.generate_eps_tau()
    pproc.tau_expect = pproc.tau_flat + pproc.eps_tau

    # Shift selected tau values (including uncertainty)
    pproc.change_tau_expect(pproc.C, pproc.ncell, pproc.slice_left)

    # Compute and store metric value
    if pproc.meta_metric == 'gte_thresh':
        pproc.calculate_metric_gte(metric_thresh)
    pproc.metric_expect_rank[rank_sample_i] = pproc.metric_expect

    # Update progress (root only)
    #if (pproc.rank == pproc.root) and (rank_sample_i % 500 == 0):
    if (pproc.rank == pproc.root) and (rank_sample_i % pbar_step == 0):
        pbar.update(pbar_step); print()

# Collect data across ranks
# Initialize data to store sample means across all ranks
sendcounts = np.array(pproc.comm.gather(len(pproc.metric_expect_rank), root=pproc.root))
if pproc.rank == pproc.root:
    sampled_metric_expect = np.empty(sum(sendcounts))
    #sampled_xs_means = np.ones(sum(sendcounts)) * np.nan
else:
    sampled_metric_expect = None
    #sampled_xs_means = None
# Now gather data
pproc.comm.Gatherv(pproc.metric_expect_rank, sampled_metric_expect, root=pproc.root)
#pproc.comm.Gatherv(pproc.xs_means_rank, sampled_xs_means, root=pproc.root)

if pproc.rank == pproc.root:
    pbar.close()

    # Compute meta metric under no change for reference later
    pproc.tau_expect = pproc.tau_flat
    pproc.metric_spl = metric_spl_all[0]
    if pproc.meta_metric == 'gte_thresh':
        pproc.calculate_metric_gte(metric_thresh)
    meta_metric_nochange = pproc.metric_expect
    np.save(pproc.data_dir + '/meta_metric_nochange.npy', meta_metric_nochange)

    ## Put in baseline meta metric values where relevant
    #demo_col_i = np.nonzero(param_keys == 'demographic_index')[0][0]
    #mask = ...
    #sampled_metric_expect[fixed_metric_mask] = meta_metric_nochange

    print(f"{timeit.default_timer() - start_time} seconds")
    meta_metric_all = sampled_metric_expect[:,None]
    np.save(pproc.data_dir + '/meta_metric_all.npy', meta_metric_all)
