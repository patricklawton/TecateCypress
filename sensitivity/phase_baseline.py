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
constants.update({'overwrite_metrics': True}) 
constants['overwrite_results'] = True
metric_thresh = 0.975 # Threshold of pop metric value used for calculating meta metric

# Get list of samples for each parameter
constants['tauc_min_samples'] = np.arange(2, 18, 4)
constants['ncell_samples'] =  150#250
constants['slice_samples'] = 300#500

# Define ordered list of parameter keys
param_keys = ['C', 'ncell', 'slice_left',
              'mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'demographic_index']
uncertain_params = ['mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'demographic_index']

# Initialize "Phase" instance for processing samples
pproc = Phase(**constants)
pproc.initialize()
pproc.init_strategy_variables(overwrite=True, suffix='_baseline')

# Save C_vec to file for reference during optimization
np.save(pproc.data_dir + '/C_vec_baseline.npy', pproc.C_vec)

# Read in all splined interpolations of metric(tau)
with open(pproc.data_dir + "/metric_spl_all.pkl", "rb") as handle:
    metric_spl_all = pickle.load(handle)

# Theoretical (or ad-hoc) maxima/minima for parameters
'''Reserve the demo sample index 0 for mean lambda(tau)'''
minima = {
    'mu_tau': 0.,
    'sigm_tau': 0.,
    'mu_tauc': 0,
    'sigm_tauc': 0.,
    'demographic_index': 0
}
maxima = {
    'mu_tau': 0.,
    'sigm_tau': 0,
    'mu_tauc': 0.,
    'sigm_tauc': 0,
    'demographic_index': 0
}

# Start timer to track runtime
start_time = timeit.default_timer()

# Population full list of parameter combinations 'x_all'
if pproc.rank != pproc.root:
    x_decision = None
else:
    for demographic_index, metric_spl in metric_spl_all.items():
        # Check that spline lower bound makes sense
        assert metric_spl(pproc.min_tau) > 0

    ## Initialize decision combinations ### 
    num_decision_combs = pproc.C_vec.size * pproc.ncell_vec.size * pproc.slice_left_all.size

    decision_combs = product(pproc.C_vec,
                    pproc.ncell_vec,
                    pproc.slice_left_all)

    # Initialize decision parameter combinations
    x_decision = np.full((num_decision_combs, 3), np.nan)

    # Filter out any invalid param samples
    for decision_comb_i, decision_comb in enumerate(decision_combs):
        # Check that slice is within allow range
        if decision_comb[2] > (pproc.slice_right_max - decision_comb[1]): continue
        x_decision[decision_comb_i, :] = decision_comb
    nan_filt = np.any(np.isnan(x_decision), axis=1)
    x_decision = x_decision[~nan_filt, :]

    # Shuffle to give all procs ~ the same amount of work
    pproc.rng.shuffle(x_decision)

# Broadcast samples to all ranks
x_decision = pproc.comm.bcast(x_decision, root=pproc.root)

# Add columns for uncertain param samples
x_all = np.hstack((
                      x_decision,
                      np.zeros((x_decision.shape[0], len(uncertain_params)))
                  ))

# Set number of samples as instance attribute
pproc.total_samples = x_decision.shape[0]

# Initialze stuff for parallel processing
pproc.prep_rank_samples()
if pproc.rank == pproc.root:
    pbar = tqdm(total=pproc.rank_samples, position=0, dynamic_ncols=True, file=sys.stderr)
    pbar_step = int(pproc.rank_samples/100) if pproc.rank_samples >= 100 else 1

# Generate meta metric values (i.e. results)
for rank_sample_i, x_i in enumerate(range(pproc.rank_start, pproc.rank_start + pproc.rank_samples)):
    x = x_all[x_i]

    # Assign parameter values for this sample
    for i, param in enumerate(param_keys):
        if i in [1,2]:
            param_val = int(x[i])
        elif param == 'demographic_index':
            # Retrieve the spline function for this demographic sample
            demographic_index = int(x[i])
            param_val = metric_spl_all[demographic_index]
            param = 'metric_spl'
        else:
            param_val = float(x[i])
        setattr(pproc, param, param_val)

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
    if (pproc.rank == pproc.root) and (rank_sample_i % pbar_step == 0):
        pbar.update(pbar_step); print()

# Initialize data to store samples across all ranks
sendcounts = np.array(pproc.comm.gather(len(pproc.metric_expect_rank), root=pproc.root))
if pproc.rank == pproc.root:
    sampled_metric_expect = np.empty(sum(sendcounts))
else:
    sampled_metric_expect = None
# Now gather data
pproc.comm.Gatherv(pproc.metric_expect_rank, sampled_metric_expect, root=pproc.root)

if pproc.rank == pproc.root:
    pbar.close()

    # Compute meta metric under no change for reference later
    pproc.tau_expect = pproc.tau_flat
    pproc.metric_spl = metric_spl_all[0]
    if pproc.meta_metric == 'gte_thresh':
        pproc.calculate_metric_gte(metric_thresh)
    meta_metric_nochange = pproc.metric_expect
    np.save(pproc.data_dir + '/meta_metric_nochange.npy', meta_metric_nochange)

    print(f"{timeit.default_timer() - start_time} seconds")
    np.save(pproc.data_dir + '/x_all_baseline.npy', x_all)
    np.save(pproc.data_dir + '/meta_metric_all_baseline.npy', sampled_metric_expect[:,None])
