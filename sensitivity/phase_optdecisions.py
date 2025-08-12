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

# Define the number of eps samples per decision combination
constants['num_eps_combs'] = 5_000_000

# Define ordered list of parameter keys
param_keys = ['C', 'ncell', 'slice_left',
              'mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'demographic_index']
uncertain_params = ['mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'demographic_index']

# Initialize "Phase" instance for processing samples
pproc = Phase(**constants)
pproc.initialize()

# Define C
C_vec = np.load(pproc.data_dir + '/C_vec.npy')
C = 6
assert np.any(np.isclose(C_vec/pproc.ncell_tot, C))

# Get optimal decisions at a few values of q (% decrease in S_opt_baseline)
#q_vec = np.arange(0.0, 1, 0.2)
q_vec = np.array([0.0]) 
Sstar_i_optdecisions = np.empty(len(q_vec))
decision_opt_uncertain = np.load(pproc.data_dir + '/decision_opt_uncertain.npy')
rob_thresh_vec = np.load(pproc.data_dir + '/rob_thresh_vec.npy')
n_opt = np.empty(len(q_vec) + 1)
l_opt = np.empty(len(q_vec) + 1)

# First, get optima under baseline conditions
S_opt_baseline = np.load(pproc.data_dir + '/S_opt_baseline.npy')
decision_opt_baseline = np.load(pproc.data_dir + '/decision_opt_baseline.npy')
n_opt[0] = decision_opt_baseline[0]
l_opt[0] = decision_opt_baseline[1]

# Now get the optimal decisions for (1-q) * optimal S baseline
for q_i, q in enumerate(q_vec):
    Sstar_i = np.argmin(np.abs(rob_thresh_vec - ((1 - q) * S_opt_baseline)) )
    Sstar_i_optdecisions[q_i] = Sstar_i
    n_opt[q_i + 1] = decision_opt_uncertain[Sstar_i, 0]
    l_opt[q_i + 1] = decision_opt_uncertain[Sstar_i, 1]

    # Replace this q value with the closest one we have available
    '''I dont think this is really necessary'''
    Sstar = rob_thresh_vec[Sstar_i]
    q_vec[q_i] = 1 - (Sstar / S_opt_baseline)
np.save(pproc.data_dir + '/q_vec.npy', q_vec)
np.save(pproc.data_dir + '/Sstar_i_optdecisions.npy', Sstar_i_optdecisions.astype(int))

# Now set decision attributes of phase processor
constants['ncell_samples'] = n_opt
constants['slice_samples'] = l_opt 
constants['tauc_min_samples'] = np.repeat(C, len(q_vec) + 1)

# Re-initialize with these decision parameters
pproc = Phase(**constants)
pproc.initialize()

# Save decision parameters to file
pproc.init_strategy_variables(overwrite=True, suffix='_optdecisions')

# Read in all splined interpolations of metric(tau)
with open(pproc.data_dir + "/metric_spl_all.pkl", "rb") as handle:
    metric_spl_all = pickle.load(handle)

# Theoretical (or ad-hoc) maxima/minima for parameters
'''Reserve the demo sample index 0 for mean lambda(tau)'''
minima = {
    'mu_tau': -0.75,
    'sigm_tau': 0.,
    'mu_tauc': -0.75,
    'sigm_tauc': 0.,
    'demographic_index': 1
}
maxima = {
    'mu_tau': 0.15,
    'sigm_tau': 0.2,
    'mu_tauc': 0.15,
    'sigm_tauc': 0.2,
    'demographic_index': len(metric_spl_all) - 1 
}

# Start timer to track runtime
start_time = timeit.default_timer()

# Population full list of parameter combinations 'x_all'
if pproc.rank != pproc.root:
    x_decision = None
    decision_indices = None
    fixed_metric_mask = None
else:
    # Store indices of demographic samples that will have a fixed value of S=0
    fixed_metric_mask = np.full(len(metric_spl_all), False)
    tau_test = np.linspace(pproc.min_tau, pproc.final_max_tau, 1000)
    for demographic_index, metric_spl in metric_spl_all.items():
        # Check that spline lower bound makes sense
        assert metric_spl(pproc.min_tau) > 0
        # Check if lambda ever greater than threshold
        if np.all(metric_spl(tau_test) < metric_thresh) and (pproc.meta_metric == 'gte_thresh'):
            fixed_metric_mask[demographic_index] = True
    print(f'{np.count_nonzero(fixed_metric_mask)} of {len(metric_spl_all)} demograhpic samples are always unstable')

    ## Initialize decision combinations ### 
    num_decision_combs = pproc.C_vec.size * pproc.ncell_vec.size * pproc.slice_left_all.size

    if isinstance(pproc.ncell_samples, np.ndarray) and isinstance(pproc.slice_samples, np.ndarray):
        decision_combs = [_ for _ in zip(pproc.C_vec, pproc.ncell_vec, pproc.slice_left_all)]

    # Initialize decision parameter combinations
    x_decision = np.full((num_decision_combs, 3), np.nan)

    # Filter out any invalid param samples
    for decision_comb_i, decision_comb in enumerate(decision_combs):
        # Check that slice is within allow range
        if decision_comb[2] > (pproc.slice_right_max - decision_comb[1]): continue
        x_decision[decision_comb_i, :] = decision_comb
    nan_filt = np.any(np.isnan(x_decision), axis=1)
    x_decision = x_decision[~nan_filt, :]

    # Recalculate number of decision combinations
    num_decision_combs = x_decision.shape[0]

    # Shuffle to give all procs ~ the same amount of work
    pproc.rng.shuffle(x_decision)

    # Generate keys for each decision combination for use in the h5 file
    # for example, (C_i=0, n_i=1, l_i=0) -> '010'
    assert np.all(np.isin(x_decision[:,0], pproc.C_vec))
    assert np.all(np.isin(x_decision[:,1], pproc.ncell_vec))
    assert np.all(np.isin(x_decision[:,2], pproc.slice_left_all))
    decision_indices = np.zeros((num_decision_combs, 3)).astype(int)
    decision_indices[:,0] = np.array([np.argwhere(pproc.C_vec == v)[0][0] for v in x_decision[:,0]])
    decision_indices[:,1] = np.array([np.argwhere(pproc.ncell_vec == v)[0][0] for v in x_decision[:,1]])
    decision_indices[:,2] = np.array([np.argwhere(pproc.slice_left_all == v)[0][0] for v in x_decision[:,2]])
    decision_indices = np.array(['.'.join([str(x) for x in indices]) for indices in decision_indices])
    np.save(pproc.data_dir + '/decision_indices_optdecisions.npy', decision_indices)

# Broadcast samples to all ranks
x_decision = pproc.comm.bcast(x_decision, root=pproc.root)
decision_indices = pproc.comm.bcast(decision_indices, root=pproc.root)
fixed_metric_mask = pproc.comm.bcast(fixed_metric_mask, root=pproc.root)

# Initialize results file
results = h5py.File(pproc.data_dir + '/phase_optdecisions.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
for idx in decision_indices:
    results.create_dataset(idx, (pproc.num_eps_combs,), dtype='float64')
    results.create_dataset(idx + 'uncertainty_samples', (pproc.num_eps_combs, len(uncertain_params)), dtype='float64')

# Set number of samples as instance attribute
pproc.total_samples = x_decision.shape[0]

# Initialze stuff for parallel processing
pproc.prep_rank_samples()
if pproc.rank == pproc.root:
    pbar = tqdm(total=pproc.rank_samples, position=0, dynamic_ncols=True, file=sys.stderr)
    pbar_step = int(pproc.rank_samples/100) if pproc.rank_samples >= 100 else 1

# Generate meta metric values (i.e. results)
for rank_sample_i, decision_i in enumerate(range(pproc.rank_start, pproc.rank_start + pproc.rank_samples)):
    # Initialize array for results at this decision
    meta_metric_all = np.full(pproc.num_eps_combs, np.nan)

    # Initialize x_all with repeats of this particular decision
    x_all = np.tile(x_decision[decision_i], (pproc.num_eps_combs, 1))

    # Add columns for uncertain param samples
    x_all = np.hstack((
                          x_all,
                          np.full((x_all.shape[0], len(uncertain_params)), np.nan)
                      ))

    # Loop over each uncertain param to populate samples 
    for i, uncertain_param in enumerate(uncertain_params):
        uncertain_param_i = i + 3
         
        # Get min, max and type for this param
        _min = minima[uncertain_param]
        _max = maxima[uncertain_param]
        assert (type(_min)==type(_min))
        _type = type(_min)

        # Generate and store nonzero samples
        if uncertain_param in ['mu_tau', 'mu_tauc']:
            # Pct change from baseline will be drawn from beta distributions 
            alpha = 1.5 # Start by defining an ad-hoc value for alpha

            # Solve for the transformed mode of zero
            x_mode = (-_min) / (_max - _min)

            # Use transformed mode to solve for 2nd shape parameter _beta
            _beta = ((alpha - 1) / x_mode) - alpha + 2

            # Draw samples from the beta distribution and apply linear transform
            samples = pproc.rng.beta(alpha, _beta, pproc.num_eps_combs)
            samples = _min + ((_max - _min) * samples)
        elif uncertain_param in ['sigm_tau', 'sigm_tauc']:
            # Multipliers to get spread of tau/tauc post pct change, assume uniform
            samples = pproc.rng.uniform(_min, _max, pproc.num_eps_combs)
        elif uncertain_param == 'demographic_index':
            # Samples of pop parameters follow inferred posterior, so just select
            # indices of pre-generated samples uniformly
            samples = pproc.rng.integers(_min, _max, pproc.num_eps_combs, endpoint=True)
        else:
            sys.exit(f'No protocol specified for how to draw samples of {uncertain_param}') 
        x_all[:, uncertain_param_i] = samples

    # Compute outcome under all uncertainty samples
    for x_i, x in enumerate(x_all):
        # First, check if lambda(tau) < lambda^* for all tau; for these S=0
        if fixed_metric_mask[int(x[-1])]:
            meta_metric_all[x_i] = 0.0
        
        # Otherwise, actually compute the new value of S
        else:
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
            pproc.tau_expect = pproc.generate_tau() 

            # Shift selected tau values (including uncertainty)
            pproc.change_tau_expect(pproc.C, pproc.ncell, pproc.slice_left)

            # Compute and store metric value
            if pproc.meta_metric == 'gte_thresh':
                pproc.calculate_metric_gte(metric_thresh)
            meta_metric_all[x_i] = pproc.metric_expect

    # Store meta metric values and uncertainty samples in h5
    key = decision_indices[decision_i]
    results[key][:] = meta_metric_all
    results[key + 'uncertainty_samples'][:] = x_all[:, 3:]

    # Update progress (root only)
    if (pproc.rank == pproc.root) and (rank_sample_i % pbar_step == 0):
        pbar.update(pbar_step); print()

if pproc.rank == pproc.root:
    pbar.close()
    print(f"{timeit.default_timer() - start_time} seconds")

results.close()
