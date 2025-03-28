from model import Model
import json
import numpy as np
from matplotlib import pyplot as plt
import timeit
from scipy.special import gamma
from scipy.stats import weibull_min
from tqdm import tqdm
import signac as sg
import itertools
from global_functions import lambda_s, s
from mpi4py import MPI
import pickle
import os

# Initialize MPI stuff
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_procs = comm.Get_size()
root = 0
if rank==root: print(f'num_procs={num_procs}')

# Function to return parameter combinations as dicts
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

# For sampling from various probability distributions
rng = np.random.default_rng()

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle))

# Constants
c = 1.42
delta_t = 1
num_reps = 1_000
metrics = ['lambda_s', 's']
data_dir = "checkinits_data"

# Nominal values of other parameters
nominals = {
    'Aeff': 7.29, 
    'b': 40, 
    't_final': 300,
    'q': 0.9 
}
# Theoretical (or ad-hoc) maxima/minima for parameters
minima = {
    'Aeff': 1, 
    'b': 2, 
    't_final': 100,
    'q': np.nan 
}
maxima = {
    'Aeff': np.nan, 
    'b': 60, 
    't_final': 800,
    'q': 0.99 
}
# Samples for each parameter range within bounds
num_samples = {
    'Aeff': 8,
    'b': 15,
    't_final': 15,
    'q': 8
}

# Ranges of parameters we're checking sensitivity to
ranges = {}
for i, (key, nominal_val) in enumerate(nominals.items()):
    if np.isnan(minima[key]):
        lower = 0.1 * nominal_val
    else:
        lower = minima[key]

    if np.isnan(maxima[key]):
        upper = 10*nominal_val
    else:
        upper = maxima[key]
    
    ranges[key] = np.linspace(lower, upper, num_samples[key]).astype(type(nominal_val))
# Add some samples to b's range
ranges['b'] = np.append(ranges['b'], np.arange(120, 360, 60))
# Save ranges to file
if rank == root:
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    with open(data_dir + "/ranges.pkl", 'wb') as handle:
        pickle.dump(ranges, handle)

if rank == root:
    # Use ordering of ranges to initialize results matrix metric_evals
    sizes = []
    for key in ranges.keys():
        sizes.append(ranges[key].size)
    metric_evals = np.full(sizes, np.nan)

    # Get all initial condition combinations
    ranges_tmp = ranges.copy() # Only use the max t_final, reuse for metric calc below
    ranges_tmp.update({'t_final': np.array([max(ranges['t_final'])])})
    combinations = list(product_dict(**ranges_tmp))
    np.random.shuffle(combinations)
else:
    metric_evals = None
    combinations = None
combinations = comm.bcast(combinations, root=root)

# Separate data for this rank
rank_combinations = np.array_split(combinations, num_procs)[rank] 
if rank == root:
    pbar = tqdm(total=len(rank_combinations), position=0)
rank_evals_size = rank_combinations.size*ranges['t_final'].size
rank_metric_evals = np.full((len(metrics), rank_evals_size, len(ranges.keys()) + 1), np.nan)

# Loop over combinations
comb_i = 0
for comb in rank_combinations:
    N_0_1 = comb['Aeff']*params['K_adult']
    N_0_1_vec = np.repeat(N_0_1, num_reps)
    t_vec = np.arange(delta_t, comb['t_final']+delta_t, delta_t)
    init_age = params['a_mature'] - (np.log((1/comb['q'])-1) / params['eta_rho']) # Age where (q*100)% of reproductive capacity reached
    init_age = int(init_age + 0.5) # Round to nearest integer

    # Initialize model instance with the specified parameters
    model = Model(**params)
    model.set_effective_area(comb['Aeff'])
    model.init_N(N_0_1_vec, init_age)
    model.set_t_vec(t_vec)
    model.set_weibull_fire(b=comb['b'], c=c)
    model.generate_fires()

    # Run the simulation
    model.simulate(method="discrete", census_every=1, progress=False)

    # Loop over t_final values, reusing the longest simulation
    for tf in ranges['t_final']:
        comb.update({'t_final': tf}) 
 
        # Get the relevant indices in results matrix for this combination
        indices = []
        for key, val in comb.items():
            index = np.argwhere(ranges[key] == val)[0][0]
            indices.append(index)

        for metric_i, metric in enumerate(metrics):
            # Get the metric value
            if metric == 'lambda_s':
                metric_all = lambda_s(model.N_tot_vec[:, :tf-1], compressed=False)
            elif metric == 's':
                metric_all = s(model.N_tot_vec[:, :tf-1], compressed=False)
            metric_val = np.mean(metric_all)

            # Store results
            rank_metric_evals[metric_i, comb_i, :] = indices + [metric_val]

        comb_i += 1

    # Update progress (root only)
    if rank == root:
        pbar.update(1); print()

# Gather results across all ranks
gathered_metric_evals = comm.gather(rank_metric_evals, root=root)
if rank == root:
    # Store data in final results array
    for metric_i, metric in enumerate(metrics):
        for rank_metric_evals in gathered_metric_evals:
            for data in rank_metric_evals[metric_i, ...]:
                indices = tuple(data[:-1].astype(int))
                metric_evals[indices] = data[-1]
        # Save to file
        np.save(data_dir + f'/metric_evals_{metric}', metric_evals)
