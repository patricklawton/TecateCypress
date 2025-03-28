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
from global_functions import lambda_s
from mpi4py import MPI
import pickle

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
    't_final': 600,
    'q': 0.99 
}
# Samples for each parameter range within bounds
num_samples = {
    'Aeff': 8,
    'b': 20,
    't_final': 11,
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
    with open('ranges.pkl', 'wb') as handle:
        pickle.dump(ranges, handle)

if rank == root:
    # Use ordering of ranges to initialize results matrix metric_evals
    sizes = []
    for key in ranges.keys():
        sizes.append(ranges[key].size)
    metric_evals = np.full(sizes, np.nan)

    # Get all initial condition combinations
    combinations = list(product_dict(**ranges))
    np.random.shuffle(combinations)
else:
    metric_evals = None
    combinations = None
combinations = comm.bcast(combinations, root=root)

# Separate data for this rank
rank_combinations = np.array_split(combinations, num_procs)[rank] 
if rank == root:
    pbar = tqdm(total=len(rank_combinations), position=0)
rank_metric_evals = np.full((rank_combinations.size, len(ranges.keys()) + 1), np.nan)
rank_progress = 0

# Loop over combinations
for comb_i, comb in enumerate(rank_combinations):
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

    # Get the metric value
    lam_s_all = lambda_s(model.N_tot_vec, compressed=False)
    metric_val = np.mean(lam_s_all)
 
    # Get the relevant indices in results matrix for this combination
    indices = []
    for key, val in comb.items():
        index = np.argwhere(ranges[key] == val)[0][0]
        indices.append(index)

    # Store results
    rank_metric_evals[comb_i, :] = indices + [metric_val]

    rank_progress += 1
    #print(f'rank {rank} run {rank_progress} times, just ran t={comb["t_final"]}')

    # Update progress (root only)
    if rank == root:
        pbar.update(1)
        print()

# Gather results across all ranks
gathered_metric_evals = comm.gather(rank_metric_evals, root=root)
if rank == root:
    # Store data in final results array
    for rank_metric_evals in gathered_metric_evals:
        for data in rank_metric_evals:
            indices = tuple(data[:-1].astype(int))
            metric_evals[indices] = data[-1]
    # Save to file
    np.save('metric_evals', metric_evals)
