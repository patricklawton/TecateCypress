import numpy as np
from project import Phase
import torch
import sys
from matplotlib import pyplot as plt
from mpi4py import MPI
from tqdm import tqdm
import timeit

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
constants['overwrite_results'] = False
#constants['num_train'] = 2_000_000
constants['num_train'] = 1000

pproc = Phase(**constants)
pproc.initialize()
assert pproc.metric_exp_spl(pproc.min_tau) > 0

#### Get samples of each parameter we want to train on ####

# Function to return parameter combinations as dicts
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

# For sampling from various probability distributions
rng = np.random.default_rng()

# Theoretical (or ad-hoc) maxima/minima for parameters
minima = {
    'C': 5.*pproc.ncell_tot,
    'ncell': int(0.02*pproc.ncell_tot),
    # 'ncell': int(pproc.ncell_tot - pproc.slice_right_max),
    'slice_left': int(0.* pproc.ncell_tot),
    'mu_tau': -10.,
    'sigm_tau': 0.,
    'mu_tauc': -10.,
    'sigm_tauc': 0.
}
maxima = {
    'C': 5.*pproc.ncell_tot,
    'ncell': int(1. * pproc.slice_right_max),
    'slice_left': int(1.*pproc.ncell_tot),
    'mu_tau': 0.,
    'sigm_tau': 6.,
    'mu_tauc': 0.,
    'sigm_tauc': 6.
}
param_keys = np.array(list(minima.keys()))

if pproc.rank != pproc.root:
    train_x = None
else:
    # Start timer to track runtime
    start_time = timeit.default_timer()

    # Generate parameter values for training
    train_x = np.full((pproc.num_train, len(minima)), np.nan)
    for param_i, ((key1, _min), (key2, _max)) in enumerate(zip(minima.items(), maxima.items())):
        assert(key1 == key2); key = key1 = key2

        try:
            assert (type(_min)==type(_min))
            _type = type(_min)
        except:
            print('hey thats bad', type(_min), type(_min)); sys.exit()

        # Skip slice_left for now, sample after ncell has been sampled
        if key == 'slice_left': continue

        if _type == float:
            param_samples = rng.uniform(_min, _max, len(train_x))
        elif _type == int:
            param_samples = rng.integers(_min, _max, len(train_x))
        train_x[:, param_i] = param_samples
    # Sample slice_left only where feasible given ncell samples
    ncell_column = np.nonzero(param_keys == 'ncell')[0][0]
    slice_left_max_vec = pproc.slice_right_max - train_x[:,ncell_column]
    slice_left_column = np.nonzero(param_keys == 'slice_left')[0][0]
    train_x[:, slice_left_column] = rng.uniform(0, slice_left_max_vec)
train_x = pproc.comm.bcast(train_x, root=pproc.root)

# Initialze stuff for parallel processing
pproc.prep_rank_samples()
if pproc.rank == pproc.root:
    print('here', pproc.rank_samples)
    pbar = tqdm(total=pproc.rank_samples, position=0, dynamic_ncols=True, file=sys.stderr)

# Generate metric values for training
for rank_sample_i, x_i in enumerate(range(pproc.rank_start, pproc.rank_start + pproc.rank_samples)):
    x = train_x[x_i] # Retrieve parameter sample
    pproc.global_sample_i = x_i # Referenced in change_tau_expect

    for i, param in enumerate(param_keys):
        # Assign parameter values for this sample
        setattr(pproc, param, x[i].astype(type(minima[param])))

    # Reset tau values to baseline
    pproc.tau_expect = pproc.tau_flat

    # Add in uncertainty on baseline tau values
    pproc.generate_eps_tau()
    pproc.tau_expect = pproc.tau_flat + pproc.eps_tau

    # Shift selected tau values (including uncertainty)
    pproc.change_tau_expect(pproc.C, pproc.ncell, pproc.slice_left)

    # Compute and store metric value
    pproc.calculate_metric_gte()
    pproc.metric_expect_rank[rank_sample_i] = pproc.metric_expect

    ## Update progress (root only)
    #if pproc.rank == pproc.root:
    #    pbar.update(1); print()
    #    pbar.refresh()
    #    sys.stderr.flush()

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
    print("NOT SAVING DATA")
    #train_y = sampled_metric_expect[:,None]
    #np.save('train_x.npy', train_x)
    #np.save('train_y.npy', train_y)
    #plt.hist(train_y);
    #plt.show()
