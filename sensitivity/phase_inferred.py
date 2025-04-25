import numpy as np
from project import Phase
import torch
import sys
from matplotlib import pyplot as plt

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

pproc = Phase(**constants)
pproc.initialize()
assert pproc.metric_exp_spl(pproc.min_tau) > 0

#### Get samples of each parameter we want to train on ####
NUM_TRAIN = 1000

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
    'mu_tau': 6.,
    'sigm_tau': 10.,
    'mu_tauc': 6.,
    'sigm_tauc': 10.
}
param_keys = np.array(list(minima.keys()))

# Generate parameter values for training
train_x = np.full((NUM_TRAIN, len(minima)), np.nan)
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


# Generate metric values for training
train_y = np.full((len(train_x), 1), np.nan)
for x_i, x in enumerate(train_x):
    for i, param in enumerate(param_keys):
        # Assign parameter values for this sample
        setattr(pproc, param, x[i].astype(type(minima[param])))

    if pproc.slice_left > (pproc.slice_right_max - pproc.ncell):
        continue # Skip bc slice start too high

    # Reset tau values to baseline
    pproc.tau_expect = pproc.tau_flat

    # Add in uncertainty on baseline tau values
    pproc.generate_eps_tau(pproc.mu_tau, pproc.sigm_tau)
    pproc.tau_expect = pproc.tau_flat + pproc.eps_tau

    # Shift selected tau values (including uncertainty)
    pproc.change_tau_expect(pproc.C, pproc.ncell, pproc.slice_left)

    # Compute and store metric value
    pproc.calculate_metric_expect()
    train_y[x_i] = pproc.metric_expect

plt.hist(train_y);
plt.show()
