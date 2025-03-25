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
# Read in parameters from a specific sample
#project = sg.get_project()
#job = project.get_job('workspace/a1c4d78b08a78b045e0f257a8b496f75')
#params = job.sp.params

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
    'b': 120, 
    't_final': np.nan,
    'q': 0.99 
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

    ranges[key] = np.linspace(lower, upper, 11).astype(type(nominal_val))
print(ranges)

# Use ordering of ranges to initialize results matrix metric_evals
sizes = []
for key in ranges.keys():
    sizes.append(ranges[key].size)
metric_evals = np.full(sizes, np.nan)

combinations = list(product_dict(**ranges))
for comb in tqdm(combinations):
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
    #model.simulate(method="discrete", census_every=1, progress=False)

    ## Get the metric value
    #lam_s_all = lambda_s(model.N_tot)
    #lam = np.mean(lam_s_all)
 
    # Get the relevant indices in results matrix for this combination
    # Store results
    print(comb)
