from model import Model
from scipy.integrate import quad, solve_ivp
import os
import json
import numpy as np
# For sampling from various probability distributions
rng = np.random.default_rng()
from matplotlib import pyplot as plt
import timeit
from scipy.special import gamma
from scipy.stats import weibull_min
from tqdm import tqdm

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle))

# Constants
overwrite_discrete = True
overwrite_fire = True
Aeff = 7.29 #2.38
fri = 40
c = 1.42
b = fri / gamma(1+1/c)
t_final = 400
# Get the average habitat suitability within the Otay Mtn Wilderness area
sdmfn = "SDM_1995.asc"
sdm = np.loadtxt(sdmfn,skiprows=6)
otay = np.loadtxt("otayraster.asc", skiprows=6)
sdm_otay = sdm[otay==1] #index "1" indicates the specific part where study was done
h_o = np.mean(sdm_otay[sdm_otay!=0]) #excluding zero, would be better to use SDM w/o threshold
A_o = 0.1 #area of observed sites in Ha
delta_t = 1
num_reps = 1_000
N_0_1 = Aeff*params['K_adult']
N_0_1_vec = np.repeat(N_0_1, num_reps)
init_age = round(params['a_mature']) + 20
t_vec = np.arange(delta_t, t_final+delta_t, delta_t)

# Set up model instance shared by discrete and nint sims
discrete_fn = 'nint_data/N_tot_mean_discrete.npy'
t_fire_vec_fn = 'nint_data/t_fire_vec.npy'
model = Model(**params)
model.set_effective_area(Aeff)
model.init_N(N_0_1_vec, init_age)
model.set_t_vec(t_vec)
model.set_weibull_fire(b=b, c=c)
# Generate vector of fire occurances
if (os.path.isfile(t_fire_vec_fn)==False) or (overwrite_fire):
    model.generate_fires()
    t_fire_vec = model.t_fire_vec
    np.save(t_fire_vec_fn, t_fire_vec)
else:
    t_fire_vec = np.load(t_fire_vec_fn)
    model.t_fire_vec = t_fire_vec
# Run discrete simulation
start_time = timeit.default_timer()
if (os.path.isfile(discrete_fn)==False) or (overwrite_discrete):
    model.simulate(method="discrete", census_every=1, progress=True)
    N_tot_mean_disc = model.N_tot_vec.mean(axis=0)
    np.save(discrete_fn, N_tot_mean_disc)
else:
    N_tot_mean_disc = np.load(discrete_fn)
#t_fire_vec = np.array([[1,0,0,1,0,0,0,1,0,0,0]])
#t_final = len(t_fire_vec[0])
#t_vec = np.arange(delta_t, t_final+delta_t, delta_t)
elapsed = timeit.default_timer() - start_time
print('{} seconds'.format(elapsed))

start_time = timeit.default_timer()
model.simulate(method="nint", census_every=1, progress=True)
N_tot_mean_nint= model.N_tot_vec.mean(axis=0)
nint_fn = 'nint_data/N_tot_mean_nint.npy'
np.save(nint_fn, N_tot_mean_nint)
elapsed = timeit.default_timer() - start_time
print('{} seconds'.format(elapsed))
