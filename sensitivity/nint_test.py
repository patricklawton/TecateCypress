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
import signac as sg

# Read in map parameters
#params = {}
#for pr in ['mortality', 'fecundity']:
#    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
#        params.update(json.load(handle))
# Read in parameters from a specific sample
project = sg.get_project()
job = project.get_job('workspace/a066a04d26bc802ca30a20ad67c57cf4')
params = job.sp.params

# Constants
overwrite_discrete = False
overwrite_fire = True
A_cell = 270**2 / 10_000 #Ha
sdm_mean = 0.51
#Aeff = 7.29 
Aeff = 1.0
#Aeff = np.round(4*A_cell*sdm_mean)
#fri = 40
c = 1.42
b = 84#fri / gamma(1+1/c)
t_final = 5000
## Get the average habitat suitability within the Otay Mtn Wilderness area
sdmfn = "../shared_maps/SDM_1995.asc"
sdm = np.loadtxt(sdmfn,skiprows=6)
otay = np.loadtxt("../shared_maps/otayraster.asc", skiprows=6)
sdm_otay = sdm[otay==1] #index "1" indicates the specific part where study was done
h_o = np.mean(sdm_otay[sdm_otay!=0]) #excluding zero, would be better to use SDM w/o threshold
A_o = 0.1 #area of observed sites in Ha
delta_t = 1
num_reps = 3_000
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

# Run numerical integration simulation
start_time = timeit.default_timer()
model.simulate(method="nint", census_every=1, progress=True)
N_tot_mean_nint = model.N_tot_vec.mean(axis=0)
nint_fn = 'nint_data/N_tot_mean_nint.npy'
np.save(nint_fn, N_tot_mean_nint)
np.save('nint_data/census_t.npy', model.census_t)
np.save('nint_data/N_tot.npy', model.N_tot_vec)
elapsed = timeit.default_timer() - start_time
print('{} seconds'.format(elapsed))
