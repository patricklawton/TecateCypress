import numpy as np
from model import Model
import json
from matplotlib import pyplot as plt

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle)) 
#params['sigm_max'] = 0.0
#params['rho_max'] = 100
#params['alph_m'] = 0.2*params['alph_m']
#params['K_adult'] = 50*params['K_adult']
#params['eta'] = 0.2

A = 1 #ha
num_reps = 500
N_0_1 = np.repeat(0.9*A*params['K_adult'], num_reps)
#N_0_1 = np.repeat(13*A*params['K_adult'], num_reps)
#init_age = 20
init_age = round(params['a_mature']) + 10
t_vec = np.arange(1, 152)

model = Model(**params)
model.set_area(A)
model.init_N(N_0_1, init_age)
model.simulate(t_vec=t_vec, census_every=2, fire_probs=0.02)
np.save('N_tot_vec.npy', model.N_tot_vec)

fig, axs = plt.subplots(4, 1, figsize=(7,20))
for N_tot_vec in model.N_tot_vec[::int(num_reps/10)]:
    axs[0].plot(model.census_yrs,N_tot_vec)
    axs[1].plot(model.census_yrs,N_tot_vec)
axs[0].axhline(params['K_adult'], ls='--', c='k', alpha=0.35)
axs[1].set_ylim(-0.03*params['K_adult'],params['K_adult'])
#print(len(model.N_tot_vec.mean(axis=0)))
axs[2].plot(model.census_yrs, model.N_tot_vec.mean(axis=0), c='k')
axs[2].axhline(params['K_adult'], ls='--', c='k', alpha=0.35)
axs[3].plot(model.census_yrs, model.N_tot_vec.mean(axis=0), c='k')
axs[3].axhline(params['K_adult'], ls='--', c='k', alpha=0.35)
axs[3].set_ylim(-0.03*params['K_adult'],1.5*params['K_adult'])
fig.savefig('sim_test.jpeg', bbox_inches='tight')
