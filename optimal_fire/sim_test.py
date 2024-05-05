import numpy as np
from model import Model
import json
from matplotlib import pyplot as plt
import timeit
from scipy.special import gamma

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle)) 

A = 1 #ha
num_reps = 1000
delta_t = 1
N_0_1 = np.repeat(0.9*A*params['K_adult'], num_reps)
fire_prob = 1/40
fri = 110
c = 1.42
b = fri / gamma(1+1/c)
#t_max = fri*3
t_max = 152
#init_age = 50
init_age = round(params['a_mature']) + 10
#init_age = delta_t
#print(init_age)
#t_vec = np.arange(1, 152)
t_vec = np.arange(delta_t, t_max, delta_t)

start_time = timeit.default_timer()
model = Model(**params)
model.set_area(A)
model.init_N(N_0_1, init_age)
#model.set_fire_probabilities(fire_probs=fire_prob)
model.set_weibull_fire(b=b, c=c)
model.simulate(t_vec=t_vec, census_every=1)
elapsed = timeit.default_timer() - start_time
print('{} seconds'.format(elapsed))
np.save('N_tot_vec.npy', model.N_tot_vec)
np.save('census_t.npy', model.census_t)

fig, axs = plt.subplots(4, 1, figsize=(7,20))
slice_step = int(num_reps/10) if num_reps >= 10 else 1
for N_tot_vec in model.N_tot_vec[::slice_step]:
    axs[0].plot(model.census_t,N_tot_vec)
    axs[1].plot(model.census_t,N_tot_vec)
axs[0].axhline(params['K_adult'], ls='--', c='k', alpha=0.35)
axs[1].set_ylim(-0.03*params['K_adult'],params['K_adult'])
axs[2].plot(model.census_t, model.N_tot_vec.mean(axis=0), c='k')
axs[2].axhline(params['K_adult'], ls='--', c='k', alpha=0.35)
axs[3].plot(model.census_t, model.N_tot_vec.mean(axis=0), c='k')
axs[3].axhline(params['K_adult'], ls='--', c='k', alpha=0.35)
axs[3].set_ylim(-0.03*params['K_adult'],1.5*params['K_adult'])
fig.savefig('sim_test.png', bbox_inches='tight')
