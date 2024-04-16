import numpy as np
from model import Model
import json
import signac as sg
from flow import FlowProject

# Open up signac project
project = sg.get_project()

# Read in map parameters
params = {}
for pr in ['mortality', 'fecundity']:
    with open('../model_fitting/{}/map.json'.format(pr), 'r') as handle:
        params.update(json.load(handle)) 

@FlowProject.post(lambda job: job.doc.get('simulated'))
@FlowProject.operation
def run_sims(job):
    A = 1 #ha
    num_reps = 10000
    N_0_1 = np.repeat(0.9*A*params['K_adult'], num_reps)
    init_age = round(params['a_mature']) + 10
    t_vec = np.arange(1, 152)

    model = Model(**params)
    model.set_area(A)
    model.init_N(N_0_1, init_age)
    model.simulate(t_vec=t_vec, census_every=2, fire_probs=job.sp.fire_prob)
    
    job.data['N_tot_vec'] = model.N_tot_vec
    job.data['census_yrs'] = model.census_yrs
    job.doc['simulated'] = True

if __name__ == "__main__":
    FlowProject().main()
