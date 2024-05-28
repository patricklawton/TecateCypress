import numpy as np
from model import Model
import signac as sg
from flow import FlowProject
import pickle
from scipy.optimize import curve_fit

# Open up signac project
project = sg.get_project()

sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])

with open('../model_fitting/mortality/fixed.pkl', 'rb') as handle:
    mort_fixed = pickle.load(handle)

@FlowProject.post(lambda job: job.doc.get('simulated'))
@FlowProject.operation
def run_sims(job):
    params = job.sp['params']
    params.update(mort_fixed)
    A = 1 #ha
    delta_t = 1
    num_reps = 1_000
    N_0_1 = 0.9*A*params['K_adult']
    N_0_1_vec = np.repeat(N_0_1, num_reps)
    '''Could figure out mature age in a more sophisticated way'''
    init_age = round(params['a_mature']) + 20
    t_vec = np.arange(delta_t, 200+delta_t, delta_t)

    for b in b_vec:
        model = Model(**params)
        model.set_area(A)
        model.init_N(N_0_1_vec, init_age)
        model.set_weibull_fire(b=b, c=1.42)
        model.simulate(t_vec=t_vec, census_every=1) 
        # Store some results
        N_tot_mean = model.N_tot_vec.mean(axis=0)
        job.data['N_tot_mean/{}'.format(b)] = N_tot_mean 
        job.data['fractional_change/{}'.format(b)] = (np.mean(N_tot_mean[-40:]) - N_0_1) / N_0_1
        frac_extirpated = np.array([sum(model.N_tot_vec[:,t_i]==0)/model.N_tot_vec.shape[0] for t_i in range(model.N_tot_vec.shape[1])])
        max_i = np.nonzero(N_tot_mean == max(N_tot_mean))[0][0]
        # Set a threshold for the max extirpation fraction before we stop fitting a line
        extir_thresh = 0.95
        if np.any(frac_extirpated >= extir_thresh):
            max_extir_i = min(np.nonzero(frac_extirpated >= extir_thresh)[0])
        else:
            max_extir_i = model.N_tot_vec.shape[1]
        def line(x, m):
            return m*x
        if max_i >= max_extir_i:
            print(max_i, max_extir_i)
            print(frac_extirpated)
            #print(job.sp.rep)
            job.data['extirpation_rate/{}'.format(b)] = 0
        else:
            popt, pcov = curve_fit(line, model.census_t[max_i:max_extir_i], frac_extirpated[max_i:max_extir_i])
            job.data['extirpation_rate/{}'.format(b)] = popt[0]

    job.data['census_t'] = model.census_t
    job.doc['simulated'] = True

if __name__ == "__main__":
    FlowProject().main()
