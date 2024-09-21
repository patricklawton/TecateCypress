import numpy as np
from model import Model
import signac as sg
from flow import FlowProject
import pickle
from scipy.optimize import curve_fit
from global_functions import line

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
    Aeff = job.sp['Aeff'] #ha
    delta_t = 1
    num_reps = 1_250
    N_0_1 = Aeff*params['K_adult']
    N_0_1_vec = np.repeat(N_0_1, num_reps)
    '''Could figure out mature age in a more sophisticated way'''
    init_age = round(params['a_mature']) + 20
    t_vec = np.arange(delta_t, job.sp.t_final+delta_t, delta_t)

    for b in b_vec:
        model = Model(**params)
        model.set_effective_area(Aeff)
        model.init_N(N_0_1_vec, init_age)
        model.set_t_vec(t_vec)
        model.set_weibull_fire(b=b, c=1.42)
        model.generate_fires()
        model.simulate(method=job.sp.method, census_every=1, progress=False) 
        # Store some results
        N_tot_mean = model.N_tot_vec.mean(axis=0)
        job.data[f'N_tot_mean/{b}'] = N_tot_mean 
        job.data[f'N_tot/{b}'] = model.N_tot_vec
        frac_extirpated = np.array([sum(model.N_tot_vec[:,t_i]==0)/model.N_tot_vec.shape[0] for t_i in range(model.N_tot_vec.shape[1])])
        job.data[f'frac_extirpated/{b}'] = frac_extirpated

    job.data['census_t'] = model.census_t
    job.doc['simulated'] = True

@FlowProject.pre(lambda job: job.doc.get('simulated'))
@FlowProject.post(lambda job: job.doc.get('fractional_change_computed'))
@FlowProject.operation
def compute_fractional_change(job):
    slice_i = round(job.sp.t_final * 0.25)
    Aeff = job.sp['Aeff'] #ha
    N_0_1 = Aeff*mort_fixed['K_adult']
    for b in b_vec:
        with job.data:
            N_tot_mean = np.array(job.data[f'N_tot_mean/{b}'])
        job.data[f'fractional_change/{b}'] = (np.mean(N_tot_mean[-slice_i:]) - N_0_1) / N_0_1
    job.doc['fractional_change_computed'] = True

@FlowProject.pre(lambda job: job.doc.get('simulated'))
@FlowProject.post(lambda job: job.doc.get('Nf_computed'))
@FlowProject.operation
def compute_Nf(job):
    slice_i = round(job.sp.t_final * 0.25)
    for b in b_vec:
        with job.data:
            N_tot_mean = np.array(job.data[f'N_tot_mean/{b}'])
        job.data[f'Nf/{b}'] = np.mean(N_tot_mean[-slice_i:])
    job.doc['Nf_computed'] = True

@FlowProject.pre(lambda job: job.doc.get('simulated'))
@FlowProject.post(lambda job: job.doc.get('decay_rate_computed'))
@FlowProject.operation
def compute_decay_rate(job):
    with job.data:
        census_t = np.array(job.data["census_t"])
        for b in b_vec:
            N_tot_mean = np.array(job.data[f"N_tot_mean/{b}"])

            burn_in_end_i = 200
            final_i = len(N_tot_mean)

            x = census_t[burn_in_end_i:final_i]
            y = N_tot_mean[burn_in_end_i:final_i]
            popt, pcov = curve_fit(line, x, y)
            job.data[f'decay_rate/{b}'] = popt[0] / line(x[0], *popt)
    job.doc['decay_rate_computed'] = True

@FlowProject.pre(lambda job: job.doc.get('simulated'))
@FlowProject.post(lambda job: job.doc.get('mu_s_computed'))
@FlowProject.operation
def compute_mu_s(job):
    with job.data:
        census_t = np.array(job.data["census_t"])
        for b in b_vec:
            N_tot_mean = np.array(job.data[f"N_tot_mean/{b}"])
            burn_in_end_i = 200
            zero_is = np.nonzero(N_tot_mean == 0)[0]
            if len(zero_is) > 0:
                if (min(zero_is) < burn_in_end_i) or ((min(zero_is) - burn_in_end_i) < 300):
                    start_i = 0
                    final_i = min(zero_is)
                    tooquick = True
                else:
                    start_i = burn_in_end_i
                    final_i = min(zero_is)
                    tooquick = False
            else:
                start_i = burn_in_end_i
                final_i = len(N_tot_mean)
                tooquick = False

            t = census_t[start_i:final_i]
            N_mean_t = N_tot_mean[start_i:final_i]
            if tooquick:
                mu_s = -np.log(N_mean_t[0]) / len(t)
            else:
                mu_s = np.sum(np.log(N_mean_t[1:] / np.roll(N_mean_t, 1)[1:])) / len(t)
            job.data[f'mu_s/{b}'] = mu_s 
    job.doc['mu_s_computed'] = True

@FlowProject.pre(lambda job: job.doc.get('simulated'))
@FlowProject.post(lambda job: job.doc.get('lambda_s_computed'))
@FlowProject.operation
def compute_lambda_s(job):
    with job.data:
        census_t = np.array(job.data["census_t"])
        burn_in_end_i = 200
        for b in b_vec:
            #N_tot_mean = np.array(job.data[f"N_tot_mean/{b}"])
            N_tot = np.array(job.data[f"N_tot/{b}"])
            nonzero_counts = np.count_nonzero(N_tot, axis=1)
            extirpated_replicas = np.nonzero(nonzero_counts < job.sp.t_final)[0]

            # First handle replicas where extirpations occur
            lam_s_extir = []
            for rep_i in extirpated_replicas:
                N_t = N_tot[rep_i]
                zero_i_min = nonzero_counts[rep_i]
                final_i = zero_i_min
                if (zero_i_min < burn_in_end_i) or ((zero_i_min - burn_in_end_i) < 300):
                    start_i = 0
                    tooquick = True
                else:
                    start_i = burn_in_end_i
                    tooquick = False
                t = census_t[start_i:final_i]
                N_slice = N_t[start_i:final_i]
                if tooquick:
                    #lam_s_replica = -np.log(N_slice[0]) / len(t)
                    lam_s_replica = np.exp(-np.log(N_slice[0]) / len(t))
                else:
                    #lam_s_replica = np.sum(np.log(N_slice[1:] / np.roll(N_slice, 1)[1:])) / len(t)
                    lam_s_replica = np.product(N_slice[1:] / np.roll(N_slice, 1)[1:]) ** (1/len(t))
                lam_s_extir.append(lam_s_replica)

            # Now handle cases with no extirpation
            N_tot = np.delete(N_tot, extirpated_replicas, axis=0)
            start_i = burn_in_end_i
            final_i = N_tot.shape[1]
            N_slice = N_tot[:,start_i:final_i]
            #log_ratios = np.log(N_slice[:,1:] / np.roll(N_slice, 1, 1)[:,1:])
            #lam_s_vec = np.sum(log_ratios, axis=1) / N_slice.shape[1]
            lam_products = np.product(N_slice[:,1:] / np.roll(N_slice, 1, 1)[:,1:], axis=1)
            lam_s_vec = lam_products ** (1/N_slice.shape[1]) 

            job.data[f'lambda_s/{b}'] = np.mean(np.concatenate((lam_s_vec, lam_s_extir))) 

    job.doc['lambda_s_computed'] = True

if __name__ == "__main__":
    FlowProject().main()
