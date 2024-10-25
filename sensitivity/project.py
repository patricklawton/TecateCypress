# Imports for running simulations
import numpy as np
from model import Model
import signac as sg
from flow import FlowProject
import pickle
from scipy.optimize import curve_fit
# Additional imports for phase analysis
from matplotlib import pyplot as plt
import matplotlib
from scipy.special import gamma
import scipy
import os
import json
from tqdm.auto import tqdm
from mpi4py import MPI
import timeit
import copy
import sys
import itertools
from itertools import product
MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)

# Open up signac project
project = sg.get_project()

sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])

with open('../model_fitting/mortality/fixed.pkl', 'rb') as handle:
    mort_fixed = pickle.load(handle)

def adjustmaps(maps):
    '''For making SDM and FDM the same shape'''
    dim_len = []
    for dim in range(2):
        dim_len.append(min([m.shape[dim] for m in maps]))
    for mi, m in enumerate(maps):
        maps[mi] = m[0:dim_len[0], 0:dim_len[1]]
    return maps

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
        # Skip if simulations already run at this b value
        if str(b) in list(job.data['N_tot'].keys()): continue
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
    slice_i = 200 #i.e. the end of the burn in period
    avg_start_i = round(job.sp.t_final * 0.10)
    #Aeff = job.sp['Aeff'] #ha
    #N_0_1 = Aeff*mort_fixed['K_adult']
    #num_reps = np.array(job.data[f'N_tot/{min(b_vec)}']).shape[0]
    N_tot_mean_all = np.ones((len(b_vec), job.sp.t_final)) * np.nan
    with job.data:
        for b_i, b in enumerate(b_vec):
            #'''Remove rows with bad init abundance, don't know why it's happening yet'''
            #N_tot = np.array(job.data[f'N_tot/{b}'])
            #check_N0 = np.nonzero(N_tot[:,0] > N_0_1)[0]
            #N_tot = np.delete(N_tot, check_N0, axis=0)
            #N_tot_mean = N_tot.mean(axis=0)
            #''''''
            N_tot_mean = np.array(job.data[f'N_tot_mean/{b}'])
            N_tot_mean_all[b_i] = N_tot_mean
            #job.data[f'fractional_change/{b}'] = (np.mean(N_tot_mean[-slice_i:]) - N_0_1) / N_0_1
        N_0_all = N_tot_mean_all[:, slice_i]
        r_all = (N_tot_mean_all[:, -avg_start_i:].mean(axis=1) - N_0_all) / N_0_all
        # If N=0 by end of the burn in, set to -1
        r_all[np.isnan(r_all)] = -1.0
        for b_i, b in enumerate(b_vec):
            job.data[f'fractional_change/{b}'] = r_all[b_i]
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
    def line(x, m, b):
        return m*x + b

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
            burn_in_end_i = 0
            zero_is = np.nonzero(N_tot_mean == 0)[0]
            if len(zero_is) > 0:
                final_i = min(zero_is)
            else:
                final_i = len(N_tot_mean)
            t = census_t[burn_in_end_i:final_i]
            N_mean_t = N_tot_mean[burn_in_end_i:final_i]

            mu_s = np.product(N_mean_t[1:] / np.roll(N_mean_t, 1)[1:]) ** (1/len(t))
            job.data[f'mu_s/{b}'] = mu_s 
    job.doc['mu_s_computed'] = True

@FlowProject.pre(lambda job: job.doc.get('simulated'))
@FlowProject.post(lambda job: job.doc.get('lambda_s_computed'))
@FlowProject.operation
def compute_lambda_s(job):
    with job.data:
        census_t = np.array(job.data["census_t"])
        burn_in_end_i = 0
        for b in b_vec:
            ## Skip if lambda_s already computed at this b value
            #if str(b) in list(job.data['lambda_s'].keys()): continue
            N_tot = np.array(job.data[f"N_tot/{b}"])
            nonzero_counts = np.count_nonzero(N_tot, axis=1)
            extirpated_replicas = np.nonzero(nonzero_counts < job.sp.t_final)[0]

            # First handle replicas where extirpations occur
            lam_s_extir = []
            for rep_i in extirpated_replicas:
                N_t = N_tot[rep_i]
                zero_i_min = nonzero_counts[rep_i]
                final_i = zero_i_min

                t = census_t[burn_in_end_i:final_i]
                N_slice = N_t[burn_in_end_i:final_i]
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

            lam_s_all = np.concatenate((lam_s_vec, lam_s_extir))
            if len(lam_s_all) != 0:
                job.data[f'lambda_s/{b}'] = np.mean(lam_s_all) 
            else:
                #N_tot_mean = np.array(job.data[f"N_tot_mean/{b}"])
                lam_s = np.exp(-np.log(job.sp.Aeff*job.sp.params.K_adult) / len(t))
                job.data[f'lambda_s/{b}'] = lam_s

    job.doc['lambda_s_computed'] = True

if __name__ == "__main__":
    FlowProject().main()
    
class Phase:
    def __init__(self, **kwargs):
        # Get MPI info
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        # Loop through the keyword arguments and set them as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        with sg.H5Store('shared_data.h5').open(mode='r') as sd:
            self.b_vec = np.array(sd['b_vec'])
        self.tau_vec = self.b_vec * gamma(1+1/self.c)
        if np.isnan(self.final_max_tau): 
            # NaN here means set to max of fri_vec
            self.final_max_tau = max(self.tau_vec)
        self.delta_tau_vec = np.linspace(self.delta_tau_min, self.delta_tau_max, self.delta_tau_samples)
        self.baseline_A_vec = np.linspace(self.baseline_A_min, self.baseline_A_max, self.baseline_A_samples)

        # Generate resource allocation values
        self.ncell_baseline_vec = np.round(self.baseline_A_vec / self.A_cell).astype(int) 
        self.C_vec = self.ncell_baseline_vec * self.tauc_baseline

    def compute_tauc_slice(self, x, method, tau_slice):
        if method == 'initlinear':
            tauc_slice = x[0]*tau_slice + x[1]
        elif method == 'initinverse':
            tauc_slice = x[0] / ((tau_slice/x[1]) + 1)
        return tauc_slice    

    def maximize_preflat(self, method, C, ncell, tau_slice, x0=None, penalty_weight=0.05):
        # Our objective function, the expected value of all tauc > (C/ncell)
        def tauc_expect_preflat(x):
            tauc_slice = self.compute_tauc_slice(x, method, tau_slice)
            tau_mid = (max(tau_slice) - min(tau_slice)) / 2
            # Multiply by -1 bc we want max, not min
            return -1 * np.mean(tauc_slice[tau_slice < tau_mid])

        constraints = []
        # Sum of all tauc must be eq to C
        def tauc_total_con(x):
            tauc_slice = self.compute_tauc_slice(x, method, tau_slice)
            return np.sum(tauc_slice) - C
        constraints.append({'type': 'eq', 'fun': tauc_total_con})
        if method == 'initlinear':
            # Need additional constraint that all tauc > 0
            def tauc_positive(x):
                tauc_slice = self.compute_tauc_slice(x, method, tau_slice)
                return np.min(tauc_slice)
            constraints.append({'type': 'ineq', 'fun': tauc_positive})
            bounds = [(None,0), (0,None)]
            result = scipy.optimize.minimize(tauc_expect_preflat, x0, constraints=constraints, bounds=bounds)
        elif method == 'initinverse':
            # Use a global optimization algorithm for this scaling method
            def penalized_objective(x, penalty_weight=penalty_weight):
                obj_value = tauc_expect_preflat(x)
                constraint_violation = tauc_total_con(x)
                # Add penalty for violating the equality constraint
                penalty = penalty_weight * constraint_violation**2
                return obj_value + penalty
            bounds = [(0,1e4), (0,1e1)]
            result = scipy.optimize.differential_evolution(penalized_objective, bounds=bounds)
        return result

    def initialize(self):
        '''
        Read in and/or organize data used for subsequent analysis
        Do other miscellaneous pre-processing, for example define state variable space
        '''
        if self.rank != self.root:
            # Init data to be read on root
            self.metric_data = None
            self.tau_edges = None
            self.tau_flat = None
            self.data_dir = None
        else:
            # Handle data reading on root alone
            project = sg.get_project()
            tau_diffs = np.diff(self.tau_vec)
            tau_step = (self.b_vec[1]-self.b_vec[0]) * gamma(1+1/self.c)
            self.tau_edges = np.concatenate(([0], np.arange(tau_step/2, self.tau_vec[-1]+tau_step, tau_step)))

            jobs = project.find_jobs({'doc.simulated': True, 'Aeff': self.Aeff, 
                                      't_final': self.t_final, 'method': self.sim_method})
            self.data_dir = f"data/Aeff_{self.Aeff}/tfinal_{self.t_final}"
            if not os.path.isdir(self.data_dir):
                os.makedirs(self.data_dir)
            fn = self.data_dir + "/all_tau.npy"
            if (not os.path.isfile(fn)) or self.overwrite_metrics:
                all_tau = np.tile(self.tau_vec, len(jobs))
                np.save(fn, all_tau)
            else:
                all_tau = np.load(fn)

            self.data_dir = f"data/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}"
            fn = self.data_dir + f"/metric_data.pkl"
            if not os.path.isdir(self.data_dir + f"/metric_{self.metric}"):
                os.makedirs(self.data_dir + f"/metric_{self.metric}")
            if (not os.path.isfile(fn)) or self.overwrite_metrics:
                self.metric_data = {}
                print(f"Creating {self.metric} histogram") 
                if self.metric == 'r': metric_label = 'fractional_change'
                else: metric_label = self.metric
                all_metric = np.array([])
                for job_i, job in enumerate(jobs):
                    with job.data as data:
                        metric_vec = []
                        for b in self.b_vec:
                            if self.metric == 'P_s':
                                # Just consider the cummulative extinction probability by some timestep T
                                T = 300
                                metric_vec.append(1.0 - float(data[f'frac_extirpated/{b}'][T-1]))
                            else:
                                metric_vec.append(float(data[f'{metric_label}/{b}']))
                    all_metric = np.append(all_metric, metric_vec)
                    
                metric_min, metric_max = (min(all_metric), max(all_metric))
                if self.metric in ['lambda_s', 'mu_s']:
                    coarse_step = 0.02
                    fine_step = coarse_step/100
                    #fine_start = 0.6
                    #coarse_grained = np.arange(metric_min, fine_start, coarse_step)
                    #fine_grained = np.arange(fine_start, metric_max + fine_step, fine_step)
                    #metric_edges = np.concatenate((coarse_grained[:-1], fine_grained))
                    metric_edges = np.arange(metric_min, metric_max + fine_step, fine_step)
                else:
                    #metric_thresh = 0.98
                    #metric_min, metric_max = (np.quantile(all_metric, 1-metric_thresh), np.quantile(all_metric, metric_thresh))
                    metric_min, metric_max = (all_metric.min(), all_metric.max())
                    metric_bw = (metric_max - metric_min) / 200
                    metric_edges = np.arange(metric_min, metric_max + metric_bw, metric_bw)
                
                # First plot the metric probability density
                fig, ax = plt.subplots(figsize=(13,8))
                metric_hist = ax.hist2d(all_tau, all_metric, bins=[self.tau_edges, metric_edges], 
                                 norm=matplotlib.colors.LogNorm(vmax=int(len(all_metric)/len(self.b_vec))), 
                                 density=False)
                cbar = ax.figure.colorbar(metric_hist[-1], ax=ax, location="right")
                cbar.ax.set_ylabel('demographic robustness', rotation=-90, fontsize=10, labelpad=20)
                ax.set_xlabel('<FRI>')
                ax.set_ylabel(self.metric)
                figs_dir = f"figs/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}/"
                if not os.path.isdir(figs_dir):
                    os.makedirs(figs_dir)
                print("saving sensitivity figure")
                fig.savefig(figs_dir + f"sensitivity", bbox_inches='tight')
                plt.close(fig)
                # Now remake with density=True for calculations later
                metric_hist = np.histogram2d(all_tau, all_metric, bins=[self.tau_edges, metric_edges], 
                                             density=True)

                self.metric_data.update({'all_metric': all_metric})
                self.metric_data.update({'metric_hist': metric_hist[:3]})
                with open(fn, 'wb') as handle:
                    pickle.dump(self.metric_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(fn, 'rb') as handle:
                    self.metric_data = pickle.load(handle)

            # Read in FDM
            usecols = np.arange(self.ul_coord[0], self.lr_coord[0])
            fdmfn = '../shared_maps/FDE_current_allregions.asc'
            if fdmfn[-3:] == 'txt':
                fdm = np.loadtxt(fdmfn)
            else:
                # Assume these are uncropped .asc maps
                usecols = np.arange(self.ul_coord[0], self.lr_coord[0])
                fdm = np.loadtxt(fdmfn,skiprows=6+self.ul_coord[1],
                                         max_rows=self.lr_coord[1], usecols=usecols)

            # Read in SDM
            sdmfn = "../shared_maps/SDM_1995.asc"
            sdm = np.loadtxt(sdmfn,skiprows=6+self.ul_coord[1],
                                     max_rows=self.lr_coord[1], usecols=usecols)
            sdm, fdm = adjustmaps([sdm, fdm])

            # Convert FDM probabilities to expected fire return intervals
            delta_t = 30
            b_raster = delta_t / np.power(-np.log(1-fdm), 1/self.c)
            tau_raster = b_raster * gamma(1+1/self.c)

            # Flatten and filter FDM & SDM
            # Ignore tau above what we simulated, only a small amount
            # Why are there any zeros in FDM at all?
            maps_filt = (sdm > 0) & (fdm > 0) #& (tau_raster <= max(self.tau_vec))
            self.tau_flat = tau_raster[maps_filt]
        
        # Broadcast data used later to all ranks
        self.metric_data = self.comm.bcast(self.metric_data, root=self.root)
        self.tau_edges = self.comm.bcast(self.tau_edges, root=self.root)
        self.tau_flat = self.comm.bcast(self.tau_flat, root=self.root)
        self.data_dir = self.comm.bcast(self.data_dir, root=self.root)

        # Generate samples of remaining state variables
        self.ncell_tot = len(self.tau_flat)
        # Use the max post alteration tau to get an upper bound on right hand of initial tau slices
        self.tau_argsort_ref = np.argsort(self.tau_flat)
        tau_sorted = self.tau_flat[self.tau_argsort_ref] 
        if max(tau_sorted) > self.final_max_tau:
            self.slice_right_max = min(np.nonzero(tau_sorted >= self.final_max_tau)[0])
        else:
            self.slice_right_max = len(tau_sorted) - 1
        # Min left bound set by user-defined constant
        slice_left_min = np.nonzero(tau_sorted > self.min_tau)[0][0]
        # Generate slice sizes of the tau distribution
        self.ncell_vec = np.linspace(max(self.ncell_baseline_vec), self.slice_right_max, self.ncell_samples)
        self.ncell_vec = np.round(self.ncell_vec).astype(int)
        # Max left bound set by smallest slice size
        self.slice_left_max = self.slice_right_max - min(self.ncell_vec)
        # Generate slice left bound indices, reference tau_argsort_ref for full slice indices
        self.slice_left_all = np.linspace(slice_left_min, self.slice_left_max, self.slice_samples)
        self.slice_left_all = np.round(self.slice_left_all).astype(int)

        if self.tauc_method != "flat":
            # Generate tauc scaling parameters, if needed
            check1 = os.path.isfile(self.data_dir + f"/v_all_{self.tauc_method}.npy")
            check2 = os.path.isfile(self.data_dir + f"/w_all_{self.tauc_method}.npy")
            if np.all([check1, check2, self.overwrite_scaleparams == False]):
                generate_params = False
                self.v_all = np.load(self.data_dir + f"/v_all_{self.tauc_method}.npy")
                self.w_all = np.load(self.data_dir + f"/w_all_{self.tauc_method}.npy")
            else:
                generate_params = True

            if generate_params:
                if self.rank != self.root:
                    self.v_all = None
                    self.w_all = None
                else:
                    print(f"Generating scaling parameters for {self.tauc_method}")
                    self.v_all = np.ones((len(self.C_vec), len(self.ncell_vec), len(self.slice_left_all))) * np.nan
                    self.w_all = np.ones((len(self.C_vec), len(self.ncell_vec), len(self.slice_left_all))) * np.nan

                # Distribute work among ranks
                task_list = list(product(enumerate(self.C_vec), enumerate(self.ncell_vec), enumerate(self.slice_left_all)))
                rank_tasks = np.array_split(task_list, self.num_procs)[self.rank]  # Split tasks across ranks
                # Each rank performs optimization on its assigned subset of tasks
                rank_v_all = []
                rank_w_all = []

                for (C_i, C), (ncell_i, ncell), (slice_left_i, slice_left) in rank_tasks:
                    self.slice_left_max = self.slice_right_max - ncell
                    if slice_left > self.slice_left_max: continue

                    tau_slice = tau_sorted[slice_left:slice_left+ncell]
                    '''Try translating the initial tau so they start at zero'''
                    tau_slice = tau_slice - min(tau_slice)

                    if self.tauc_method == "initlinear":
                        x0 = [-0.003, 900]
                        result = self.maximize_preflat(self.tauc_method, C, ncell, tau_slice, x0=x0)
                    elif self.tauc_method == "initinverse":
                        penalty_weight = 0.05
                        result = self.maximize_preflat(self.tauc_method, C, ncell, tau_slice, penalty_weight=penalty_weight)
                        while not result.success:
                            penalty_weight *= 0.5
                            result = self.maximize_preflat(self.tauc_method, C, ncell, tau_slice, penalty_weight=penalty_weight)

                    rank_v_all.append((C_i, ncell_i, slice_left_i, result.x[0]))
                    rank_w_all.append((C_i, ncell_i, slice_left_i, result.x[1]))

                # Gather results from all ranks on the root rank
                gathered_v_all = self.comm.gather(rank_v_all, root=self.root)
                gathered_w_all = self.comm.gather(rank_w_all, root=self.root)
                # Combine results on the root rank
                if self.rank == self.root:
                    for rank_v, rank_w in zip(gathered_v_all, gathered_w_all):
                        for (C_i, ncell_i, slice_left_i, v_value) in rank_v:
                            self.v_all[C_i, ncell_i, slice_left_i] = v_value
                        for (C_i, ncell_i, slice_left_i, w_value) in rank_w:
                            self.w_all[C_i, ncell_i, slice_left_i] = w_value
                # Broadcast to all ranks
                self.v_all = self.comm.bcast(self.v_all, root=self.root)
                self.w_all = self.comm.bcast(self.w_all, root=self.root)

        # Get bins of initial tau for phase plotting per (delta_tau, C)
        tau_range = 50 - tau_sorted[0]
        self.plotting_tau_bw = tau_range / self.plotting_tau_bw_ratio
        self.plotting_tau_bin_edges = np.arange(tau_sorted[0], 50, self.plotting_tau_bw)
        self.plotting_tau_bin_cntrs = np.array([edge + self.plotting_tau_bw/2 for edge in self.plotting_tau_bin_edges])

        # Store the mean tau in each preset slice for reference later
        self.tau_means_ref = []
        for ncell in self.ncell_vec:
            tau_means_ncell = np.ones(len(self.slice_left_all)) * np.nan
            self.slice_left_max = self.slice_right_max - ncell
            for slice_i, slice_left in enumerate(self.slice_left_all):
                if slice_left <= self.slice_left_max:
                    tau_means_ncell[slice_i] = np.mean(tau_sorted[slice_left:slice_left+ncell])
            self.tau_means_ref.append(tau_means_ncell)

        if self.rank == self.root:
            # Save all state variables
            np.save(self.data_dir + "/delta_tau_vec.npy", self.delta_tau_vec)
            np.save(self.data_dir + "/C_vec.npy", self.C_vec)
            np.save(self.data_dir + "/ncell_vec.npy", self.ncell_vec)
            np.save(self.data_dir + "/slice_left_all.npy", self.slice_left_all)
            if self.tauc_method != "flat":
                np.save(self.data_dir + f"/v_all_{self.tauc_method}.npy", self.v_all)
                np.save(self.data_dir + f"/w_all_{self.tauc_method}.npy", self.w_all)

            # Initialize data for <metric> across (delta_tau, C, ncell, slice_left) space
            self.phase = np.ones((
                                  len(self.delta_tau_vec), len(self.C_vec),
                                  len(self.ncell_vec), len(self.slice_left_all)
                                  )) * np.nan
            self.phase_xs = np.ones((
                                  len(self.delta_tau_vec), len(self.C_vec),
                                  len(self.ncell_vec), len(self.slice_left_all)
                                  )) * np.nan

    def prep_rank_samples(self, ncell): 
        self.slice_left_max = self.slice_right_max - ncell #slice needs to fit
        num_samples = len(self.slice_left_all)

        # Get size and position of sample chunk for this rank
        self.rank_samples = num_samples // self.num_procs
        num_larger_procs = num_samples - self.num_procs*self.rank_samples
        if self.rank < num_larger_procs:
            self.rank_samples = self.rank_samples + 1
            self.rank_start = self.rank * self.rank_samples
        elif self.rank_samples > 0:
            self.rank_start = num_larger_procs + self.rank*self.rank_samples
        else:
            self.rank_start = -1
            self.rank_samples = 0

        # Initialize data for this rank's chunk of samples
        self.metric_expect_rank = np.ones(self.rank_samples) * np.nan
        self.xs_means_rank = np.ones(self.rank_samples) * np.nan

        # Add one sample for computing the no change scenario
        if (ncell==max(self.ncell_vec)) and (self.rank==self.root):
            self.rank_samples += 1

    def change_tau_expect(self, C_i, ncell_i, slice_left_i):
        C = self.C_vec[C_i]
        ncell = self.ncell_vec[ncell_i]
        slice_left = self.slice_left_all[slice_left_i]
        slice_indices = self.tau_argsort_ref[slice_left:slice_left + ncell]
        tau_slice = self.tau_expect[slice_indices]
        # Set max tauc per cell
        final_max_tauc = self.final_max_tau - tau_slice
        # First create array of replacement tau
        replacement_tau = np.ones(ncell) #Initialize
        if self.tauc_method == "flat":
            '''generate these outside loop to speed up'''
            tauc = C / ncell
            tauc_slice = np.repeat(tauc, ncell)
        else:
            v = self.v_all[C_i, ncell_i, slice_left_i]
            w = self.w_all[C_i, ncell_i, slice_left_i]
            tau_slice_ref = self.tau_flat[slice_indices]
            tau_slice_ref = tau_slice_ref - min(tau_slice_ref)
            '''might be worth generating these slices outside loops'''
            tauc_slice = self.compute_tauc_slice([v,w], self.tauc_method, tau_slice_ref)
        
        # Add uncertainty to tau
        ...

        # Find where tauc will push tau beyond max
        xs_filt = (tauc_slice > final_max_tauc) 
        replacement_tau[xs_filt] = self.final_max_tau
        replacement_tau[xs_filt==False] = (tau_slice + tauc_slice)[xs_filt==False]
        # Now replace them in the full array of tau
        self.tau_expect[slice_indices] = replacement_tau 
        # Store the mean value of excess resources, keep at nan if no excess
        xsresources = (tauc_slice - final_max_tauc)[xs_filt]
        if len(xsresources) > 0:
            self.xs_means_rank[slice_left_i-self.rank_start] = np.sum(xsresources) / C

    def calculate_metric_expect(self):
        # Get expected value of metric
        self.metric_expect = 0
        metric_hist = self.metric_data['metric_hist']
        metric_edges = metric_hist[2]
        metric_vals = []
        diffs = np.diff(metric_edges)
        for edge_i, edge in enumerate(metric_edges[:-1]):
            metric_vals.append(edge + diffs[edge_i]/2) 
        metric_vals = np.array(metric_vals)
        for tau_i in range(len(self.tau_edges) - 1):
            '''For final bin, include all cells gte left bin edge, this will inflate the probability of the 
               final bin for cases where delta_tau and/or tauc push tau values past final_max_tau, but 
               hopefully that's a fine approximation for now.'''
            # Get the expected values in this tau bin
            if tau_i < len(self.tau_edges) - 2:
                ncell_within_slice = np.count_nonzero((self.tau_expect >= self.tau_edges[tau_i]) & 
                                                      (self.tau_expect < self.tau_edges[tau_i+1]))
            else:
                ncell_within_slice = np.count_nonzero(self.tau_expect >= self.tau_edges[tau_i])
            # Can skip if zero fire probability in bin
            if ncell_within_slice == 0: continue

            # First get the probability of being in the tau bin
            P_dtau = ncell_within_slice / self.ncell_tot

            # Now get <metric>
            metric_tau = metric_hist[0][tau_i]
            P_metric_tau = scipy.stats.rv_histogram((metric_tau, metric_hist[2]), density=True)
            m_min = metric_hist[2][min(np.nonzero(metric_tau)[0])]
            m_max = metric_hist[2][max(np.nonzero(metric_tau)[0])]
            metric_filt = (metric_vals >= m_min) & (metric_vals <= m_max)
            m = metric_vals[metric_filt]
            metric_expect_tau = np.trapz(y=P_metric_tau.pdf(m)*m, x=m)
            self.metric_expect += metric_expect_tau * P_dtau

    def process_samples(self, delta_tau, C, ncell):
        '''is relocating these indicies significantly slowing things down?'''
        C_i = np.nonzero(self.C_vec == C)[0][0]
        ncell_i = np.nonzero(self.ncell_vec == ncell)[0][0]
        delta_tau_i = np.nonzero(self.delta_tau_vec == delta_tau)[0][0]
        self.slice_left_max = self.slice_right_max - ncell
        # Loop over sampled realizations of this fire alteration strategy
        for rank_sample_i, slice_left_i in enumerate(range(self.rank_start, self.rank_start + self.rank_samples)):
            # First, reset tau with uncertainty
            self.tau_expect = self.tau_flat + delta_tau
            # Check that we are not on the no change scenario
            if rank_sample_i < len(self.metric_expect_rank): 
                slice_left = self.slice_left_all[slice_left_i]
                # Also, check that slice is within allowed range
                if slice_left > self.slice_left_max: continue
                # Now, adjust the tau distribution at cells in slice
                self.change_tau_expect(C_i, ncell_i, slice_left_i)
            self.calculate_metric_expect()
            # Store sample if not computing no change scenario
            if rank_sample_i < len(self.metric_expect_rank): 
                self.metric_expect_rank[slice_left_i-self.rank_start] = self.metric_expect
            # Otherwise save nochange to file
            elif self.rank == self.root:
                self.data_dir = f"data/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}/deltatau_{delta_tau}/"
                if not os.path.isdir(self.data_dir):
                    os.makedirs(self.data_dir)
                fn = self.data_dir + f"nochange_{self.tauc_method}.json"
                with open(fn, "w") as handle:
                    json.dump({f'{self.metric}_expect_nochange': self.metric_expect}, handle)

        # Collect data across ranks
        # Initialize data to store sample means across all ranks
        sendcounts = np.array(self.comm.gather(len(self.metric_expect_rank), root=self.root))
        if self.rank == self.root:
            sampled_metric_expect = np.empty(sum(sendcounts))        
            sampled_xs_means = np.ones(sum(sendcounts)) * np.nan
        else:
            sampled_metric_expect = None
            sampled_xs_means = None
        # Now gather data
        self.comm.Gatherv(self.metric_expect_rank, sampled_metric_expect, root=self.root)
        self.comm.Gatherv(self.xs_means_rank, sampled_xs_means, root=self.root)

        if self.rank == self.root:
            # Save data to full phase matrix
            self.phase[delta_tau_i, C_i, ncell_i, :] = sampled_metric_expect
            self.phase_xs[delta_tau_i, C_i, ncell_i, :] = sampled_xs_means
    
    def plot_phase_slice(self, delta_tau, C, xs=False):
        C_i = np.nonzero(self.C_vec == C)[0][0]
        delta_tau_i = np.nonzero(self.delta_tau_vec == delta_tau)[0][0]
        phase_slice = self.phase[delta_tau_i, C_i, :, :]
        if xs:
            phase_slice = self.phase_xs[delta_tau_i, C_i, :, :]

        # Read in no change value
        self.data_dir = f"data/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}/deltatau_{delta_tau}/"
        fn = self.data_dir + f"nochange_{self.tauc_method}.json"
        with open(fn, "r") as handle:
            metric_nochange = json.load(handle)[f'{self.metric}_expect_nochange']

        # Coarse grain data into plotting matrix
        plotting_matrix = np.ones((len(self.plotting_tau_bin_edges), len(self.ncell_vec))) * np.nan
        for ncell_i in range(len(self.ncell_vec)):
            for tau_i, tau_left in enumerate(self.plotting_tau_bin_edges):
                if tau_i < len(self.plotting_tau_bin_edges) - 2:
                    tau_filt = (self.tau_means_ref[ncell_i] >= tau_left) & (self.tau_means_ref[ncell_i] < tau_left+self.plotting_tau_bw)
                else:
                    tau_filt = self.tau_means_ref[ncell_i] >= tau_left
                metric_expect_slice = phase_slice[ncell_i, tau_filt]
                if (len(metric_expect_slice) > 0) and (np.all(np.isnan(metric_expect_slice)) == False):
                    plotting_matrix[len(self.plotting_tau_bin_edges)-1-tau_i, ncell_i] = np.mean(metric_expect_slice)

        # Now actually make the plot
        fig, ax = plt.subplots(figsize=(12,12))
        axfontsize = 16
        if xs:
            metric_lab = '$<xs>$' 
        else:
            metric_labels = ['$<P_s>$', '$<r>$', '$<\mu>$', '$<\lambda>$']
            metrics = np.array(['P_s', 'r', 'mu_s', 'lambda_s'])
            metric_i = np.nonzero(metrics == self.metric)[0][0]
            metric_lab = metric_labels[metric_i]
        cmap = copy.copy(matplotlib.cm.plasma)
        plotting_matrix = np.ma.masked_where(np.isnan(plotting_matrix),  plotting_matrix)
        plotting_matrix_flat = plotting_matrix.flatten()
        if len(plotting_matrix_flat[plotting_matrix_flat != np.ma.masked]) == 0:
            phase_max = 0
        else:
            phase_max = max(plotting_matrix_flat[plotting_matrix_flat != np.ma.masked])
        cmap.set_bad('white')
        im = ax.imshow(plotting_matrix, norm=matplotlib.colors.Normalize(vmin=metric_nochange, vmax=phase_max), cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax, location="right", shrink=0.6)
        cbar.ax.set_ylabel(fr'{metric_lab}', rotation=-90, fontsize=axfontsize, labelpad=20)
        ytick_spacing = 2
        ytick_labels = np.flip(self.plotting_tau_bin_cntrs)[::ytick_spacing]
        yticks = np.arange(0,len(self.plotting_tau_bin_cntrs),ytick_spacing)
        ax.set_yticks(yticks, labels=np.round(ytick_labels, decimals=3));
        ax.set_ylabel(fr'Average initial $\tau$ in area where $\tau$ is changed', fontsize=axfontsize)
        xtick_spacing = 3
        xticks = np.arange(0,len(self.ncell_vec),xtick_spacing)
        xtick_labels = np.round(self.ncell_vec/self.ncell_tot, 3)
        ax.set_xticks(xticks, labels=xtick_labels[::xtick_spacing]);
        ax.set_xlabel(r'Fraction of species range where $\tau$ is altered ($A/A_{\text{range}}$)', fontsize=axfontsize)
        if self.tauc_method == "flat":
            tauc_vec = C / self.ncell_vec
        else:
            tauc_vec = None
        if hasattr(tauc_vec, "__len__"):
            secax = ax.secondary_xaxis('top')
            secax.set_xticks(xticks, labels=np.round(tauc_vec[::xtick_spacing], decimals=3));
            secax.set_xlabel(r'Change in $\tau$ per unit area ($\hat{\tau}$)', fontsize=axfontsize)
        figs_dir = f"figs/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}/deltatau_{delta_tau}/C_{C}/"
        if not os.path.isdir(figs_dir):
            os.makedirs(figs_dir)
        if xs:
            fn = figs_dir + f"/phase_xs_slice_{self.tauc_method}.png"
        else:
            fn = figs_dir + f"/phase_slice_{self.tauc_method}.png"
        fig.savefig(fn, bbox_inches='tight')
        plt.close(fig)
