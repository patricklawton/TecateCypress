# Imports for running simulations
import numpy as np
from model import Model
import signac as sg
from flow import FlowProject
import pickle
from scipy.optimize import curve_fit
from global_functions import line
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
from global_functions import adjustmaps, plot_phase
import itertools
MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)

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

    def initialize(self):
        '''
        Read in and/or organize data used for subsequent analysis
        Do other miscellaneous pre-processing, for example define statae variable space
        '''
        if self.rank != self.root:
            # Init data to be read on root
            self.metric_data = None
            self.tau_edges = None
            self.tau_flat = None
        else:
            # Handle data reading on root alone
            project = sg.get_project()
            tau_diffs = np.diff(self.tau_vec)
            tau_step = (self.b_vec[1]-self.b_vec[0]) * gamma(1+1/self.c)
            self.tau_edges = np.concatenate(([0], np.arange(tau_step/2, self.tau_vec[-1]+tau_step, tau_step)))

            jobs = project.find_jobs({'doc.simulated': True, 'Aeff': self.Aeff, 
                                      't_final': self.t_final, 'method': self.sim_method})
            data_root = f"data/Aeff_{self.Aeff}/tfinal_{self.t_final}"
            figs_root = f"figs/Aeff_{self.Aeff}/tfinal_{self.t_final}"
            if not os.path.isdir(data_root):
                os.makedirs(data_root)
            fn = data_root + "/all_tau.npy"
            if (not os.path.isfile(fn)) or self.overwrite_metrics:
                all_tau = np.tile(self.tau_vec, len(jobs))
                np.save(fn, all_tau)
            else:
                all_tau = np.load(fn)

            fn = data_root + "/metric_data.pkl"
            if (not os.path.isfile(fn)) or self.overwrite_metrics:
                self.metric_data = {m: {} for m in self.metrics}
                for metric in self.metrics:
                    print(f"Creating {metric} histogram") 
                    if metric == 'r': metric_label = 'fractional_change'
                    else: metric_label = metric
                    all_metric = np.array([])
                    for job_i, job in enumerate(jobs):
                        with job.data as data:
                            metric_vec = []
                            for b in self.b_vec:
                                metric_vec.append(float(data[f'{metric_label}/{b}']))
                        all_metric = np.append(all_metric, metric_vec)
                        
                    metric_min, metric_max = (min(all_metric), max(all_metric))
                    if metric == 'mu_s':
                        coarse_grained = np.arange(metric_min, -0.02, 0.02)
                        fine_grained = np.arange(-0.02, metric_max + 0.001, 0.0001)
                        metric_edges = np.concatenate((coarse_grained[:-1], fine_grained))
                    elif metric == 'lambda_s':
                        coarse_step = 0.02
                        fine_step = coarse_step/100
                        #fine_start = 0.6
                        #coarse_grained = np.arange(metric_min, fine_start, coarse_step)
                        #fine_grained = np.arange(fine_start, metric_max + fine_step, fine_step)
                        #metric_edges = np.concatenate((coarse_grained[:-1], fine_grained))
                        metric_edges = np.arange(metric_min, metric_max + fine_step, fine_step)
                    else:
                        metric_thresh = 0.98
                        metric_min, metric_max = (np.quantile(all_metric, 1-metric_thresh), np.quantile(all_metric, metric_thresh))
                        metric_bw = (metric_max - metric_min) / 50
                        metric_edges = np.arange(metric_min, metric_max + metric_bw, metric_bw)
                    
                    # First plot the metric probability density
                    fig, ax = plt.subplots(figsize=(13,8))
                    metric_hist = ax.hist2d(all_tau, all_metric, bins=[self.tau_edges, metric_edges], 
                                     norm=matplotlib.colors.LogNorm(vmax=int(len(all_metric)/len(self.b_vec))), 
                                     density=False)
                    cbar = ax.figure.colorbar(metric_hist[-1], ax=ax, location="right")
                    cbar.ax.set_ylabel('demographic robustness', rotation=-90, fontsize=10, labelpad=20)
                    ax.set_xlabel('<FRI>')
                    ax.set_ylabel(metric)
                    if not os.path.isdir(figs_root):
                        os.makedirs(figs_root)
                    fig.savefig(figs_root + f"/sensitivity_{metric}", bbox_inches='tight')
                    plt.close(fig)
                    # Now remake with density=True for calculations later
                    metric_hist = np.histogram2d(all_tau, all_metric, bins=[self.tau_edges, metric_edges], 
                                                 density=True)

                    self.metric_data[metric].update({'all_metric': all_metric})
                    self.metric_data[metric].update({'metric_hist': metric_hist[:3]})
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
            maps_filt = (sdm > 0) & (fdm > 0) & (tau_raster <= max(self.tau_vec))
            self.tau_flat = tau_raster[maps_filt]
        
        # Broadcast data used later to all ranks
        self.metric_data = self.comm.bcast(self.metric_data, root=self.root)
        self.tau_edges = self.comm.bcast(self.tau_edges, root=self.root)
        self.tau_flat = self.comm.bcast(self.tau_flat, root=self.root)

        # Generate samples of remaining state variables
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
        slice_left_max = self.slice_right_max - min(self.ncell_vec)
        # Generate slice left bound indices, reference tau_argsort_ref for full slice indices
        self.slice_left_all = np.linspace(slice_left_min, slice_left_max, self.slice_samples)
        self.slice_left_all = np.round(self.slice_left_all).astype(int)

        # Get bins of initial tau for phase plotting per (delta_tau, C)
        tau_range = tau_sorted[self.slice_right_max] - tau_sorted[0]
        tau_bw = tau_range / self.tau_bw_ratio
        self.tau_bin_edges = np.arange(tau_sorted[0], tau_sorted[self.slice_right_max], tau_bw)
        self.tau_bin_cntrs = np.array([edge + tau_bw/2 for edge in self.tau_bin_edges])
        # Store the mean tau in each preset slice for reference later
        self.tau_means_ref = []
        for ncell in self.ncell_vec:
            tau_means_ncell = np.ones(len(self.slice_left_all)) * np.nan
            slice_left_max = self.slice_right_max - ncell
            for slice_i, slice_left in enumerate(self.slice_left_all):
                if slice_left <= slice_left_max:
                    tau_means_ncell[slice_i] = np.mean(tau_sorted[slice_left:slice_left+ncell])
            self.tau_means_ref.append(tau_means_ncell)

        if self.tauc_method == "scaledtoinit":
            # Solve for tau_max at every (C, ncell, slice_left) for delta_tau=0, then reuse for delta_tau > 0
            if self.rank != self.root:
                self.tau_max_all = None
            else:
                self.tau_max_all = np.ones((len(self.C_vec), 
                                            len(self.ncell_vec), 
                                            len(self.slice_left_all))) * np.nan
                for C_i, C in enumerate(self.C_vec):
                    for ncell_i, ncell in enumerate(self.ncell_vec):
                        slice_left_max = self.slice_right_max - ncell #slice needs to fit
                        for sl_i, slice_left in enumerate(self.slice_left_all):
                            if slice_left > slice_left_max: continue
                            tau_slice = tau_sorted[slice_left:slice_left+ncell]
                            def C_diff(tau_max):
                                # Linear decrease from tau_min to tau_max
                                tauc_slice = tau_max - tau_slice
                                # No negative values allowed, cut off tauc at zero
                                tauc_slice = np.where(tau_slice < tau_max, tauc_slice, 0)
                                return np.abs(C - np.sum(tauc_slice))
                            # Use scipy to solve for tau_max which satisfies C
                            '''Multiple tau_max may be possible? But I'm only solving for one'''
                            diffmin = scipy.optimize.minimize_scalar(C_diff, bounds=(C/ncell, 1e5))
                            self.tau_max_all[C_i, ncell_i, sl_i] = diffmin.x
                # Save tau_max data
                np.save(data_root + "/tau_max_all.npy", self.tau_max_all)
            self.tau_max_all = self.comm.bcast(self.tau_max_all, root=self.root)

        if self.rank == 0:
            # Save all state variables
            np.save(data_root + "/delta_tau_vec.npy", self.delta_tau_vec)
            np.save(data_root + "/C_vec.npy", self.C_vec)
            np.save(data_root + "/ncell_vec.npy", self.ncell_vec)
            np.save(data_root + "/slice_left_all.npy", self.slice_left_all)

            # Save tau bin centers for plotting
            np.save(data_root + "/tau_bin_cntrs.npy", self.tau_bin_cntrs)

            # Initialize data for <metric> across (delta_tau, C, ncell, slice_left) space
            self.phase = np.ones((
                                  len(self.delta_tau_vec), len(self.C_vec),
                                  len(self.ncell_vec), len(self.slice_left_all)
                                  )) * np.nan

    def prep_rank_samples(self, metric, ncell): 
        slice_left_max = self.slice_right_max - ncell #slice needs to fit
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
        self.tau_means_rank = np.ones(self.rank_samples) * np.nan
        self.metric_expect_rank = np.ones(self.rank_samples) * np.nan
        if metric == self.metrics[0]:
            self.rank_xs_means = np.ones(self.rank_samples) * np.nan

        # Add one sample for computing the no change scenario
        if (ncell==max(self.ncell_vec)) and (self.rank==self.root):
            self.rank_samples += 1
            #print(f"adding nochange to rank {self.rank} for {self.rank_samples} total iterations at ncell {ncell}")

    def change_tau_expect(self, metric, delta_tau, C_i, ncell_i, slice_left_sample_i):
            C = self.C_vec[C_i]
            ncell = self.ncell_vec[ncell_i]
            # Get left bound index of slice for this rank
            slice_left = self.slice_left_all[slice_left_sample_i]
            # First, check that slice is within allowed range
            if slice_left <= self.slice_left_max:
                # If so, get full range of indices from reference
                slice_indices = self.tau_argsort_ref[slice_left:slice_left + ncell]
                tau_slice = self.tau_expect[slice_indices]
                # Store mean value of slice
                self.tau_means_rank[slice_left_sample_i-self.rank_start] = np.mean(tau_slice)
                # Set max tauc per cell
                final_max_tauc = self.final_max_tau - tau_slice
                # First create array of replacement tau
                replacement_tau = np.ones(ncell) #Initialize
                if self.tauc_method == "flat":
                    tauc = C / ncell
                    tauc_slice = np.repeat(tauc, ncell)
                elif self.tauc_method == "scaledtoinit":
                    tau_max = self.tau_max_all[C_i, ncell_i, slice_left_sample_i]
                    tau_slice_ref = self.tau_flat[slice_indices]
                    '''might be worth generating these slices outside loops'''
                    tauc_slice = tau_max - tau_slice_ref
                    tauc_slice = np.where(tau_slice_ref < tau_max, tauc_slice, 0)
                    '''shouldn't also be capped by tauc_baseline?'''
                # Find where tauc will push tau beyond max
                xs_filt = (tauc_slice > final_max_tauc) 
                replacement_tau[xs_filt] = self.final_max_tau
                replacement_tau[xs_filt==False] = (tau_slice + tauc_slice)[xs_filt==False]
                # Now replace them in the full array of tau
                self.tau_expect[slice_indices] = replacement_tau 
                if metric == self.metrics[0]:
                    # Store the mean value of excess resources, keep at nan if no excess
                    xsresources = (tauc_slice - final_max_tauc)[xs_filt]
                    if len(xsresources) > 0:
                        self.rank_xs_means[slice_left_sample_i-self.rank_start] = np.mean(xsresources)
                '''cut off tau distribution at max tau we simulated, shouldn't leave this forever'''
                self.tau_expect = self.tau_expect[self.tau_expect <= max(self.tau_vec)] 
                self.ncell_tot = len(self.tau_expect)

    def process_samples(self, metric, delta_tau, C, ncell):
        print(f"process {self.rank_samples} samples on rank {self.rank}")
        C_i = np.nonzero(self.C_vec == C)[0][0]
        ncell_i = np.nonzero(self.ncell_vec)[0][0]
        # Loop over sampled realizations of this fire alteration strategy
        for rank_sample_i, slice_left_sample_i in enumerate(range(self.rank_start, self.rank_start + self.rank_samples)):
            # First reset tau with uncertainty
            self.tau_expect = self.tau_flat + delta_tau
            # Adjust the tau distribution at cells in slice
            # Skip if computing no change scenario
            if rank_sample_i < len(self.tau_means_rank): 
                self.change_tau_expect(metric, delta_tau, C_i, ncell_i, slice_left_sample_i)
            ...
