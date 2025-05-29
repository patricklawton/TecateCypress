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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy
from scipy.special import gamma
from scipy.interpolate import make_lsq_spline
import os
import json
from tqdm.auto import tqdm
from mpi4py import MPI
import timeit
import copy
import sys
import itertools
from itertools import product
import h5py
from global_functions import adjustmaps, lambda_s, s
MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)

# Open up signac project
project = sg.get_project()

sd_fn = project.fn('shared_data.h5')
with sg.H5Store(sd_fn).open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])

with open('../model_fitting/mortality/fixed.pkl', 'rb') as handle:
    mort_fixed = pickle.load(handle)
with open('../model_fitting/fecundity/fixed.pkl', 'rb') as handle:
    fec_fixed = pickle.load(handle)

@FlowProject.post(lambda job: job.doc.get('simulated'))
@FlowProject.operation
def run_sims(job):
    params = job.sp['params']
    params.update(mort_fixed)
    params.update(fec_fixed)
    Aeff = job.sp['Aeff'] #ha
    delta_t = 1
    num_reps = 5_000
    N_0_1 = Aeff*params['K_adult']
    N_0_1_vec = np.repeat(N_0_1, num_reps)
    init_age = params['a_mature'] - (np.log((1/0.90)-1) / params['eta_rho']) # Age where 90% of reproductive capacity reached
    init_age = int(init_age + 0.5) # Round to nearest integer
    t_vec = np.arange(delta_t, job.sp.t_final+delta_t, delta_t)

    for b in b_vec:
        ## Skip if simulations already run at this b value
        #if 'frac_extirpated' in list(job.data.keys()):
        #    if str(b) in list(job.data['frac_extirpated'].keys()): 
        #        continue
        # Initialize model instance
        model = Model(**params)
        model.set_effective_area(Aeff)
        model.init_N(N_0_1_vec, init_age)
        model.set_t_vec(t_vec)
        model.set_weibull_fire(b=b, c=1.42)
        model.generate_fires()
        # Run simulation
        model.simulate(method=job.sp.method, census_every=1, progress=False) 
        # Store some results
        #N_tot_mean = model.N_tot_vec.mean(axis=0)
        #job.data[f'N_tot_mean/{b}'] = N_tot_mean 
        # Mask where N_tot > 0 
        valid_mask = model.N_tot_vec > 0
        # Store only the first and final abundances, and the number of timesteps in between
        valid_timesteps = np.sum(valid_mask, axis=1) - 1
        ext_mask = valid_timesteps < (model.N_tot_vec.shape[1] - 1)
        final_N = np.take_along_axis(model.N_tot_vec, valid_timesteps[..., None], axis=1)[:, 0]
        first_and_final = np.full((model.N_tot_vec.shape[0], 2), np.nan)
        first_and_final[:, 0] = model.N_tot_vec[:, 0]
        first_and_final[:, 1] = final_N
        job.data[f'first_and_final/{b}'] = first_and_final
        job.data[f'valid_timesteps/{b}'] = valid_timesteps
        job.data[f'ext_mask/{b}'] = ext_mask 
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
    compressed = True
    with job.data:
        census_t = np.array(job.data["census_t"])
        for b in b_vec:
            if compressed:
                valid_timesteps = np.array(job.data[f"valid_timesteps/{b}"])
                ext_mask = np.array(job.data[f"ext_mask/{b}"])
                N_tot = np.array(job.data[f"first_and_final/{b}"])
            else:
                valid_timesteps = None
                ext_mask = None
                N_tot = np.array(job.data[f"N_tot/{b}"])
            lam_s_all = lambda_s(N_tot, compressed=compressed, valid_timesteps=valid_timesteps, ext_mask=ext_mask)
            job.data[f'lambda_s/{b}'] = np.mean(lam_s_all) 
            job.data[f'lambda_s_all/{b}'] = lam_s_all
            job.data[f'lambda_s_std/{b}'] = np.std(lam_s_all, ddof=1)

            if np.any(np.isnan(lam_s_all)):
                print('theres nans')
            if np.any(lam_s_all == 1):
                print(f'theres exact 1s for b={b}')

    job.doc['lambda_s_computed'] = True

@FlowProject.pre(lambda job: job.doc.get('simulated'))
@FlowProject.post(lambda job: job.doc.get('s_computed'))
@FlowProject.operation
def compute_s(job):
    with job.data:
        compressed = True
        for b in b_vec:
            if compressed:
                valid_timesteps = np.array(job.data[f"valid_timesteps/{b}"])
                ext_mask = np.array(job.data[f"ext_mask/{b}"])
                N_tot = np.array(job.data[f"first_and_final/{b}"])
            else:
                valid_timesteps = None
                ext_mask = None
                N_tot = np.array(job.data[f"N_tot/{b}"])
            s_all = s(N_tot, compressed=compressed, valid_timesteps=valid_timesteps, ext_mask=ext_mask)
            if np.any(np.isnan(s_all)): print('theres nans')
            job.data[f's/{b}'] = np.nanmean(s_all) 
            job.data[f's_all/{b}'] = s_all

    job.doc['s_computed'] = True

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

        # Set generator for random uncertainties
        self.rng = np.random.default_rng()
        
        # Create empty file for final results (if overwriting)
        if self.rank == self.root:
            self.data_dir = f"{self.meta_metric}/data/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}"
            if not os.path.isdir(self.data_dir):
                os.makedirs(self.data_dir)
            fn = self.data_dir + f"/phase_{self.tauc_method}.h5"
            if (not os.path.isfile(fn)) or self.overwrite_results:
                with h5py.File(fn, 'w') as handle:
                    pass #Just create file and leave empty for now

    def initialize(self):
        '''
        Read in and/or organize data used for subsequent analysis
        Do other miscellaneous pre-processing, for example define state variable space
        '''
        if self.rank != self.root:
            # Init data to be read on root
            self.tau_flat = None
            self.data_dir = None
        else:
            # Handle data reading on root alone
            project = sg.get_project()
            tau_diffs = np.diff(self.tau_vec)
            tau_step = (self.b_vec[1]-self.b_vec[0]) * gamma(1+1/self.c)

            jobs = project.find_jobs({'doc.simulated': True, 'Aeff': self.Aeff, 
                                      't_final': self.t_final, 'method': self.sim_method})
            self.data_dir = f"{self.meta_metric}/data/Aeff_{self.Aeff}/tfinal_{self.t_final}"
            fn = self.data_dir + "/tau_all.npy"
            if (not os.path.isfile(fn)) or self.overwrite_metrics:
                tau_all = np.tile(self.tau_vec, len(jobs))
                np.save(fn, tau_all)
            else:
                tau_all = np.load(fn)

            self.data_dir = f"{self.meta_metric}/data/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}"
            fn = self.data_dir + f"/metric_all.npy"
            if (not os.path.isfile(fn)) or self.overwrite_metrics:
                # Loop over all jobs and process metric data
                metric_all = np.array([]) # To collect data across all jobs
                metric_spl_all = {} # To collect spline interpolations across all jobs
                for job_i, job in enumerate(jobs):
                    with job.data as data:
                        # Get the metric values across all b (i.e. tau) samples for this demo param sample
                        metric_vec = []
                        for b in self.b_vec:
                            if self.metric == 'P_s':
                                # Just consider the cummulative extinction probability by some timestep T
                                T = 200
                                metric_vec.append(1.0 - float(data[f'frac_extirpated/{b}'][T-1]))
                            else:
                                metric_vec.append(float(data[f'{self.metric}/{b}']))

                        # Store metric samples
                        metric_all = np.append(metric_all, metric_vec)

                        # Create and store interpolation function for this demo sample 
                        t = self.tau_vec[2:-2:2] 
                        k = 3
                        t = np.r_[(self.tau_vec[1],)*(k+1), t, (self.tau_vec[-1],)*(k+1)]
                        metric_spl = make_lsq_spline(self.tau_vec[1:], metric_vec[1:], t, k)
                        metric_spl_all.update({job.sp.demographic_index: metric_spl})

                # Save collected metric data to file
                with open(fn, 'wb') as handle:
                    np.save(fn, metric_all)
                    
                # Create interpolating function for average metric(tau) values
                metric_vec = np.ones(self.tau_vec.size) * np.nan
                for tau_i, tau in enumerate(self.tau_vec):
                    tau_filt = (tau_all == tau)
                    metric_slice = metric_all[tau_filt]
                    metric_vec[tau_i] = np.mean(metric_slice)
                t = self.tau_vec[2:-2:2] 
                k = 3
                t = np.r_[(self.tau_vec[1],)*(k+1), t, (self.tau_vec[-1],)*(k+1)]
                metric_spl = make_lsq_spline(self.tau_vec[1:], metric_vec[1:], t, k)
                metric_spl_all.update({0: metric_spl}) # Store on demographic index 0

                # Save interpolated metric(tau) functions to file
                with open(self.data_dir + "/metric_spl_all.pkl", "wb") as handle:
                    pickle.dump(metric_spl_all, handle)
            #else:
                #metric_all = np.load(fn)
                #with open(self.data_dir + "/metric_spl_all.pkl", "rb") as handle:
                #    metric_spl_all = pickle.load(handle)

            # Plot the metric probability density
            if (not os.path.isfile(self.data_dir + f"/metric_all.npy")) or self.overwrite_metrics:
                print(f"Creating {self.metric} histogram") 
                fig, ax = plt.subplots(figsize=(13,8))
                tau_diffs = np.diff(self.tau_vec)
                tau_step = tau_diffs[1]
                tau_edges = np.concatenate((
                                [self.tau_vec[0]],
                                [tau_diffs[0]/2],
                                np.arange(self.tau_vec[1]+tau_step/2, self.tau_vec[-1]+tau_step, tau_step)
                                           ))
                min_edge_i = 2
                metric_min = min(metric_all[(tau_all >= tau_edges[min_edge_i]) & (tau_all < tau_edges[min_edge_i+1])])
                metric_edges = np.linspace(metric_min, metric_all.max()*1.005, 50)
                cmap = copy.copy(matplotlib.cm.YlGn)
                im = ax.hist2d(tau_all, metric_all, bins=[tau_edges, metric_edges],
                                 norm=matplotlib.colors.LogNorm(vmax=int(len(metric_all)/len(self.b_vec))),
                                 density=False,
                                cmap=cmap)
                # Add the colorbar to inset axis
                cbar_ax = inset_axes(ax, width="5%", height="50%", loc='center',
                                     bbox_to_anchor=(0.5, -0.2, 0.55, 1.1),
                                     bbox_transform=ax.transAxes, borderpad=0)
                sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=None)
                cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", ticks=[0, 0.25, 0.5, 0.75, 1.])
                ax.set_xlabel(r'$\tau$')
                ax.set_ylabel(self.metric)
                figs_dir = f"{self.meta_metric}/figs/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}/"
                if not os.path.isdir(figs_dir):
                    os.makedirs(figs_dir)
                fig.savefig(figs_dir + f"sensitivity", bbox_inches='tight', dpi=50)
                plt.close(fig)

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
        self.tau_flat = self.comm.bcast(self.tau_flat, root=self.root)
        self.data_dir = self.comm.bcast(self.data_dir, root=self.root)

        # Store the total number of cells as an instance variable
        self.ncell_tot = len(self.tau_flat)

        # Initialize tau_expect with tau_flat
        self.tau_expect = self.tau_flat

        # Use the max post alteration tau to get an upper bound on right hand of initial tau slices
        self.tau_argsort_ref = np.argsort(self.tau_flat)
        tau_sorted = self.tau_flat[self.tau_argsort_ref] 
        if max(tau_sorted) > self.final_max_tau:
            self.slice_right_max = min(np.nonzero(tau_sorted >= self.final_max_tau)[0])
        else:
            self.slice_right_max = len(tau_sorted) - 1
        
    def init_strategy_variables(self): 
        tau_sorted = self.tau_flat[self.tau_argsort_ref] 
        # Generate samples of remaining state variables
        # Get samples of total shift to fire regime (C)  
        self.C_vec = self.tauc_min_samples * self.ncell_tot
        # Min left bound set by user-defined constant
        slice_left_min = np.nonzero(tau_sorted > self.min_tau)[0][0]
        # Generate slice sizes of the tau distribution
        self.ncell_vec = np.linspace(self.ncell_min, self.slice_right_max, self.ncell_samples)
        self.ncell_vec = np.round(self.ncell_vec).astype(int)
        # Max left bound set by smallest slice size
        self.slice_left_max = self.slice_right_max - min(self.ncell_vec)
        # Generate slice left bound indices, reference tau_argsort_ref for full slice indices
        self.slice_left_all = np.linspace(slice_left_min, self.slice_left_max, self.slice_samples)
        self.slice_left_all = np.round(self.slice_left_all).astype(int)

        if self.tauc_method != "flat":
            self.generate_scale_params(tau_sorted)

        if self.rank == self.root:
            # Save all state variables
            np.save(self.data_dir + "/C_vec.npy", self.C_vec)
            np.save(self.data_dir + "/ncell_vec.npy", self.ncell_vec)
            np.save(self.data_dir + "/slice_left_all.npy", self.slice_left_all)
            if self.tauc_method != "flat":
                np.save(self.data_dir + f"/v_all_{self.tauc_method}.npy", self.v_all)
                np.save(self.data_dir + f"/w_all_{self.tauc_method}.npy", self.w_all)

            # Initialize data for <metric> across (C, ncell, slice_left) space
            self.phase = np.ones((
                                  len(self.C_vec), len(self.ncell_vec), len(self.slice_left_all)
                                )) * np.nan
            self.phase_xs = np.ones((
                                    len(self.C_vec), len(self.ncell_vec), len(self.slice_left_all)
                                   )) * np.nan

    def generate_scale_params(self, tau_sorted):
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

    def change_tau_expect(self, C, ncell, slice_left):
        slice_indices = self.tau_argsort_ref[slice_left:slice_left + ncell]
        tau_slice = self.tau_expect[slice_indices]
        # Set max tauc per cell
        final_max_tauc = self.final_max_tau - tau_slice
        # First create array of replacement tau
        replacement_tau = np.ones(ncell) #Initialize
        '''could pre-generate tauc slices to speed up'''
        if self.tauc_method == "flat":
            tauc = C / ncell
            tauc_slice = np.repeat(tauc, ncell)
        else:
            C_i = np.nonzero(self.C_vec == C)[0][0]
            ncell_i = np.nonzero(self.ncell_vec == ncell)[0][0]
            slice_left_i = np.nonzero(self.slice_left_all == slice_left)[0][0]
            v = self.v_all[C_i, ncell_i, slice_left_i]
            w = self.w_all[C_i, ncell_i, slice_left_i]
            tau_slice_ref = self.tau_flat[slice_indices]
            tau_slice_ref = tau_slice_ref - min(tau_slice_ref)
            tauc_slice = self.compute_tauc_slice([v,w], self.tauc_method, tau_slice_ref)
        
        # Add uncertainty to tauc slice
        self.generate_eps_tauc(self.mu_tauc, self.sigm_tauc, ncell)
        tauc_slice = tauc_slice + self.eps_tauc

        # Find where tauc will push tau beyond max
        xs_filt = (tauc_slice > final_max_tauc) 
        replacement_tau[xs_filt] = self.final_max_tau
        replacement_tau[xs_filt==False] = (tau_slice + tauc_slice)[xs_filt==False]

        # Now replace them in the full array of tau
        self.tau_expect[slice_indices] = replacement_tau 

        # Replace any tau lt min with min
        self.tau_expect = np.where(self.tau_expect < self.min_tau, self.min_tau, self.tau_expect)

        # Store the mean value of excess resources, keep at nan if no excess
        xsresources = (tauc_slice - final_max_tauc)[xs_filt]
        if hasattr(self, 'xs_means_rank') and (len(xsresources) > 0):
            self.xs_means_rank[self.global_sample_i-self.rank_start] = np.sum(xsresources) / C

    def change_tau_expect_vectorized(self, C_vec, ncell_vec, slice_left_vec, mu_tauc_vec, sigm_tauc_vec):
        n_samples = len(C_vec)

        for i in range(n_samples):
            C = C_vec[i]
            ncell = ncell_vec[i]
            slice_left = slice_left_vec[i]
            mu_tauc = mu_tauc_vec[i]
            sigm_tauc = sigm_tauc_vec[i]

            # Get indices for this sample
            slice_indices = self.tau_argsort_ref[slice_left:slice_left + ncell]

            # Extract current tau slice
            tau_slice = self.tau_expect[i, slice_indices]
            final_max_tauc = self.final_max_tau - tau_slice

            # Compute tauc with noise
            tauc = C / ncell
            tauc_slice = np.full(ncell, tauc)
            eps_tauc = self.rng.normal(loc=mu_tauc, scale=sigm_tauc, size=ncell)
            tauc_slice += eps_tauc

            # Apply capping logic
            replacement_tau = np.where(
                tauc_slice > final_max_tauc,
                self.final_max_tau,
                tau_slice + tauc_slice
            )

            # Insert back into tau_expect
            self.tau_expect[i, slice_indices] = replacement_tau

        # Clip with min_tau
        self.tau_expect = np.where(self.tau_expect < self.min_tau, self.min_tau, self.tau_expect)


    def generate_eps_tau(self):
        self.eps_tau = self.rng.normal(loc=self.mu_tau, scale=self.sigm_tau, size=len(self.tau_flat)) 

    def generate_eps_tau_vectorized(self):
        assert self.mu_tau.shape == self.sigm_tau.shape
        #self.eps_tau = self.rng.normal(
        #                               loc=np.tile(self.mu_tau, (len(self.tau_flat),1)).T, 
        #                               scale=np.tile(self.sigm_tau, (len(self.tau_flat),1)).T, 
        #                               size=(len(self.mu_tau), len(self.tau_flat))
        #                              )
        mu_tau = self.mu_tau[:, None]
        sigm_tau = self.sigm_tau[:, None]     # shape (n, 1)
        m = len(self.tau_flat)
        self.eps_tau = self.rng.normal(loc=mu_tau, scale=sigm_tau, size=(len(mu_tau), m))

    def generate_eps_tauc(self, mu_tauc, sigm_tauc, ncell):
        self.mu_tauc = mu_tauc
        self.sigm_tauc = sigm_tauc
        self.eps_tauc = self.rng.normal(loc=self.mu_tauc, scale=self.sigm_tauc, size=ncell) 

    def prep_rank_samples(self, ncell=None): 
        # Determine the number of samples to parallelize based on some instance variable
        if hasattr(self, 'num_train'):
            # Handle case where we're generating training data for NN
            num_samples = self.num_train 
        #if hasattr(self, 'slice_left_all'):
        elif hasattr(self, 'total_samples'):
            num_samples = self.total_samples
        else:
            # Handle case where we're doing brute force calculations
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
        if (hasattr(self, 'slice_left_all')) and (ncell==max(self.ncell_vec)) and (self.rank==self.root):
            self.rank_samples += 1

    def calculate_metric_expect(self):
        # Get expected value of metric
        '''Replace any tau > max simulated with max tau, similar approximation as before'''
        tau_with_cutoff = np.where(self.tau_expect > self.tau_vec.max(), self.tau_vec.max(), self.tau_expect)
        metric_dist = self.metric_spl(tau_with_cutoff)
        if self.metric == 'P_s':
            # Metric value is bounded by zero, anything lt zero is an interpolation error
            metric_dist[metric_dist < 0] = 0.0
            if np.any(metric_dist < 0): sys.exit(f"metric_expect is negative ({self.metric_expect}), exiting!")
        self.metric_expect = np.mean(metric_dist)

    def calculate_metric_gte(self, threshold, vectorized=False):
        # Get density of metric_k values above some threshold
        '''Replace any tau > max simulated with max tau, similar approximation as before'''
        tau_with_cutoff = np.where(self.tau_expect > self.tau_vec.max(), self.tau_vec.max(), self.tau_expect)
        metric_dist = self.metric_spl(tau_with_cutoff)
        if self.metric == 'P_s':
            #threshold = 0.5
            # Metric value is bounded by zero, anything lt zero is an interpolation error
            metric_dist[metric_dist < 0] = 0.0
            if np.any(metric_dist < 0): sys.exit(f"metric_expect is negative ({self.metric_expect}), exiting!")
        '''Still calling this metric_expect for now but should change this to metapop_metric or something'''
        if vectorized:
            self.metric_expect = np.count_nonzero(metric_dist >= threshold, axis=1) / self.ncell_tot
        else:
            self.metric_expect = np.count_nonzero(metric_dist >= threshold) / self.ncell_tot

    def process_samples(self, C, ncell):
        '''is relocating these indicies significantly slowing things down?'''
        C_i = np.nonzero(self.C_vec == C)[0][0]
        ncell_i = np.nonzero(self.ncell_vec == ncell)[0][0]
        self.slice_left_max = self.slice_right_max - ncell
        # Loop over sampled realizations of this fire alteration strategy
        for rank_sample_i, slice_left_i in enumerate(range(self.rank_start, self.rank_start + self.rank_samples)):
            self.global_sample_i = slice_left_i # Referenced in change_tau_expect

            # First, reset tau with uncertainty
            self.generate_eps_tau()
            self.tau_expect = self.tau_flat + self.eps_tau

            # Check that we are not on the no change scenario
            if rank_sample_i < len(self.metric_expect_rank): 
                slice_left = self.slice_left_all[slice_left_i]
                # Also, check that slice is within allowed range
                if slice_left > self.slice_left_max: continue
                # Now, adjust the tau distribution at cells in slice
                self.change_tau_expect(C, ncell, slice_left)

            # Calculate the metric value
            if self.meta_metric == 'distribution_avg':
                # Calculate <metric>
                self.calculate_metric_expect()
            elif self.meta_metric == 'gte_thresh':
                # Calculate <metric>_k density above some threshold
                self.calculate_metric_gte()

            # Store sample if not computing no change scenario
            if rank_sample_i < len(self.metric_expect_rank): 
                self.metric_expect_rank[slice_left_i-self.rank_start] = self.metric_expect
            # Otherwise save <metric> under no change
            elif self.rank == self.root:
                self.data_dir = f"{self.meta_metric}/data/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}"
                fn = self.data_dir + f"/phase_{self.tauc_method}.h5"
                with h5py.File(fn, "a") as handle:
                    data_key = f"{np.round(self.mu_tau, 3)}/{np.round(self.sigm_tau, 3)}/"
                    data_key += f"{np.round(self.mu_tauc, 3)}/{np.round(self.sigm_tauc, 3)}/metric_nochange"
                    if data_key in handle:
                        handle[data_key][...] = self.metric_expect 
                    else:
                        handle[data_key] = self.metric_expect

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
            self.phase[C_i, ncell_i, :] = sampled_metric_expect
            self.phase_xs[C_i, ncell_i, :] = sampled_xs_means

    def store_phase(self, xs=False): 
        if self.rank == self.root:
            fn = self.data_dir + f"/phase_{self.tauc_method}.h5"
            with h5py.File(fn, "a") as handle:
                data_key = f"{np.round(self.mu_tau, 3)}/{np.round(self.sigm_tau, 3)}/"
                data_key += f"{np.round(self.mu_tauc, 3)}/{np.round(self.sigm_tauc, 3)}/phase"
                if xs:
                    data_key += "_xs"
                    data = self.phase_xs
                else:
                    data = self.phase
                if data_key in handle:
                    handle[data_key][...] = self.phase 
                else:
                    handle[data_key] = self.phase
    
    def plot_phase_slice(self, C, tau_sorted, xs=False):
        if self.rank == self.root:
            # Get bins of initial tau for phase plotting per (eps_tau, C)
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

            C_i = np.nonzero(self.C_vec == C)[0][0]
            phase_slice = self.phase[C_i, :, :]
            if xs:
                phase_slice = self.phase_xs[C_i, :, :]

            # Read in no change value
            fn = self.data_dir + f"/phase_{self.tauc_method}.h5"
            with h5py.File(fn, "r") as handle:
                data_key = f"{np.round(self.mu_tau, 3)}/{np.round(self.sigm_tau, 3)}/"
                data_key += f"{np.round(self.mu_tauc, 3)}/{np.round(self.sigm_tauc, 3)}/metric_nochange"
                metric_nochange = handle[data_key][()]

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
            # Save to file
            figs_dir = f"{self.meta_metric}/figs/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}/"
            figs_dir += f"mutau_{np.round(self.mu_tau,3)}/sigmtau_{np.round(self.sigm_tau, 3)}/"
            figs_dir += f"mutauc_{np.round(self.mu_tauc,3)}/sigmtauc_{np.round(self.sigm_tauc, 3)}/C_{C}"
            if not os.path.isdir(figs_dir):
                os.makedirs(figs_dir)
            if xs:
                fn = figs_dir + f"/phase_xs_slice_{self.tauc_method}.png"
            else:
                fn = figs_dir + f"/phase_slice_{self.tauc_method}.png"
            fig.savefig(fn, bbox_inches='tight', dpi=50)
            plt.close(fig)
