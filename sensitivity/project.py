# Imports for running simulations
import numpy as np
from model import Model
import signac as sg
from flow import FlowProject
import pickle
# Additional imports for phase analysis
from scipy.special import gamma
from scipy.interpolate import make_lsq_spline, RBFInterpolator
from scipy.optimize import minimize
from mpi4py import MPI
import copy
import sys
import os
import json
import h5py
from tqdm.auto import tqdm
from global_functions import adjustmaps, lambda_s, Rescaler
from itertools import product

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
        
        # Create directory for final results (if not already there)
        if self.rank == self.root:
            self.data_dir = f"{self.meta_metric}/data/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}"
            if not os.path.isdir(self.data_dir):
                os.makedirs(self.data_dir)

    def initialize(self):
        '''
        Read in and organize data used for subsequent analysis
        Also pre-process (interpolate) population sim data if needed 
        '''
        if self.rank != self.root:
            # Init data to be read on root
            self.tau_flat = None
            self.data_dir = None
            self.figs_dir = None
            self.num_demographic_samples = None
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
            self.figs_dir = f"{self.meta_metric}/figs/Aeff_{self.Aeff}/tfinal_{self.t_final}/metric_{self.metric}"
            #self.figs_dir = os.path.join('/','Volumes', 'Macintosh HD', 'Users', 'patrick',
            #                             'Google Drive', 'My Drive', 'Research', 'Regan', 'Figs/')
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

                # Save interpolated metric(tau) functions to file
                with open(self.data_dir + "/metric_spl_all.pkl", "wb") as handle:
                    pickle.dump(metric_spl_all, handle)                    
            else:
                # Read in all splined interpolations of metric(tau)
                with open(self.data_dir + "/metric_spl_all.pkl", "rb") as handle:
                    metric_spl_all = pickle.load(handle)
            
            # Set total number of pop model param sets to instance att
            self.num_demographic_samples = len(metric_spl_all)

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
            maps_filt = (sdm > 0) & (fdm > 0) 
            self.tau_flat = tau_raster[maps_filt]
 
        # Broadcast data used later to all ranks
        self.tau_flat = self.comm.bcast(self.tau_flat, root=self.root)
        self.data_dir = self.comm.bcast(self.data_dir, root=self.root)
        self.figs_dir = self.comm.bcast(self.figs_dir, root=self.root)
        self.num_demographic_samples = self.comm.bcast(self.num_demographic_samples, root=self.root)

        # Optionally store some map related data
        if hasattr(self, 'extra_attributes') and (self.extra_attributes != None):
            named_extra_atts = ['tau_raster', 'maps_filt', 'metric_spl_all']
            assert (np.all([att in named_extra_atts for att in self.extra_attributes]))
            if 'tau_raster' in self.extra_attributes:
                self.tau_raster = self.comm.bcast(tau_raster, root=self.root)
            if 'maps_filt' in self.extra_attributes:
                self.maps_filt = self.comm.bcast(maps_filt, root=self.root)
            if 'metric_spl_all' in self.extra_attributes:
                self.metric_spl_all = self.comm.bcast(metric_spl_all, root=self.root)

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
        
    def init_decision_parameters(self, overwrite=False, suffix=''): 
        tau_sorted = self.tau_flat[self.tau_argsort_ref] 
        # Generate samples of remaining state variables
        # Get samples of total shift to fire regime (C)  
        self.C_vec = self.tauc_min_samples * self.ncell_tot
        # Min left bound set by user-defined constant
        slice_left_min = np.nonzero(tau_sorted > self.min_tau)[0][0]
        # Generate slice sizes of the tau distribution
        if isinstance(self.ncell_samples, int):
            self.ncell_vec = np.linspace(self.ncell_min, self.slice_right_max, self.ncell_samples)
            self.ncell_vec = np.round(self.ncell_vec).astype(int)
        elif isinstance(self.ncell_samples, np.ndarray):
            self.ncell_vec = self.ncell_samples.copy()
        else:
            sys.exit('ncell_samples needs to be int or numpy array') 
        # Max left bound set by smallest slice size
        self.slice_left_max = self.slice_right_max - min(self.ncell_vec)
        # Generate slice left bound indices, reference tau_argsort_ref for full slice indices
        if isinstance(self.slice_samples, int):
            self.slice_left_all = np.linspace(slice_left_min, self.slice_left_max, self.slice_samples)
            self.slice_left_all = np.round(self.slice_left_all).astype(int)
        elif isinstance(self.slice_samples, np.ndarray):
            self.slice_left_all = self.slice_samples.copy()
        else:
            sys.exit('slice_samples needs to be int or numpy array')

        if self.rank == self.root:
            if overwrite:
                # Save all state variables
                np.save(self.data_dir + f"/C_vec{suffix}.npy", self.C_vec)
                np.save(self.data_dir + f"/ncell_vec{suffix}.npy", self.ncell_vec)
                np.save(self.data_dir + f"/slice_left_all{suffix}.npy", self.slice_left_all)

                # If mc sampling, save number of uncertainty samples to file
                if hasattr(self, 'num_eps_combs'):
                    np.save(self.data_dir + f'/num_eps_combs{suffix}.npy', self.num_eps_combs)
    
    def load_decision_parameters(self, suffix=''):
        '''
        Load pre-existing decision parameters given an initialized Phase instance
        '''
        self.C_vec = np.load(self.data_dir + f'/C_vec{suffix}.npy')
        self.ncell_vec = np.load(self.data_dir + f'/ncell_vec{suffix}.npy')
        self.slice_left_all = np.load(self.data_dir + f'/slice_left_all{suffix}.npy')
        self.slice_left_max = self.slice_right_max - min(self.ncell_vec)

    def change_tau_expect(self, C, ncell, slice_left):
        slice_indices = self.tau_argsort_ref[slice_left:slice_left + ncell]
        tau_slice = self.tau_expect[slice_indices]
        # Set max tauc per cell
        final_max_tauc = self.final_max_tau - tau_slice
        # First create array of replacement tau
        replacement_tau = np.ones(ncell) #Initialize
        '''could pre-generate tauc slices to speed up'''
        tauc = C / ncell
        
        # Add uncertainty to tauc slice
        '''
        Draw per-pop % changes from baseline with normal noise, then apply to tauc values
        '''
        perpop_p = self.rng.normal(loc=self.mu_tauc, scale=self.sigm_tauc, size=ncell)
        tauc_slice = (1 + perpop_p) * tauc

        # Find where tauc will push tau beyond max
        xs_filt = (tauc_slice > final_max_tauc) 
        replacement_tau[xs_filt] = self.final_max_tau
        replacement_tau[xs_filt==False] = (tau_slice + tauc_slice)[xs_filt==False]

        # Now replace them in the full array of tau
        self.tau_expect[slice_indices] = replacement_tau 

        # Replace any tau lt min with min (again)
        # Have to do this twice bc tauc can become negative under uncertainty
        self.tau_expect = np.where(self.tau_expect < self.min_tau, self.min_tau, self.tau_expect)

        ## Make sure all tau values are reasonable (i.e. not negative or below min)
        #assert np.all(self.tau_expect >= self.min_tau)

    def generate_eps_tau(self):
        mu_tau = self.mu_tau * self.tau_flat
        sigm_tau = np.abs(mu_tau * self.sigm_tau)
        self.eps_tau = self.rng.normal(loc=mu_tau, scale=sigm_tau, size=len(self.tau_flat)) 

    def generate_tau(self):
        '''
        Draw per-pop % changes from baseline with normal noise, then apply to tau distribution
        '''
        perpop_p = self.rng.normal(loc=self.mu_tau, scale=self.sigm_tau, size=len(self.tau_flat))
        self.tau_expect = (1 + perpop_p) * self.tau_flat

        # Replace any tau lt min with min
        self.tau_expect = np.where(self.tau_expect < self.min_tau, self.min_tau, self.tau_expect)

    def generate_eps_tauc(self, mu_tauc, sigm_tauc, ncell):
        self.eps_tauc = self.rng.normal(loc=mu_tauc, scale=sigm_tauc, size=ncell) 

    def prep_rank_samples(self, ncell=None): 
        # Determine the number of samples to parallelize based on some instance variable
        if hasattr(self, 'num_train'):
            # Handle case where we're generating training data for NN
            num_samples = self.num_train 
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

    def calculate_metric_expect(self):
        # Get expected value of metric
        '''Replace any tau > max simulated with max tau, similar approximation as before'''
        tau_with_cutoff = np.where(self.tau_expect > self.tau_vec.max(), self.tau_vec.max(), self.tau_expect)
        metric_dist = self.metric_spl(tau_with_cutoff)
        self.metric_expect = np.mean(metric_dist)

    def calculate_metric_gte(self, threshold):
        # Get density of metric_k values above some threshold
        '''Replace any tau > max simulated with max tau, similar approximation as before'''
        tau_with_cutoff = np.where(self.tau_expect > self.tau_vec.max(), self.tau_vec.max(), self.tau_expect)
        metric_dist = self.metric_spl(tau_with_cutoff)
        '''Still calling this metric_expect for now but should change this to metapop_metric or something'''
        self.metric_expect = np.count_nonzero(metric_dist >= threshold) / self.ncell_tot

    def compute_nochange(self, metric_thresh):
        with open(self.data_dir + "/metric_spl_all.pkl", "rb") as handle:
            metric_spl_all = pickle.load(handle)
        self.tau_expect = self.tau_flat
        self.metric_spl = metric_spl_all[0]
        self.calculate_metric_gte(metric_thresh)
        np.save(self.data_dir + '/meta_metric_nochange.npy', self.metric_expect)

    def process_samples(self, minima, maxima, suffix, metric_thresh):
        # Define ordered list of parameter keys
        param_keys = ['C', 'ncell', 'slice_left',
                      'mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'demographic_index']

        # Read in all splined interpolations of metric(tau)
        with open(self.data_dir + "/metric_spl_all.pkl", "rb") as handle:
            metric_spl_all = pickle.load(handle)

        # Generate parameter combinations
        if self.rank != self.root:
            x_decision = None
            decision_indices = None
            fixed_metric_mask = None
        else:
            '''Could move next few lines to initialize method and save to file / instance att'''
            # Store indices of demographic samples that will have a fixed value of S=0
            fixed_metric_mask = np.full(len(metric_spl_all), False)
            tau_test = np.linspace(self.min_tau, self.final_max_tau, 1000)
            for demographic_index, metric_spl in metric_spl_all.items():
                # Check that spline lower bound makes sense
                assert metric_spl(self.min_tau) > 0
                # Check if lambda ever greater than threshold
                if np.all(metric_spl(tau_test) < metric_thresh) and (self.meta_metric == 'gte_thresh'):
                    fixed_metric_mask[demographic_index] = True
            print(f'{np.count_nonzero(fixed_metric_mask)} of {len(metric_spl_all)} demograhpic samples are always unstable')

            ## Initialize decision combinations ### 
            max_decision_combs = self.C_vec.size * self.ncell_vec.size * self.slice_left_all.size
            x_decision = np.full((max_decision_combs, 3), np.nan)

            # Generate combinations
            if isinstance(self.ncell_samples, np.ndarray) and isinstance(self.slice_samples, np.ndarray):
                decision_combs = [_ for _ in zip(self.C_vec, self.ncell_vec, self.slice_left_all)]
            elif isinstance(self.ncell_samples, int) and isinstance(self.slice_samples, int):
                decision_combs = product(self.C_vec,
                                self.ncell_vec,
                                self.slice_left_all)
            else:
                sys.exit("ncell_samples and slice_samples need to be the same type of either int or ndarray")

            # Place combinations in x_decision
            for decision_comb_i, decision_comb in enumerate(decision_combs):
                # Check that slice is within allow range
                if decision_comb[2] > (self.slice_right_max - decision_comb[1]): continue
                x_decision[decision_comb_i, :] = decision_comb

            # Filter out any invalid param samples
            nan_filt = np.any(np.isnan(x_decision), axis=1)
            x_decision = x_decision[~nan_filt, :]

            # Also filter out any repeats
            x_decision = np.unique(x_decision, axis=0)

            # Compute the final number of decision combinations
            num_decision_combs = x_decision.shape[0]

            # Shuffle to give all procs ~ the same amount of work
            self.rng.shuffle(x_decision)

            if suffix != "_baseline":
                # Generate keys for each decision combination for use in the h5 file
                # for example, (C_i=0, n_i=1, l_i=0) -> '0.1.0'
                decision_indices = np.zeros((num_decision_combs, 3)).astype(int)
                
                # Make sure all parameter values can be mapped to decision vectors
                assert np.all(np.isin(x_decision[:,0], self.C_vec))
                assert np.all(np.isin(x_decision[:,1], self.ncell_vec))
                assert np.all(np.isin(x_decision[:,2], self.slice_left_all))

                # Now actually generate and save the index keys 
                if isinstance(self.ncell_samples, np.ndarray) and isinstance(self.slice_samples, np.ndarray):
                    decision_indices[:,0] = np.array([np.argwhere(self.C_vec == v)[0][0] for v in x_decision[:,0]])
                    decision_indices[:,1] = np.array([np.argwhere(self.ncell_vec == v)[0][0] for v in x_decision[:,1]])
                    decision_indices[:,2] = np.array([np.argwhere(self.slice_left_all == v)[0][0] for v in x_decision[:,2]])
                else:
                    decision_indices[:,0] = np.searchsorted(self.C_vec, x_decision[:,0])
                    decision_indices[:,1] = np.searchsorted(self.ncell_vec, x_decision[:,1])
                    decision_indices[:,2] = np.searchsorted(self.slice_left_all, x_decision[:,2])
                decision_indices = np.array(['.'.join([str(x) for x in indices]) for indices in decision_indices])
                np.save(self.data_dir + f'/decision_indices{suffix}.npy', decision_indices)
            else:
                # For baseline we don't use these indices; just put a dummy index in
                decision_indices = np.array(['0.0.0'])

        # Broadcast samples to all ranks
        x_decision = self.comm.bcast(x_decision, root=self.root)
        decision_indices = self.comm.bcast(decision_indices, root=self.root)
        fixed_metric_mask = self.comm.bcast(fixed_metric_mask, root=self.root)

        # Initialize results file
        results = h5py.File(self.data_dir + f'/phase{suffix}.h5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
        for idx in decision_indices:
            if suffix != "_baseline": 
                results.create_dataset(idx, (self.num_eps_combs,), dtype='float64')
                results.create_dataset(idx + 'uncertainty_samples', (self.num_eps_combs, len(minima)), dtype='float64')
            else:
                results.create_dataset(idx, (x_decision.shape[0],), dtype='float64')
                results.create_dataset(idx + 'decision_samples', x_decision.shape, dtype='float64')

        # Set number of samples as instance attribute
        self.total_samples = x_decision.shape[0]

        # Initialze stuff for parallel processing
        self.prep_rank_samples()
        if self.rank == self.root:
            pbar = tqdm(total=self.rank_samples, position=0, dynamic_ncols=True, file=sys.stderr)
            pbar_step = int(self.rank_samples/100) if self.rank_samples >= 100 else 1

        # Generate meta metric values (i.e. results)
        for rank_sample_i, decision_i in enumerate(range(self.rank_start, self.rank_start + self.rank_samples)):
            # Initialize array for results at this decision
            meta_metric_all = np.full(self.num_eps_combs, np.nan)

            # Initialize x_all with repeats of this particular decision
            x_all = np.tile(x_decision[decision_i], (self.num_eps_combs, 1))

            # Add columns for uncertain param samples
            x_all = np.hstack((
                                  x_all,
                                  np.full((x_all.shape[0], len(minima)), np.nan)
                              ))

            # Loop over each uncertain param to populate samples 
            for i, uncertain_param in enumerate(param_keys[3:]):
                uncertain_param_i = i + 3
                 
                # Get min, max and type for this param
                _min = minima[uncertain_param]
                _max = maxima[uncertain_param]
                assert (type(_min)==type(_min))
                _type = type(_min)

                if _min == _max:
                    samples = np.repeat(_min, self.num_eps_combs)
                else:
                    # Generate and store nonzero samples
                    if uncertain_param in ['mu_tau', 'mu_tauc']:
                        # Pct change from baseline will be drawn from beta distributions 
                        alpha = 1.5 # Start by defining an ad-hoc value for alpha

                        # Solve for the transformed mode of zero
                        x_mode = (-_min) / (_max - _min)

                        # Use transformed mode to solve for 2nd shape parameter _beta
                        _beta = ((alpha - 1) / x_mode) - alpha + 2

                        # Draw samples from the beta distribution and apply linear transform
                        samples = self.rng.beta(alpha, _beta, self.num_eps_combs)
                        samples = _min + ((_max - _min) * samples)
                    elif uncertain_param in ['sigm_tau', 'sigm_tauc']:
                        # Multipliers to get spread of tau/tauc post pct change, assume uniform
                        samples = self.rng.uniform(_min, _max, self.num_eps_combs)
                    elif uncertain_param == 'demographic_index':
                        # Samples of pop parameters follow inferred posterior, so just select
                        # indices of pre-generated samples uniformly
                        samples = self.rng.integers(_min, _max, self.num_eps_combs, endpoint=True)
                    else:
                        sys.exit(f'No protocol specified for how to draw samples of {uncertain_param}') 
                x_all[:, uncertain_param_i] = samples

            # Compute outcome under all uncertainty samples
            for x_i, x in enumerate(x_all):
                # First, check if lambda(tau) < lambda^* for all tau; for these S=0
                if fixed_metric_mask[int(x[-1])]:
                    meta_metric_all[x_i] = 0.0
                
                # Otherwise, actually compute the new value of S
                else:
                    # Assign parameter values for this sample
                    for i, param in enumerate(param_keys):
                        if i in [1,2]:
                            param_val = int(x[i])
                        elif param == 'demographic_index':
                            # Retrieve the spline function for this demographic sample
                            demographic_index = int(x[i])
                            param_val = metric_spl_all[demographic_index]
                            param = 'metric_spl'
                        else:
                            param_val = float(x[i])
                        setattr(self, param, param_val)

                    # Add in uncertainty on baseline tau values
                    self.generate_tau() 

                    # Shift selected tau values (including uncertainty)
                    self.change_tau_expect(self.C, self.ncell, self.slice_left)

                    # Compute and store metric value
                    if self.meta_metric == 'gte_thresh':
                        self.calculate_metric_gte(metric_thresh)
                    if suffix == '_baseline':
                        self.metric_expect_rank[rank_sample_i] = self.metric_expect    
                    else:
                        meta_metric_all[x_i] = self.metric_expect

            # Store meta metric values and uncertainty samples in h5
            if suffix != '_baseline':
                key = decision_indices[decision_i]
                results[key][:] = meta_metric_all
                results[key + 'uncertainty_samples'][:] = x_all[:, 3:]

            # Update progress (root only)
            if (self.rank == self.root) and (rank_sample_i % pbar_step == 0):
                pbar.update(pbar_step); print()

        if suffix == '_baseline':
            # We use a dummy key for the baseline case    
            key = '0.0.0'

            # Gather data; first initialize data to store samples across all ranks
            sendcounts = np.array(self.comm.gather(len(self.metric_expect_rank), root=self.root))
            if self.rank == self.root:
                sampled_metric_expect = np.empty(sum(sendcounts))
            else:
                sampled_metric_expect = None
            self.comm.Gatherv(self.metric_expect_rank, sampled_metric_expect, root=self.root)

            results[key][:] = sampled_metric_expect
            results[key + 'decision_samples'][:] = x_decision

        if self.rank == self.root:
            pbar.close()

        results.close()

    def postprocess_phase_uncertain(self):
        phase = h5py.File(self.data_dir + '/phase_uncertain.h5', 'r')
        decision_indices = np.load(self.data_dir + '/decision_indices_uncertain.npy')
        Sstar_vec = np.linspace(0, 1, 150) # Range of target outcomes for computing robustness
        np.save(self.data_dir + "/Sstar_vec.npy", Sstar_vec)

        # Calculate the robustness at each threshold and strategy combination
        rob_all = np.full((
                            Sstar_vec.size, self.C_vec.size, 
                            self.ncell_vec.size, self.slice_left_all.size
                          ), np.nan)
        for thresh_i, thresh in enumerate(Sstar_vec):
            for indices in decision_indices:
                C_i, ncell_i, sl_i = [int(i) for i in indices.split('.')]
                key = ''.join([str(x) for x in indices])
                meta_metric_samples = np.array(phase[key])
                counts = np.count_nonzero(meta_metric_samples >= thresh)
                robustness = counts / self.num_eps_combs 
                rob_all[thresh_i, C_i, ncell_i, sl_i] = robustness

        # Save robustness results to file
        if self.rank == self.root:
            np.save(self.data_dir + "/rob_all.npy", rob_all)
            
        # Now find the strategies which optimize robustness per threshold, C combination
        maxrob = np.full((len(Sstar_vec), len(self.C_vec)), np.nan)
        argmaxrob = np.full((len(Sstar_vec), len(self.C_vec), 2), np.nan)
        for (thresh_i, thresh), (C_i, C) in product(enumerate(Sstar_vec), enumerate(self.C_vec)):
            rob_slice = rob_all[thresh_i, C_i]
            if np.any(~np.isnan(rob_slice)):
                # Store the max robustness at this (thresh, C) coordinate
                maxrob[thresh_i, C_i] = np.nanmax(rob_slice)
                
                # Also store the optimal param indices
                optimal_param_i = np.unravel_index(np.nanargmax(rob_slice, axis=None), rob_slice.shape)
                argmaxrob[thresh_i, C_i] = optimal_param_i

        # Save maxrob and argmaxrob to files
        np.save(self.data_dir + "/maxrob.npy", maxrob)
        np.save(self.data_dir + "/argmaxrob.npy", argmaxrob)

        # Close phase results file
        phase.close()

    def interp_optima_baseline(self, taucmin, nn):
        # Read decision params and S values into x_obs and y_obs, respectively
        with h5py.File(self.data_dir + '/phase_baseline.h5', 'r') as phase:
            x_obs = np.array(phase['0.0.0decision_samples'])
            y_obs = np.array(phase['0.0.0'])

        # Filter for selected C val, checking that its in data first
        assert np.any(np.isclose(self.C_vec/self.ncell_tot, taucmin))
        C_i = np.isclose(self.C_vec/self.ncell_tot, taucmin).argmax()
        C_mask = (x_obs[:, 0] == (self.C_vec[C_i]))
        x_obs = x_obs[C_mask, 1:]
        y_obs = y_obs[C_mask]

        # Rescale inputs and outputs
        x_rescaler = Rescaler(x_obs.min(axis=0), x_obs.max(axis=0))
        x_obs = x_rescaler.rescale(x_obs)
        y_rescaler = Rescaler(y_obs.min(axis=0), y_obs.max(axis=0))
        y_obs = y_rescaler.rescale(y_obs)

        # Interpolate S(n, l) 
        interp_baseline = RBFInterpolator(x_obs, y_obs, neighbors=nn, smoothing=0.0)

        # Define objective function for optimization 
        def objective(decision_params):
            S = interp_baseline([decision_params])
            return -S

        # Use optimal decision from exisiting samples as starting point
        argmax = np.nanargmax(y_obs)
        x0 = x_obs[argmax]

        # Optimize using scipy
        bounds = ((0, 1), (0, 1)) # Remeber, we rescaled the training data
        cons = [{'type': 'ineq', 'fun': lambda x:  1 - x[1] - x[0]}] # Constrain l < (n_tot - n)
        res = minimize(objective, x0, method='COBYLA', bounds=bounds, constraints=cons)
        x_opt = x_rescaler.descale(res.x).astype(int)
        S_opt = y_rescaler.descale(-res.fun)

        # Save results to file
        np.save(self.data_dir + '/decision_opt_baseline.npy', x_opt) 
        np.save(self.data_dir + '/S_opt_baseline.npy', S_opt)

    def interp_optima_uncertain(self, taucmin, nn, smoothing, num_restarts):
        # Read in required data
        decision_indices = np.load(self.data_dir + '/decision_indices.npy')
        Sstar_vec = np.load(self.data_dir + "/Sstar_vec.npy")
        rob_all = np.load(self.data_dir + "/rob_all.npy")
        meta_metric_nochange = float(np.load(self.data_dir + '/meta_metric_nochange.npy'))
        S_opt_baseline = np.load(self.data_dir + '/S_opt_baseline.npy')
        n_opt_baseline, l_opt_baseline = np.load(self.data_dir + '/decision_opt_baseline.npy')

        # Filter for selected C val, checking that its in data first
        assert np.any(np.isclose(self.C_vec/self.ncell_tot, taucmin))
        C_i = np.isclose(self.C_vec/self.ncell_tot, taucmin).argmax()
        rob_all_filtered = rob_all[:, C_i, ...]

        # Read robustness vals and decision params into y_obs and x_obs, respectively
        y_obs = rob_all_filtered.flatten()
        indices = np.unravel_index(np.arange(y_obs.size), rob_all_filtered.shape)
        x_obs = np.full((y_obs.size, len(rob_all_filtered.shape)), np.nan)
        x_obs[:, 0] = Sstar_vec[indices[0]]
        x_obs[:, 1] = self.ncell_vec[indices[1]]
        x_obs[:, 2] = self.slice_left_all[indices[2]]

        # Replace results for invalid param sets with zero
        nan_filt = np.isnan(y_obs)
        y_obs[nan_filt] = 0.0

        # Rescale inputs and outputs
        x_rescaler = Rescaler(x_obs.min(axis=0), x_obs.max(axis=0))
        x_obs = x_rescaler.rescale(x_obs)
        y_rescaler = Rescaler(y_obs.min(axis=0), y_obs.max(axis=0))
        y_obs = y_rescaler.rescale(y_obs)

        # Interpolate robustness(S^*, n, l) given C
        interp = RBFInterpolator(x_obs, y_obs, neighbors=nn, smoothing=smoothing)

        # Define objective function for optimization 
        def objective(decision_params, *args):
            Sstar = args[0]

            # Get robustness value from interpolation
            x = np.full(len(decision_params)+1, np.nan)
            x[0] = Sstar
            x[1:] = decision_params
            try:
                robustness = interp([x])
            except:
                robustness = 0

            return -robustness # Negate bc using minimization algorithm

        # Now step through S^* values and find decisions that optimize robustness
        n_opt_interp = np.full(Sstar_vec.size, np.nan)
        l_opt_interp = np.full(Sstar_vec.size, np.nan)
        for Sstar_i, Sstar in enumerate(Sstar_vec):
            # Use optimal decisions from exisiting samples as starting points
            argsort = np.argsort(rob_all_filtered[Sstar_i, :], axis=None)
            nan_filt = np.isnan(rob_all_filtered[Sstar_i, :].ravel()[argsort])
            argsort = argsort[~nan_filt]

            n_opt_samples = []
            l_opt_samples = []

            for x0_position in argsort[-num_restarts:]:
                n0_i, l0_i = np.unravel_index(x0_position, rob_all_filtered.shape[1:])
                n0, l0 = (self.ncell_vec[n0_i], self.slice_left_all[l0_i])

                # Rescale to interpolation scale
                Sstar, n0, l0 = x_rescaler.rescale([Sstar, n0, l0])

                # Use an optimizer that can handle some noise in the objective
                x0 = np.array([n0, l0])
                cons = [
                    {'type': 'ineq', 'fun': lambda x: x[0]},          # n >= 0
                    {'type': 'ineq', 'fun': lambda x: 1 - x[0]},      # n <= 1
                    {'type': 'ineq', 'fun': lambda x: x[1]},          # l >= 0
                    {'type': 'ineq', 'fun': lambda x: 1 - x[1]},      # l <= 1
                    {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}  # l < (1 - n)
                ]
                res = minimize(objective, x0, args=(Sstar,), method='COBYLA', constraints=cons)

                _, n_opt, l_opt = x_rescaler.descale([Sstar, res.x[0], res.x[1]])
                n_opt_samples.append(n_opt)
                l_opt_samples.append(l_opt)

            # Take the mean over multiple optimization runs
            n_opt_interp[Sstar_i] = np.mean(n_opt_samples)
            l_opt_interp[Sstar_i] = np.mean(l_opt_samples)

        decision_opt_uncertain = np.full((Sstar_vec.size, 2), np.nan)
        decision_opt_uncertain[:, 0] = n_opt_interp
        decision_opt_uncertain[:, 1] = l_opt_interp
        np.save(self.data_dir + '/decision_opt_uncertain.npy', decision_opt_uncertain.astype(int))
