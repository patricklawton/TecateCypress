import signac as sg
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.special import gamma
import scipy
import os
import json
from tqdm.auto import tqdm
from mpi4py import MPI
import timeit
import pickle
import copy
import sys
from global_functions import adjustmaps, plot_phase
import itertools
MPI.COMM_WORLD.Set_errhandler(MPI.ERRORS_RETURN)

# Some constants
progress = False
overwrite_metrics = False
if len(sys.argv) > 1:
    tauc_method = sys.argv[1]
    if tauc_method not in ["flat", "scaledtoinit"]: sys.exit("Invalid tauc_method")
else:
    sys.exit("Need to provide tauc_method argument")
metrics = ['lambda_s']#['mu_s']#['r', 'Nf', 'g']
metric_thresh = 0.98
c = 1.42
Aeff = 7.29
t_final = 600
sim_method = 'nint'
ul_coord = [1500, 2800]
lr_coord = [2723, 3905]
with sg.H5Store('shared_data.h5').open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])
tau_vec = b_vec * gamma(1+1/c)
#final_max_tau = 66 #yrs
final_max_tau = max(tau_vec)
min_tau = 3
A_cell = 270**2 / 1e6 #km^2
tau_bw_ratio = 50 #For binning initial tau (with uncertainty)
tauc_baseline = 200 #years, max fire return interval change possible
                    #at lowest ncell for a given C value
metric_integrand_ratio = 800
ncell_step = 8_000#5_000#3_000
slice_spacing = 8_000#500
#baseline_areas = np.arange(10, 160, 30) #km^2
baseline_areas = np.arange(10, 160, 50) #km^2
delta_tau_step = 0.25
#delta_tau_sys = np.arange(-10, 10+delta_tau_step, delta_tau_step).astype(float) #years
delta_tau_sys = np.array([-10.0, 0.0,  10.0])
rng = np.random.default_rng()

# Generate resource allocation values
ncell_baseline_vec = np.round(baseline_areas / A_cell).astype(int) 
C_vec = ncell_baseline_vec * tauc_baseline

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
root = 0
num_procs = comm_world.Get_size()

# Init data to be read on rank 0
metric_data = None
all_tau = None
tau_edges = None
K_adult = None
tau_flat = None
sdm_flat = None
maps_filt = None
mapindices = None

# Handle data reading on rank 0 alone
if my_rank == 0:
    project = sg.get_project()
    tau_step = (b_vec[1]-b_vec[0]) * gamma(1+1/c)
    tau_edges = np.concatenate(([0], np.arange(tau_step/2, tau_vec[-1]+tau_step, tau_step)))

    jobs = project.find_jobs({'doc.simulated': True, 'Aeff': Aeff, 't_final': t_final, 'method': sim_method})
    data_root = f"data/Aeff_{Aeff}/tfinal_{t_final}"
    figs_root = f"figs/Aeff_{Aeff}/tfinal_{t_final}"
    if not os.path.isdir(data_root):
        os.makedirs(data_root)
    fn = data_root + "/all_tau.npy"
    if (not os.path.isfile(fn)) or overwrite_metrics:
        all_tau = np.tile(tau_vec, len(jobs))
        np.save(fn, all_tau)
    else:
        all_tau = np.load(fn)

    fn = data_root + "/metric_data.pkl"
    if (not os.path.isfile(fn)) or overwrite_metrics:
        metric_data = {m: {} for m in metrics}
        for metric in [m for m in metrics if m != "Nf"]: #metrics:
            print(f"Creating {metric} histogram") 
            if metric == 'r': metric_label = 'fractional_change'
            elif metric in ['Nf', 'mu_s', 'lambda_s']: metric_label = metric
            elif metric == 'g': metric_label = 'decay_rate'
            all_metric = np.array([])
            for job_i, job in enumerate(jobs):
                with job.data as data:
                    metric_vec = []
                    for b in b_vec:
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
                metric_min, metric_max = (np.quantile(all_metric, 1-metric_thresh), np.quantile(all_metric, metric_thresh))
                metric_bw = (metric_max - metric_min) / 50
                metric_edges = np.arange(metric_min, metric_max + metric_bw, metric_bw)
            
            # First plot the metric probability density
            fig, ax = plt.subplots(figsize=(13,8))
            metric_hist = ax.hist2d(all_tau, all_metric, bins=[tau_edges, metric_edges], 
                             norm=matplotlib.colors.LogNorm(vmax=int(len(all_metric)/len(b_vec))), 
                             density=False)
            cbar = ax.figure.colorbar(metric_hist[-1], ax=ax, location="right")
            cbar.ax.set_ylabel('demographic robustness', rotation=-90, fontsize=10, labelpad=20)
            ax.set_xlabel('<FRI>')
            ax.set_ylabel(metric)
            print(f"on metric {metric}")
            if not os.path.isdir(figs_root):
                os.makedirs(figs_root)
            fig.savefig(figs_root + f"/sensitivity_{metric}", bbox_inches='tight')
            plt.close(fig)
            # Now remake with density=True for calculations later
            metric_hist = np.histogram2d(all_tau, all_metric, bins=[tau_edges, metric_edges], 
                                         density=True)

            metric_data[metric].update({'all_metric': all_metric})
            metric_data[metric].update({'metric_hist': metric_hist[:3]})
        with open(fn, 'wb') as handle:
            pickle.dump(metric_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(fn, 'rb') as handle:
            metric_data = pickle.load(handle)

    # Read in FDM
    usecols = np.arange(ul_coord[0],lr_coord[0])
    fdmfn = '../shared_maps/FDE_current_allregions.asc'
    if fdmfn[-3:] == 'txt':
        fdm = np.loadtxt(fdmfn)
    else:
        # Assume these are uncropped .asc maps
        usecols = np.arange(ul_coord[0],lr_coord[0])
        fdm = np.loadtxt(fdmfn,skiprows=6+ul_coord[1],
                                 max_rows=lr_coord[1], usecols=usecols)

    # Read in SDM
    sdmfn = "../shared_maps/SDM_1995.asc"
    sdm = np.loadtxt(sdmfn,skiprows=6+ul_coord[1],
                             max_rows=lr_coord[1], usecols=usecols)
    sdm, fdm = adjustmaps([sdm, fdm])

    # Convert FDM probabilities to expected fire return intervals 
    delta_t = 30
    b_raster = delta_t / np.power(-np.log(1-fdm), 1/c)
    tau_raster = b_raster * gamma(1+1/c)

    # Flatten and filter FDM & SDM
    # Ignore tau above what we simulated, only a small amount
    # Why are there any zeros in FDM at all?
    maps_filt = (sdm > 0) & (fdm > 0) #& (tau_raster <= max(tau_vec))
    mapindices = np.argwhere(maps_filt) #2d raster indicies to reference later 
    tau_flat = tau_raster[maps_filt]
    sdm_flat = sdm[maps_filt]

    # Read in K_adult (fixed, not inferred) from mortality parameters
    with open("../model_fitting/mortality/map.json", "r") as handle:
        mort_params = json.load(handle)
    K_adult = mort_params['K_adult']
# Broadcast some data to all ranks
metric_data = comm_world.bcast(metric_data)
K_adult = comm_world.bcast(K_adult)
all_tau = comm_world.bcast(all_tau)
tau_edges = comm_world.bcast(tau_edges)
tau_flat = comm_world.bcast(tau_flat)
sdm_flat = comm_world.bcast(sdm_flat)
maps_filt = comm_world.bcast(maps_filt)
mapindices = comm_world.bcast(mapindices)

# Generate samples of remaining state variables
# Use the max post alteration tau to get an upper bound on right hand of initial tau slices
tau_argsort_ref = np.argsort(tau_flat)
tau_sorted = tau_flat[tau_argsort_ref] 
if max(tau_sorted) > final_max_tau:
    slice_right_max = min(np.nonzero(tau_sorted >= final_max_tau)[0])
else:
    slice_right_max = len(tau_sorted) - 1
# Min left bound set by user-defined constant
slice_left_min = np.nonzero(tau_sorted > min_tau)[0][0]
# Generate slice sizes of the tau distribution
'''add step to endpoint? s.t. max possible ncell is included'''
ncell_vec = np.arange(max(ncell_baseline_vec), slice_right_max, ncell_step)
# Max left bound set by smallest slice size
slice_left_max = slice_right_max - min(ncell_vec)
# Generate indices of slice left bounds, 
# reference back to tau_argsort_ref for full slice indices
slice_left_all = np.arange(slice_left_min, slice_left_max, slice_spacing)

# Get bins of initial tau for phase data
tau_range = tau_sorted[slice_right_max] - tau_sorted[0]
tau_bw = tau_range / tau_bw_ratio
tau_bin_edges = np.arange(tau_sorted[0], tau_sorted[slice_right_max], tau_bw)
tau_bin_cntrs = np.array([edge + tau_bw/2 for edge in tau_bin_edges])
# Store the mean tau in each preset slice for reference later
tau_means_ref = []
for ncell in ncell_vec:
    tau_means_ncell = np.ones(len(slice_left_all)) * np.nan
    slice_left_max = slice_right_max - ncell
    for slice_i, slice_left in enumerate(slice_left_all):
        if slice_left <= slice_left_max:
            tau_means_ncell[slice_i] = np.mean(tau_sorted[slice_left:slice_left+ncell])
    tau_means_ref.append(tau_means_ncell)

if tauc_method == "scaledtoinit":
    # Solve for tau_max at every (C, ncell, slice_left) for delta_tau=0, then reuse for delta_tau > 0
    if my_rank != root:
        tau_max_all = None
    else:
        tau_max_all = np.ones((len(C_vec), len(ncell_vec), len(slice_left_all))) * np.nan
        for C_i, C in enumerate(C_vec):
            for ncell_i, ncell in enumerate(ncell_vec):
                slice_left_max = slice_right_max - ncell #slice needs to fit
                for sl_i, slice_left in enumerate(slice_left_all):
                    if slice_left > slice_left_max: continue
                    tau_slice = tau_sorted[slice_left:slice_left+ncell]
                    def C_diff(tau_max):
                        # Linear decrease from tau_min to tau_max
                        tauc_slice = tau_max - tau_slice
                        # No negative values allowed, cut off tauc at zero
                        tauc_slice = np.where(tau_slice < tau_max, tauc_slice, 0)
                        return np.abs(C - np.sum(tauc_slice))
                    diffmin = scipy.optimize.minimize_scalar(C_diff, bounds=(C/ncell, 1e5))
                    tau_max_all[C_i, ncell_i, sl_i] = diffmin.x
        # Save tau_max data
        np.save(data_root + "/tau_max_all.npy", tau_max_all)
    tau_max_all = comm_world.bcast(tau_max_all, root=0)

if my_rank == 0:
    # Save all state variables
    np.save(data_root + "/delta_tau_vec.npy", delta_tau_sys)
    np.save(data_root + "/C_vec.npy", C_vec)
    np.save(data_root + "/ncell_vec.npy", ncell_vec)
    np.save(data_root + "/slice_left_all.npy", slice_left_all)

    # Save tau bin centers for plotting
    np.save(data_root + "/tau_bin_cntrs.npy", tau_bin_cntrs)

    # Initialize data for <metric> across (delta_tau, C, ncell, slice_left) space
    full_phase = np.ones((
                    len(delta_tau_sys), len(C_vec), 
                    len(ncell_vec), len(slice_left_all)
                        )) * np.nan

# Loop over considered values of tau uncertainty
for delta_tau_i, delta_tau in enumerate(delta_tau_sys): 
    if my_rank==0: print(f"delta_tau: {delta_tau}")
    ##tau_uncertain = tau_flat + rng.normal(0, delta_tau, len(tau_flat))
    ##tau_min = 5
    ##tau_uncertain = np.where(tau_uncertain > tau_min, tau_uncertain, tau_min)
    ##fire_freqs = 1 / tau_uncertain
    # Add uncertainty to <tau> values and re-sort
    tau_expected = tau_flat + delta_tau
    # Sort habitat suitability data by tau value per cell
    sdm_sorted = sdm_flat[tau_argsort_ref]

    # Loop over different resource C values
    for C_i, C in enumerate(C_vec):
        if my_rank == 0: print(f"On C value {C}")
        #if tauc_method == "flat":
        #    tauc_vec = np.array([C/ncell for ncell in ncell_vec])
        #else:
        #    tauc_vec = None

        for metric in metrics:
            if my_rank == 0:
                start_time = timeit.default_timer()
                # Initialize final data matricies
                phase_space = np.ones((len(tau_bin_edges), len(ncell_vec))) * np.nan
                # Add a matrix for computing excess resources; only do this once
                if metric == metrics[0]:
                    phase_space_xs = np.ones((len(tau_bin_edges), len(ncell_vec))) * np.nan
                figs_root = f"figs/Aeff_{Aeff}/tfinal_{t_final}/deltatau_{delta_tau}/metric_{metric}"
                if not os.path.isdir(figs_root):
                    os.makedirs(figs_root)
                # Delete existing map figures
                if os.path.isdir(figs_root + f"/C_{C}"):
                    for item in os.listdir(figs_root + f"/C_{C}"):
                        if "map_ncell_" in item:
                            os.remove(os.path.join(figs_root + f"/C_{C}", item))
                data_root = f"data/Aeff_{Aeff}/tfinal_{t_final}/deltatau_{delta_tau}/C_{C}/metric_{metric}"
                if not os.path.isdir(data_root):
                    os.makedirs(data_root)
                # Delete existing map files
                for item in os.listdir(data_root):
                    if "map_ncell_" in item:
                        os.remove(os.path.join(data_root, item))

            # Sample slices of init tau for each ncell
            for ncell_i, ncell in enumerate(tqdm(ncell_vec, disable=(not progress))):
                slice_left_max = slice_right_max - ncell #slice needs to fit
                num_samples = len(slice_left_all)

                # Get size and position of sample chunk for this rank
                sub_samples = num_samples // num_procs
                num_larger_procs = num_samples - num_procs*sub_samples
                if my_rank < num_larger_procs:
                    sub_samples = sub_samples + 1
                    sub_start = my_rank * sub_samples
                elif sub_samples > 0:
                    sub_start = num_larger_procs + my_rank*sub_samples
                else:
                    sub_start = -1
                    sub_samples = 0

                # Initialize data for this rank's chunk of samples
                sub_tau_means = np.ones(sub_samples) * np.nan
                sub_metric_expect = np.ones(sub_samples) * np.nan
                sub_cellcounts = np.zeros(maps_filt.shape)
                sub_celltotals = np.zeros(maps_filt.shape)
                if metric == metrics[0]:
                    sub_xs_means = np.ones(sub_samples) * np.nan

                # Add one sample for computing the no change scenario
                if (ncell==max(ncell_vec)) and (my_rank==0):
                    sub_samples += 1
                    print(f"adding nochange to rank {my_rank} for {sub_samples} total iterations at ncell {ncell}")

                # Loop over sampled realizations of this fire alteration strategy
                for sub_sample_i, slice_left_sample_i in enumerate(tqdm(range(sub_start, sub_start+sub_samples), disable=True)):#(not progress))):
                    tau_expected = tau_flat + delta_tau
                    # Adjust the tau distribution at cells in slice
                    if sub_sample_i < len(sub_tau_means): #Skip if computing no change scenario
                        # Get left bound index of slice for this rank
                        slice_left = slice_left_all[slice_left_sample_i]
                        # First, check that slice is within allowed range
                        if slice_left > slice_left_max: continue
                        # If so, get full range of indices from reference
                        slice_indices = tau_argsort_ref[slice_left:slice_left + ncell]
                        tau_slice = tau_expected[slice_indices]
                        # Store mean value of slice
                        sub_tau_means[slice_left_sample_i-sub_start] = np.mean(tau_slice)
                        # Set max tauc per cell
                        final_max_tauc = final_max_tau - tau_slice
                        # First create array of replacement tau
                        replacement_tau = np.ones(ncell) #Initialize
                        if tauc_method == "flat":
                            #tauc = tauc_vec[ncell_i]
                            tauc = C / ncell
                            tauc_slice = np.repeat(tauc, ncell)
                        elif tauc_method == "scaledtoinit":
                            tau_max = tau_max_all[C_i, ncell_i, slice_left_sample_i]
                            tau_slice_ref = tau_flat[slice_indices]
                            '''might be worth generating these slices outside loops'''
                            tauc_slice = tau_max - tau_slice_ref
                            tauc_slice = np.where(tau_slice_ref < tau_max, tauc_slice, 0)
                            '''shouldn't also be capped by tauc_baseline?'''
                        # Find where tauc will push tau beyond max
                        xs_filt = (tauc_slice > final_max_tauc) 
                        replacement_tau[xs_filt] = final_max_tau
                        replacement_tau[xs_filt==False] = (tau_slice + tauc_slice)[xs_filt==False]
                        # Now replace them in the full array of tau
                        tau_expected[slice_indices] = replacement_tau 
                        if metric == metrics[0]:
                            # Store the mean value of excess resources, keep at nan if no excess
                            xsresources = (tauc_slice - final_max_tauc)[xs_filt]
                            if len(xsresources) > 0:
                                sub_xs_means[slice_left_sample_i-sub_start] = np.mean(xsresources)

                    # Get new probability distribution across fire return interval
                    '''cut off tau distribution at max tau we simulated, shouldn't leave this forever'''
                    #taus = tau_expected[tau_expected <= max(tau_vec)] 
                    taus = tau_expected
                    ncell_tot = len(taus)

                    # Get expected value of metric
                    metric_expect = 0
                    if metric != "Nf":
                        metric_hist = metric_data[metric]['metric_hist']
                        if metric in ['mu_s', 'lambda_s']:
                            metric_edges = metric_hist[2]
                            metric_vals = []
                            diffs = np.diff(metric_edges)
                            for edge_i, edge in enumerate(metric_edges[:-1]):
                                metric_vals.append(edge + diffs[edge_i]/2) 
                            metric_vals = np.array(metric_vals)
                        else:
                            dm = (max(metric_hist[2]) - min(metric_hist[2])) / metric_integrand_ratio
                            metric_vals = np.arange(min(metric_hist[2]), max(metric_hist[2])+dm, dm)
                    for tau_i in range(len(tau_edges) - 1):
                        '''For final bin, include all cells gte left bin edge, this will inflate the probability of the 
                           final bin for cases where delta_tau and/or tauc push tau values past final_max_tau, but 
                           hopefully that's a fine approximation for now.'''
                        # Get the expected values in this tau bin
                        if tau_i < len(tau_edges) - 2:
                            ncell_within_slice = np.count_nonzero((taus >= tau_edges[tau_i]) & (taus < tau_edges[tau_i+1]))
                        else:
                            ncell_within_slice = np.count_nonzero(taus >= tau_edges[tau_i])
                        # Can skip if zero fire probability in bin
                        if ncell_within_slice == 0: continue

                        # First get the probability of being in the tau bin
                        P_dtau = ncell_within_slice / ncell_tot

                        # Now get <metric>
                        if metric != "Nf":
                            metric_tau = metric_hist[0][tau_i]
                            P_metric_tau = scipy.stats.rv_histogram((metric_tau, metric_hist[2]), density=True)
                            m_min = metric_hist[2][min(np.nonzero(metric_tau)[0])]
                            m_max = metric_hist[2][max(np.nonzero(metric_tau)[0])]
                            metric_filt = (metric_vals >= m_min) & (metric_vals <= m_max)
                            m = metric_vals[metric_filt]
                            metric_expect_tau = np.trapz(y=P_metric_tau.pdf(m)*m, x=m)
                            metric_expect += metric_expect_tau * P_dtau
                        else:
                            sdm_slice = sdm_sorted[(taus >= tau_edges[tau_i]) & (taus < tau_edges[tau_i+1])]
                            all_r = metric_data['r']['all_metric']
                            r_slice = all_r[all_tau == tau_vec[tau_i]]
                            # Get expected value of Aeff for this slice of cells
                            Aeff_slice = sdm_slice * (A_cell * 100) #cell area converted km^2 -> Ha
                            Aeff_slice_hist = np.histogram(Aeff_slice, bins=50)
                            P_Aeff_tau = scipy.stats.rv_histogram(Aeff_slice_hist)
                            dAeff = (max(Aeff_slice_hist[1]) - min(Aeff_slice_hist[1])) / metric_integrand_ratio
                            Aeff_vals = np.arange(min(Aeff_slice_hist[1]), max(Aeff_slice_hist[1])+dAeff, dAeff)
                            Aeff_expect_tau = np.trapz(y=P_Aeff_tau.pdf(Aeff_vals)*Aeff_vals, x=Aeff_vals)
                            # Get expected value of integrand w/o Aeff (aka <Nf> @ Aeff=1)
                            Nf_slice = K_adult * (1 + r_slice)
                            hist_limit = np.quantile(Nf_slice, metric_thresh)
                            Nf_slice_hist = np.histogram(Nf_slice[Nf_slice < hist_limit], bins=50)
                            P_Nf_tau = scipy.stats.rv_histogram(Nf_slice_hist)
                            dNf = (max(Nf_slice_hist[1])-min(Nf_slice_hist[1])) / metric_integrand_ratio
                            Nf_vals = np.arange(min(Nf_slice_hist[1]), max(Nf_slice_hist[1])+dNf, dNf)
                            Nf_expect_tau = np.trapz(y=P_Nf_tau.pdf(Nf_vals)*Nf_vals, x=Nf_vals)
                            # Combine everything to update expected value of Nf
                            metric_expect += Nf_expect_tau * Aeff_expect_tau * P_dtau

                    # Add sample to list if not computing no change scenario
                    if sub_sample_i < len(sub_tau_means):
                        if metric_expect < 0.74903204:
                            print(f"\ntauc_slice at (C,ncell,sl_i)=({C,ncell,slice_left_sample_i}) is {tauc_slice}\ntau_slice_ref is {tau_slice_ref}\ntau_slice (with delta_tau) is {tau_slice}\nreplacement_tau are {replacement_tau}\ntau_max is {tau_max}\nlast tau_edge is {tau_edges[-1]}\nmetric_expect={metric_expect}")
                        sub_metric_expect[slice_left_sample_i-sub_start] = metric_expect

                        # Also update spatial representation of metric
                        #mapindices_slice = mapindices[freq_argsort][slice_left:slice_left+ncell]
                        #map_mask = np.zeros(maps_filt.shape, dtype=bool)
                        #map_mask[mapindices_slice[:,0], mapindices_slice[:,1]] = True
                        #sub_cellcounts[map_mask] += 1
                        #sub_celltotals[map_mask] += metric_expect
                    # Otherwise save no change scenario to file
                    elif my_rank == 0:
                        print(f"Not adding sample with index {slice_left_sample_i} / {slice_left_sample_i-sub_start} on rank {my_rank}, instead saving as nochange")
                        metric_nochange = metric_expect
                        fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/deltatau_{delta_tau}/nochange_{tauc_method}.json"
                        with open(fn, "w") as handle:
                            json.dump({f'{metric}_expect_nochange': metric_nochange}, handle)
                        if not os.path.isdir(figs_root + f"/C_{C}"):
                            os.makedirs(figs_root + f"/C_{C}")

                # Collect data across ranks
                # Initialize data to store sample means across all ranks
                sendcounts = np.array(comm_world.gather(len(sub_tau_means), root=0))
                if my_rank == 0:
                    sampled_tau_means = np.empty(sum(sendcounts))
                    sampled_metric_expect = np.empty(sum(sendcounts))        
                    if metric == metrics[0]:
                        sampled_xs_means = np.ones(sum(sendcounts)) * np.nan
                else:
                    sampled_tau_means = None
                    sampled_metric_expect = None
                    if metric == metrics[0]:
                        sampled_xs_means = None
                # Now gather data
                comm_world.Gatherv(sub_tau_means, sampled_tau_means, root=0)
                comm_world.Gatherv(sub_metric_expect, sampled_metric_expect, root=0)
                if metric == metrics[0]:
                    comm_world.Gatherv(sub_xs_means, sampled_xs_means, root=0)
                cellcounts = np.zeros(maps_filt.shape)
                comm_world.Allreduce(sub_cellcounts, cellcounts, op=MPI.SUM)
                celltotals = np.zeros(maps_filt.shape)
                comm_world.Allreduce(sub_celltotals, celltotals, op=MPI.SUM)

                if my_rank == 0:
                    # Save data to full phase matrix
                    full_phase[delta_tau_i, C_i, ncell_i, :] = sampled_metric_expect
                    # Bin results into final phase matricies
                    for tau_i, tau_left in enumerate(tau_bin_edges):
                        #tau_filt = (sampled_tau_means > tau_left) & (sampled_tau_means < tau_left+tau_bw)
                        tau_filt = (tau_means_ref[ncell_i] > tau_left) & (tau_means_ref[ncell_i] < tau_left+tau_bw)
                        metric_expect_slice = sampled_metric_expect[tau_filt]
                        if len(metric_expect_slice) > 0:
                            phase_space[len(tau_bin_edges)-1-tau_i, ncell_i] = np.mean(metric_expect_slice)
                        # Populate excess resources if on first metric
                        if metric == metrics[0]:
                            xs_means_slice = sampled_xs_means[tau_filt]
                            if not np.all(np.isnan(xs_means_slice)):
                                phase_space_xs[len(tau_bin_edges)-1-tau_i, ncell_i] = np.nanmean(xs_means_slice)   
                    # Save spatial representation for this ncell value
                    # <metric> per cell, assume at baseline if cell  
                    #metric_map = celltotals / cellcounts
                    ## Wherever cellcounts==0 -> nan, now replace within habitat cells to no_change
                    #metric_map[maps_filt & np.isnan(metric_map)] = metric_nochange
                    #fn = data_root + f"/map_ncell_{ncell}"
                    #np.save(fn, metric_map)

            if my_rank == 0:
                elapsed = timeit.default_timer() - start_time
                print('{} seconds to run metric {}'.format(elapsed, metric))
                # Save phase mats to files
                phase_fn = data_root + f"/phase_{tauc_method}.npy"
                with open(phase_fn, 'wb') as handle:
                    np.save(handle, phase_space)
                # Plot phase
                phase_fig_fn = figs_root + f"/C_{C}" + f"/phase_{tauc_method}.png"
                if tauc_method == "flat":
                    tauc_vec = C / ncell_vec
                    plot_phase(phase_space, metric, metric_nochange, tau_bin_cntrs, ncell_vec, phase_fig_fn, tauc_vec)
                else:
                    plot_phase(phase_space, metric, metric_nochange, tau_bin_cntrs, ncell_vec, phase_fig_fn)
                if metric == metrics[0]:
                    phase_fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/deltatau_{delta_tau}/C_{C}/phase_xs_{tauc_method}.npy"
                    with open(phase_fn, 'wb') as handle:
                        np.save(handle, phase_space_xs)
                    phase_fig_fn = figs_root + f"/C_{C}" + f"/phase_xs_{tauc_method}.png"
                    if tauc_method == "flat":
                        tauc_vec = C / ncell_vec
                        plot_phase(phase_space_xs, 'xs', 0, tau_bin_cntrs, ncell_vec, phase_fig_fn, tauc_vec)
                    else:
                        plot_phase(phase_space_xs, 'xs', 0, tau_bin_cntrs, ncell_vec, phase_fig_fn)

                # Plot geographical representations
                # First, get the global max across ncell values for the colorbar limit
                #vmaxes = []
                #for ncell in ncell_vec:
                #    metric_map = np.load(data_root + f"/map_ncell_{ncell}.npy")
                #    vmaxes.append(np.max(metric_map[np.isnan(metric_map) == False]))
                ## Now actually plot
                #for ncell in ncell_vec:
                #    fig, ax = plt.subplots(figsize=(12,12))
                #    cmap = copy.copy(matplotlib.cm.plasma)
                #    cmap.set_bad(alpha=0)
                #    metric_map = np.load(data_root + f"/map_ncell_{ncell}.npy")
                #    im = ax.imshow(metric_map, vmin=metric_nochange, vmax=max(vmaxes), cmap=cmap)
                #    cbar = ax.figure.colorbar(im, ax=ax, location="right", shrink=0.6)
                #    cbar.ax.set_ylabel(r'$<{}>$'.format(metric), rotation=-90, fontsize=10, labelpad=20)
                #    fig_fn = figs_root + f"/C_{C}/map_ncell_{ncell}.png"
                #    fig.savefig(fig_fn, bbox_inches='tight')
                #    plt.close(fig)
if my_rank == 0:
    fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/full_phase_{tauc_method}.npy"
    np.save(fn, full_phase)
