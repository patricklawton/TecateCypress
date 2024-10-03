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
fric_method = "scaledtoinit" #"flat"
metrics = ['lambda_s']#['mu_s']#['r', 'Nf', 'g']
metric_thresh = 0.98
metric_bw_ratio = 50
c = 1.42
Aeff = 7.29
t_final = 600
sim_method = 'nint'
ul_coord = [1500, 2800]
lr_coord = [2723, 3905]
with sg.H5Store('shared_data.h5').open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])
fri_vec = b_vec * gamma(1+1/c)
#final_max_fri = 66 #yrs
final_max_fri = max(fri_vec)
min_fri = 3
A_cell = 270**2 / 1e6 #km^2
fri_bw_ratio = 50 #For binning initial fri (with uncertainty)
fric_baseline = 200 #years, max fire return interval change possible
                    #at lowest ncell for a given C value
metric_integrand_ratio = 800
dfri = 0.01
ncell_step = 6_500#5_000#3_000
slice_spacing = 1_000#500
baseline_areas = np.arange(10, 160, 30) #km^2
delta_fri_step = 0.25
#delta_fri_sys = np.arange(-10, 10+delta_fri_step, delta_fri_step).astype(float) #years
delta_fri_sys = np.array([-10.0])
rng = np.random.default_rng()

# Generate resource allocation values
ncell_baseline_max = round(max(baseline_areas)/A_cell)
ncell_baseline_vec = np.round(baseline_areas / A_cell).astype(int) 
constraint_vec = ncell_baseline_vec * fric_baseline

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
root = 0
num_procs = comm_world.Get_size()

# Init data to be read on rank 0
metric_data = None
all_fri = None
fri_edges = None
K_adult = None
fri_flat = None
sdm_flat = None
maps_filt = None
mapindices = None

# Handle data reading on rank 0 alone
if my_rank == 0:
    project = sg.get_project()
    fri_step = (b_vec[1]-b_vec[0]) * gamma(1+1/c)
    fri_edges = np.concatenate(([0], np.arange(fri_step/2, fri_vec[-1]+fri_step, fri_step)))

    jobs = project.find_jobs({'doc.simulated': True, 'Aeff': Aeff, 't_final': t_final, 'method': sim_method})
    data_root = f"data/Aeff_{Aeff}/tfinal_{t_final}"
    figs_root = f"figs/Aeff_{Aeff}/tfinal_{t_final}"
    if not os.path.isdir(data_root):
        os.makedirs(data_root)
    fn = data_root + "/all_fri.npy"
    if (not os.path.isfile(fn)) or overwrite_metrics:
        all_fri = np.tile(fri_vec, len(jobs))
        np.save(fn, all_fri)
    else:
        all_fri = np.load(fn)

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
                
            metric_min, metric_max = (np.quantile(all_metric, 1-metric_thresh), np.quantile(all_metric, metric_thresh))
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
                metric_bw = (metric_max - metric_min) / metric_bw_ratio
                metric_edges = np.arange(metric_min, metric_max + metric_bw, metric_bw)
            
            # First plot the metric probability density
            fig, ax = plt.subplots(figsize=(13,8))
            metric_hist = ax.hist2d(all_fri, all_metric, bins=[fri_edges, metric_edges], 
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
            metric_hist = np.histogram2d(all_fri, all_metric, bins=[fri_edges, metric_edges], 
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
    fri_raster = b_raster * gamma(1+1/c)

    # Flatten and filter FDM & SDM
    # Ignore fri above what we simulated, only a small amount
    # Why are there any zeros in FDM at all?
    maps_filt = (sdm > 0) & (fdm > 0) & (fri_raster < max(fri_vec))
    mapindices = np.argwhere(maps_filt) #2d raster indicies to reference later 
    fri_flat = fri_raster[maps_filt]
    sdm_flat = sdm[maps_filt]

    # Read in K_adult (fixed, not inferred) from mortality parameters
    with open("../model_fitting/mortality/map.json", "r") as handle:
        mort_params = json.load(handle)
    K_adult = mort_params['K_adult']
# Broadcast some data to all ranks
metric_data = comm_world.bcast(metric_data)
K_adult = comm_world.bcast(K_adult)
all_fri = comm_world.bcast(all_fri)
fri_edges = comm_world.bcast(fri_edges)
fri_flat = comm_world.bcast(fri_flat)
sdm_flat = comm_world.bcast(sdm_flat)
maps_filt = comm_world.bcast(maps_filt)
mapindices = comm_world.bcast(mapindices)

# Generate samples of remaining state variables
# Use the max post alteration fri to get an upper bound on right hand of initial fri slices
fri_argsort_ref = np.argsort(fri_flat)
fri_sorted = fri_flat[fri_argsort_ref] 
if max(fri_sorted) > final_max_fri:
    slice_right_max = min(np.nonzero(fri_sorted >= final_max_fri)[0])
else:
    slice_right_max = len(fri_sorted) - 1
# Min left bound set by user-defined constant
slice_left_min = np.nonzero(fri_sorted > min_fri)[0][0]
# Generate slice sizes of the fri distribution
'''add step to endpoint? s.t. max possible ncell is included'''
ncell_vec = np.arange(ncell_baseline_max, slice_right_max, ncell_step)
# Max left bound set by smallest slice size
slice_left_max = slice_right_max - min(ncell_vec)
# Generate indices of slice left bounds, 
# reference back to fri_argsort_ref for full slice indices
slice_left_all = np.arange(slice_left_min, slice_left_max, slice_spacing)

# Get bins of initial fri for phase data
if max(fri_sorted) > final_max_fri:
    slice_right_max = min(np.nonzero(fri_sorted > final_max_fri)[0])
else:
    slice_right_max = len(fri_sorted) - 1
fri_range = fri_sorted[slice_right_max] - fri_sorted[0]
fri_bw = fri_range / fri_bw_ratio
#fri_bw = 0.75
fri_bin_edges = np.arange(fri_sorted[0], fri_sorted[slice_right_max], fri_bw)
fri_bin_cntrs = np.array([edge + fri_bw/2 for edge in fri_bin_edges])
# Store the mean fri in each preset slice for reference later
fri_means_ref = []
for ncell in ncell_vec:
    fri_means_ncell = np.ones(len(slice_left_all)) * np.nan
    slice_left_max = slice_right_max - ncell
    for slice_i, slice_left in enumerate(slice_left_all):
        if slice_left <= slice_left_max:
            fri_means_ncell[slice_i] = np.mean(fri_sorted[slice_left:slice_left+ncell])
    fri_means_ref.append(fri_means_ncell)

if fric_method == "scaledtoinit":
    # Solve for tau_max at every (C, ncell, slice_left) for delta_fri=0, then reuse for delta_fri > 0
    if my_rank != root:
        fri_max_all = None
    else:
        fri_max_all = np.ones((len(constraint_vec), len(ncell_vec), len(slice_left_all))) * np.nan
        for constraint_i, constraint in enumerate(constraint_vec):
            for ncell_i, ncell in enumerate(ncell_vec):
                slice_left_max = slice_right_max - ncell #slice needs to fit
                for sl_i, slice_left in enumerate(slice_left_all):
                    if slice_left > slice_left_max: continue
                    fri_slice = fri_sorted[slice_left:slice_left+ncell]
                    def C_diff(fri_max):
                        # Linear decrease from fri_min to fri_max
                        fric_slice = fri_max - fri_slice
                        # No negative values allowed, cut off fric at zero
                        fric_slice = np.where(fri_slice < fri_max, fric_slice, 0)
                        return np.abs(constraint - np.sum(fric_slice))
                    diffmin = scipy.optimize.minimize_scalar(C_diff, bounds=(constraint/ncell, 1e5))
                    fri_max_all[constraint_i, ncell_i, sl_i] = diffmin.x
        # Save fri_max data
        np.save(data_root + "/fri_max_all.npy", fri_max_all)
    fri_max_all = comm_world.bcast(fri_max_all, root=0)

if my_rank == 0:
    # Save all state variables
    np.save(data_root + "/delta_fri_vec.npy", delta_fri_sys)
    np.save(data_root + "/constraint_vec.npy", constraint_vec)
    np.save(data_root + "/ncell_vec.npy", ncell_vec)
    np.save(data_root + "/slice_left_all.npy", slice_left_all)

    # Save fri bin centers for plotting
    np.save(data_root + "/fri_bin_cntrs.npy", fri_bin_cntrs)

    # Initialize data for max(<metric>) across (constraint, ncell, delta_fri) 
    delta_fri_phase = np.empty((len(baseline_areas), len(ncell_vec), len(delta_fri_sys)))
    
    # Initialize data for <metric> across (delta_fri, C, ncell, slice_left) space
    full_phase = np.ones((
                    len(delta_fri_sys), len(constraint_vec), 
                    len(ncell_vec), len(slice_left_all)
                        )) * np.nan

# Loop over considered values of fri uncertainty
for delta_fri_i, delta_fri in enumerate(delta_fri_sys): 
    if my_rank==0: print(f"delta_fri: {delta_fri}")
    ##fri_uncertain = fri_flat + rng.normal(0, delta_fri, len(fri_flat))
    ##fri_min = 5
    ##fri_uncertain = np.where(fri_uncertain > fri_min, fri_uncertain, fri_min)
    ##fire_freqs = 1 / fri_uncertain
    # Add uncertainty to <fri> values and re-sort
    fri_expected = fri_flat + delta_fri
    # Sort habitat suitability data by fri value per cell
    sdm_sorted = sdm_flat[fri_argsort_ref]

    # Loop over different resource constraint values
    for constraint_i, constraint in enumerate(constraint_vec):
        if my_rank == 0: print(f"On constraint value {constraint}")
        if fric_method == "flat":
            fric_vec = np.array([constraint/ncell for ncell in ncell_vec])
        else:
            fric_vec = None

        for metric in metrics:
            if my_rank == 0:
                start_time = timeit.default_timer()
                # Initialize final data matricies
                phase_space = np.ones((len(fri_bin_edges), len(ncell_vec))) * np.nan
                # Add a matrix for computing excess resources; only do this once
                if metric == metrics[0]:
                    phase_space_xs = np.ones((len(fri_bin_edges), len(ncell_vec))) * np.nan
                figs_root = f"figs/Aeff_{Aeff}/tfinal_{t_final}/deltafri_{delta_fri}/metric_{metric}"
                if not os.path.isdir(figs_root):
                    os.makedirs(figs_root)
                # Delete existing map figures
                if os.path.isdir(figs_root + f"/const_{constraint}"):
                    for item in os.listdir(figs_root + f"/const_{constraint}"):
                        if "map_ncell_" in item:
                            os.remove(os.path.join(figs_root + f"/const_{constraint}", item))
                data_root = f"data/Aeff_{Aeff}/tfinal_{t_final}/deltafri_{delta_fri}/const_{constraint}/metric_{metric}"
                if not os.path.isdir(data_root):
                    os.makedirs(data_root)
                # Delete existing map files
                for item in os.listdir(data_root):
                    if "map_ncell_" in item:
                        os.remove(os.path.join(data_root, item))

            # Sample slices of init fri for each ncell
            for ncell_i, ncell in enumerate(tqdm(ncell_vec, disable=(not progress))):
                ## Get subset of initial fri slices for this ncell value
                #slice_left_max = slice_right_max - ncell #slice needs to fit
                #slice_left_samples = slice_left_all[slice_left_all <= slice_left_max]
                #num_samples = len(slice_left_samples)
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
                sub_fri_means = np.ones(sub_samples) * np.nan
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
                    fri_expected = fri_flat + delta_fri
                    # Adjust the fri distribution at cells in slice
                    if sub_sample_i < len(sub_fri_means): #Skip if computing no change scenario
                        # Get left bound index of slice for this rank
                        slice_left = slice_left_all[slice_left_sample_i]
                        # First, check that slice is within allowed range
                        if slice_left > slice_left_max: continue
                        # If so, get full range of indices from reference
                        slice_indices = fri_argsort_ref[slice_left:slice_left + ncell]
                        fri_slice = fri_expected[slice_indices]
                        # Store mean value of slice
                        sub_fri_means[slice_left_sample_i-sub_start] = np.mean(fri_slice)
                        # Set max fric per cell
                        final_max_fric = final_max_fri - fri_slice
                        # First create array of replacement fri
                        replacement_fri = np.ones(ncell) #Initialize
                        if fric_method == "flat":
                            fric = fric_vec[ncell_i]
                            fric_slice = np.repeat(fric, ncell)
                        elif fric_method == "scaledtoinit":
                            fri_max = fri_max_all[constraint_i, ncell_i, slice_left_sample_i]
                            fri_slice_ref = fri_flat[slice_indices]
                            '''might be worth generating these slices outside loops'''
                            fric_slice = fri_max - fri_slice_ref
                            fric_slice = np.where(fri_slice_ref < fri_max, fric_slice, 0)
                            #print(f"\nfric_slice at (C,ncell,sl_i)=({constraint,ncell,sl_i}) is:\n{fric_slice}\nfri_slice_ref is {fri_slice_ref}\nfri_max={fri_max}")
                        # Find where fric will push fri beyond max
                        xs_filt = (fric_slice > final_max_fric) 
                        replacement_fri[xs_filt] = final_max_fri
                        replacement_fri[xs_filt==False] = (fri_slice + fric_slice)[xs_filt==False]
                        # Now replace them in the full array of fri
                        fri_expected[slice_indices] = replacement_fri 
                        if metric == metrics[0]:
                            # Store the mean value of excess resources, keep at nan if no excess
                            xsresources = (fric_slice - final_max_fric)[xs_filt]
                            if len(xsresources) > 0:
                                sub_xs_means[slice_left_sample_i-sub_start] = np.mean(xsresources)

                    # Get new probability distribution across fire return interval
                    '''cut off fri distribution at max fri we simulated, shouldn't leave this forever'''
                    fris = fri_expected[fri_expected <= max(fri_vec)] 
                    fri_hist = np.histogram(fris, bins=50, density=True);
                    P_fri_xo = scipy.stats.rv_histogram((fri_hist[0], fri_hist[1]))

                    # Get expected value of metric
                    metric_expect = 0
                    if metric != "Nf":
                        metric_hist = metric_data[metric]['metric_hist']
                        if metric in ['mu_s', 'lambda_s']:
                            metric_edges = metric_hist[2]
                            metric_vals = []
                            diffs = np.diff(metric_edges)

                            '''not it'''
                            #bw_ratio = 10
                            #for edge_i, edge in enumerate(metric_edges[:-1]):
                            #    dm = diffs[edge_i] / bw_ratio
                            #    metric_vals.append(list(np.arange(edge, metric_edges[edge_i+1]+dm, dm)))
                            #metric_vals = np.array(list(itertools.chain.from_iterable(metric_vals)))

                            for edge_i, edge in enumerate(metric_edges[:-1]):
                                metric_vals.append(edge + diffs[edge_i]/2) 
                            metric_vals = np.array(metric_vals)
                        else:
                            dm = (max(metric_hist[2]) - min(metric_hist[2])) / metric_integrand_ratio
                            metric_vals = np.arange(min(metric_hist[2]), max(metric_hist[2])+dm, dm)
                    for fri_i in range(len(fri_edges) - 1):
                        # Get the expected values in this fri bin
                        within_fri_slice = (fris >= fri_edges[fri_i]) & (fris < fri_edges[fri_i+1]) 
                        # Can skip if zero fire probability in bin
                        if np.any(within_fri_slice) == False: continue

                        # First get the probability of being in the fri bin
                        fri_vals = np.arange(fri_edges[fri_i], fri_edges[fri_i+1], dfri)
                        P_dfri = np.trapz(y=P_fri_xo.pdf(fri_vals), x=fri_vals)

                        # Now get <metric>
                        if metric != "Nf":
                            metric_fri = metric_hist[0][fri_i]
                            P_metric_fri = scipy.stats.rv_histogram((metric_fri, metric_hist[2]), density=True)
                            m_min = metric_hist[2][min(np.nonzero(metric_fri)[0])]
                            m_max = metric_hist[2][max(np.nonzero(metric_fri)[0])]
                            metric_filt = (metric_vals >= m_min) & (metric_vals <= m_max)
                            m = metric_vals[metric_filt]
                            metric_expect_fri = np.trapz(y=P_metric_fri.pdf(m)*m, x=m)
                            metric_expect += metric_expect_fri * P_dfri
                        else:
                            sdm_slice = sdm_sorted[(fris >= fri_edges[fri_i]) & (fris < fri_edges[fri_i+1])]
                            all_r = metric_data['r']['all_metric']
                            r_slice = all_r[all_fri == fri_vec[fri_i]]
                            # Get expected value of Aeff for this slice of cells
                            Aeff_slice = sdm_slice * (A_cell * 100) #cell area converted km^2 -> Ha
                            Aeff_slice_hist = np.histogram(Aeff_slice, bins=50)
                            P_Aeff_fri = scipy.stats.rv_histogram(Aeff_slice_hist)
                            dAeff = (max(Aeff_slice_hist[1]) - min(Aeff_slice_hist[1])) / metric_integrand_ratio
                            Aeff_vals = np.arange(min(Aeff_slice_hist[1]), max(Aeff_slice_hist[1])+dAeff, dAeff)
                            Aeff_expect_fri = np.trapz(y=P_Aeff_fri.pdf(Aeff_vals)*Aeff_vals, x=Aeff_vals)
                            # Get expected value of integrand w/o Aeff (aka <Nf> @ Aeff=1)
                            Nf_slice = K_adult * (1 + r_slice)
                            hist_limit = np.quantile(Nf_slice, metric_thresh)
                            Nf_slice_hist = np.histogram(Nf_slice[Nf_slice < hist_limit], bins=50)
                            P_Nf_fri = scipy.stats.rv_histogram(Nf_slice_hist)
                            dNf = (max(Nf_slice_hist[1])-min(Nf_slice_hist[1])) / metric_integrand_ratio
                            Nf_vals = np.arange(min(Nf_slice_hist[1]), max(Nf_slice_hist[1])+dNf, dNf)
                            Nf_expect_fri = np.trapz(y=P_Nf_fri.pdf(Nf_vals)*Nf_vals, x=Nf_vals)
                            # Combine everything to update expected value of Nf
                            metric_expect += Nf_expect_fri * Aeff_expect_fri * P_dfri

                    # Add sample to list if not computing no change scenario
                    if sub_sample_i < len(sub_fri_means):
                        if metric_expect < 0.45:
                            print(f"\nfric_slice at (C,ncell,sl_i)=({constraint,ncell,slice_left_sample_i}) is {fric_slice}\nfri_slice_ref is {fri_slice_ref}\nfri_slice (with delta_fri) is {fri_slice}\nfri_max is {fri_max}")
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
                        with open(data_root + f"/nochange_{fric_method}.json", "w") as handle:
                            json.dump({f'{metric}_expect_nochange': metric_nochange}, handle)
                        if not os.path.isdir(figs_root + f"/const_{constraint}"):
                            os.makedirs(figs_root + f"/const_{constraint}")

                # Collect data across ranks
                # Initialize data to store sample means across all ranks
                sendcounts = np.array(comm_world.gather(len(sub_fri_means), root=0))
                if my_rank == 0:
                    sampled_fri_means = np.empty(sum(sendcounts))
                    sampled_metric_expect = np.empty(sum(sendcounts))        
                    if metric == metrics[0]:
                        sampled_xs_means = np.ones(sum(sendcounts)) * np.nan
                else:
                    sampled_fri_means = None
                    sampled_metric_expect = None
                    if metric == metrics[0]:
                        sampled_xs_means = None
                # Now gather data
                comm_world.Gatherv(sub_fri_means, sampled_fri_means, root=0)
                comm_world.Gatherv(sub_metric_expect, sampled_metric_expect, root=0)
                if metric == metrics[0]:
                    comm_world.Gatherv(sub_xs_means, sampled_xs_means, root=0)
                cellcounts = np.zeros(maps_filt.shape)
                comm_world.Allreduce(sub_cellcounts, cellcounts, op=MPI.SUM)
                celltotals = np.zeros(maps_filt.shape)
                comm_world.Allreduce(sub_celltotals, celltotals, op=MPI.SUM)

                if my_rank == 0:
                    # Save data to full phase matrix
                    full_phase[delta_fri_i, constraint_i, ncell_i, :] = sampled_metric_expect
                    # Bin results into final phase matricies
                    for fri_i, fri_left in enumerate(fri_bin_edges):
                        #fri_filt = (sampled_fri_means > fri_left) & (sampled_fri_means < fri_left+fri_bw)
                        fri_filt = (fri_means_ref[ncell_i] > fri_left) & (fri_means_ref[ncell_i] < fri_left+fri_bw)
                        metric_expect_slice = sampled_metric_expect[fri_filt]
                        if len(metric_expect_slice) > 0:
                            phase_space[len(fri_bin_edges)-1-fri_i, ncell_i] = np.mean(metric_expect_slice)
                        # Populate excess resources if on first metric
                        if metric == metrics[0]:
                            xs_means_slice = sampled_xs_means[fri_filt]
                            if not np.all(np.isnan(xs_means_slice)):
                                phase_space_xs[len(fri_bin_edges)-1-fri_i, ncell_i] = np.nanmean(xs_means_slice)   
                    # Also store data on max(<metric>) at (constraint, ncell, delta_fri)
                    phase_slice = phase_space[:, ncell_i]
                    if np.any(np.isnan(phase_slice) == False):
                        delta_fri_phase[constraint_i, ncell_i, delta_fri_i] = np.nanmax(phase_slice)
                    else:
                        delta_fri_phase[constraint_i, ncell_i, delta_fri_i] = np.nan
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
                phase_fn = data_root + f"/phase_{fric_method}.npy"
                with open(phase_fn, 'wb') as handle:
                    np.save(handle, phase_space)
                # Plot phase
                phase_fig_fn = figs_root + f"/const_{constraint}" + f"/phase_{fric_method}.png"
                plot_phase(phase_space, metric, metric_nochange, fri_bin_cntrs, ncell_vec, phase_fig_fn, fric_vec)
                if metric == metrics[0]:
                    phase_fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/deltafri_{delta_fri}/const_{constraint}/phase_xs_{fric_method}.npy"
                    with open(phase_fn, 'wb') as handle:
                        np.save(handle, phase_space_xs)
                    phase_fig_fn = figs_root + f"/const_{constraint}" + f"/phase_xs_{fric_method}.png"
                    plot_phase(phase_space_xs, 'xs', 0, fri_bin_cntrs, ncell_vec, phase_fig_fn, fric_vec)

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
                #    fig_fn = figs_root + f"/const_{constraint}/map_ncell_{ncell}.png"
                #    fig.savefig(fig_fn, bbox_inches='tight')
                #    plt.close(fig)
sys.exit()
if my_rank == 0:
    fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/delta_fri_phase_{fric_method}.npy"
    np.save(fn, delta_fri_phase)
    fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/full_phase_{fric_method}.npy"
    np.save(fn, full_phase)
