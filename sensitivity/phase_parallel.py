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

# Some constants
metrics = ['mu_s']#['r', 'Nf', 'g']
overwrite_metrics = False
metric_thresh = 0.98
metric_bw_ratio = 50
c = 1.42
Aeff = 7.29
t_final = 600
sim_method = 'nint'
ul_coord = [1500, 2800]
lr_coord = [2723, 3905]
max_fri = 66 #yrs
A_cell = 270**2 / 1e6 #km^2
fif_baseline = 1
metric_integrand_ratio = 800
dfri = 0.01
n_cell_step = 5_000#3_000
num_samples_ratio = 750#500
progress = True
#baseline_area = 20 #km^2
#baseline_areas = np.arange(10, 110, 10)
baseline_areas = np.arange(10, 155, 5)
n_cell_baseline_max = round(max(baseline_areas)/A_cell)
#delta_fri_sys = np.arange(-10, 11, 1) #yrs
#delta_fri_sys = np.arange(0,11,1)
delta_fri_sys = [0]
rng = np.random.default_rng()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

# Init data to be read on rank 0
metric_data = None
all_fri = None
fri_edges = None
fri_vec = None
K_adult = None
fri_flat = None
sdm_flat = None
maps_filt = None
mapindices = None

# Handle data reading on rank 0 alone
if my_rank == 0:
    project = sg.get_project()
    with sg.H5Store('shared_data.h5').open(mode='r') as sd:
        b_vec = np.array(sd['b_vec'])
    fri_vec = b_vec * gamma(1+1/c)
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
            if metric == 'r': metric_label = 'fractional_change'
            elif metric in ['Nf', 'mu_s']: metric_label = metric
            elif metric == 'g': metric_label = 'decay_rate'
            all_metric = np.array([])
            for job_i, job in enumerate(jobs):
                with job.data as data:
                    metric_vec = []
                    for b in b_vec:
                        metric_vec.append(float(data[f'{metric_label}/{b}']))
                all_metric = np.append(all_metric, metric_vec)
                
            metric_min, metric_max = (np.quantile(all_metric, 1-metric_thresh), np.quantile(all_metric, metric_thresh))
            metric_bw = (metric_max - metric_min) / metric_bw_ratio
            #metric_bw = 0.001
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
fri_vec = comm_world.bcast(fri_vec)
fri_flat = comm_world.bcast(fri_flat)
sdm_flat = comm_world.bcast(sdm_flat)
maps_filt = comm_world.bcast(maps_filt)
mapindices = comm_world.bcast(mapindices)

# Generate resource allocation scenarios shared across delta_fri
# Convert fri to freq, with added uncertainty on fri
fire_freqs = 1 / (fri_flat + max(delta_fri_sys)) #note we use the max increase in fri
#fri_max_uncertain = fri_flat + rng.normal(0, max(delta_fri_sys), len(fri_flat))
#fri_max_uncertain = np.where(fri_max_uncertain > 1, fri_max_uncertain, 1)
#fire_freqs = 1 / fri_max_uncertain
# Sort fire frequency 
freq_argsort = np.argsort(fire_freqs)
fire_freqs_sorted = fire_freqs[freq_argsort]
slice_left_min = np.nonzero(fire_freqs_sorted > (1/max_fri))[0][0]
# Generate resource allocation scenarios
n_cell_vec = np.arange(n_cell_baseline_max, len(fire_freqs)-slice_left_min, n_cell_step)

if my_rank == 0:
    # Save shared n_cell_vec
    np.save(data_root + "/n_cell_vec.npy", n_cell_vec)

    # Initialize data for max(<r>) across (constraint, n_cell, delta_fri) 
    delta_fri_phase = np.empty((len(baseline_areas), len(n_cell_vec), len(delta_fri_sys)))

# Loop over considered values of fire freq uncertainty
for delta_fri_i, delta_fri in enumerate(delta_fri_sys): 
    # Convert fri to freq, with added uncertainty on fri
    fire_freqs = 1 / (fri_flat + delta_fri)
    #fri_uncertain = fri_flat + rng.normal(0, delta_fri, len(fri_flat))
    #fri_min = 5
    #fri_uncertain = np.where(fri_uncertain > fri_min, fri_uncertain, fri_min)
    #fire_freqs = 1 / fri_uncertain
    # Sort fire frequency 
    freq_argsort = np.argsort(fire_freqs)
    fire_freqs_sorted = fire_freqs[freq_argsort]
    # Sort habitat suitability data
    sdm_sorted = sdm_flat[freq_argsort]

    # Get bins of initial fire frequency for phase data
    slice_left_min = np.nonzero(fire_freqs_sorted > (1/max_fri))[0][0]
    freq_range = fire_freqs_sorted[-1] - fire_freqs_sorted[slice_left_min]
    #freq_bw = freq_range/20
    freq_bw = 0.002
    freq_bin_edges = np.arange(fire_freqs_sorted[slice_left_min], fire_freqs_sorted[-1], freq_bw)
    freq_bin_cntrs = np.array([edge+freq_bw/2 for edge in freq_bin_edges])

    ## Generate resource allocation scenarios
    #n_cell_vec = np.arange(n_cell_baseline_max, len(fire_freqs)-slice_left_min, n_cell_step)

    if my_rank == 0:
        print(f"delta_fri: {delta_fri}")
        # Save some things
        data_root = f"data/Aeff_{Aeff}/tfinal_{t_final}/deltafri_{delta_fri}"
        if not os.path.isdir(data_root):
            os.makedirs(data_root)
        np.save(data_root + "/freq_bin_cntrs.npy", freq_bin_cntrs)

    # Loop over different resource constraint values
    #baseline_areas = np.array([10]) #km
    for constraint_i, baseline_area in enumerate(baseline_areas):
        # Resort fire freqs
        fire_freqs_sorted = fire_freqs[freq_argsort]
        # Get some info for this allocation scenario
        n_cell_baseline = round(baseline_area/A_cell)
        constraint = n_cell_baseline * fif_baseline
        #n_cell_vec = np.arange(100, 300, 100)
        fif_vec = np.array([constraint/n_cell for n_cell in n_cell_vec])

        # Draw all freq slice random starting points for this rank
        # Store in nested list, sublist for each fif value
        if my_rank == 0:
            start_time = timeit.default_timer()
        freq_left_samples_sub = []
        for fif_i, fif in enumerate(tqdm(fif_vec, disable=(not progress))):
            # Set number of slices to generate for fire freq slices of this size
            n_cell = n_cell_vec[np.nonzero(fif_vec == fif)[0][0]]
            slice_left_max = len(fire_freqs) - n_cell - 1 #slice needs to fit
            num_samples = round((slice_left_max-slice_left_min)/num_samples_ratio)

            # Get slice indices of samples for this rank
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

            # Add one sample for computing the no change scenario
            if (fif==max(fif_vec)) and (my_rank==0):
                sub_samples += 1

            # Draw and store freq bin edges for this fif value
            freq_left_sub_vec = np.random.uniform(fire_freqs_sorted[slice_left_min], fire_freqs_sorted[slice_left_max], size=sub_samples)
            freq_left_samples_sub.append(freq_left_sub_vec)
        if my_rank == 0:
            elapsed = timeit.default_timer() - start_time
            print('{} seconds to draw frequency samples'.format(elapsed))

        for metric in metrics:
            if my_rank == 0:
                start_time = timeit.default_timer()
                # Initialize final data matricies
                phase_space = np.zeros((len(freq_bin_edges), len(fif_vec)))
                # Add a matrix for computing excess resources; only do this once
                if metric == metrics[0]:
                    phase_space_xs = np.zeros((len(freq_bin_edges), len(fif_vec))) 
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

            # Sample random slices of init fire freqs for each intervention freq
            for fif_i, fif in enumerate(tqdm(fif_vec, disable=False)):#(not progress))):
                # Set number of slices to generate for fire freq slices of this size
                n_cell = n_cell_vec[np.nonzero(fif_vec == fif)[0][0]]
                slice_left_max = len(fire_freqs) - n_cell - 1 #slice needs to fit
                num_samples = round((slice_left_max-slice_left_min)/num_samples_ratio)

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

                # Initialize data to store sample means across all ranks
                sampled_freq_means = None
                sampled_metric_expect = None
                if metric == metrics[0]:
                    sampled_xs_means = None
                if my_rank == 0:
                    sampled_freq_means = np.empty(num_samples)
                    sampled_metric_expect = np.empty(num_samples)        
                    if metric == metrics[0]:
                        sampled_xs_means = np.ones(num_samples) * np.nan

                # Initialize data for this rank's chunk of samples
                sub_freq_means = np.empty(sub_samples)
                sub_metric_expect = np.empty(sub_samples)
                sub_cellcounts = np.zeros(maps_filt.shape)
                sub_celltotals = np.zeros(maps_filt.shape)
                if metric == metrics[0]:
                    sub_xs_means = np.ones(sub_samples) * np.nan

                # Add one sample for computing the no change scenario
                if (fif==max(fif_vec)) and (my_rank==0):
                    sub_samples += 1
                    print(f"adding nochange to rank {my_rank} for {sub_samples} total iterations at fif {fif}")

                # Loop over random realizations of this fire alteration strategy
                for sub_sample_i, fire_sample_i in enumerate(tqdm(range(sub_start, sub_start+sub_samples), disable=True)):#(not progress))):
                    # Re-sort fire frequencies and get slice for this rank
                    fire_freqs_sorted = fire_freqs[freq_argsort]
                    freq_left = freq_left_samples_sub[fif_i][sub_sample_i]
                    slice_left = np.nonzero(fire_freqs_sorted > freq_left)[0][0]
                    fire_freq_slice = fire_freqs_sorted[slice_left:slice_left+n_cell]
                    # Store mean of slice's initial fire frequency
                    # Skip if computing the no change scenario
                    if sub_sample_i < len(sub_freq_means):
                        sub_freq_means[fire_sample_i-sub_start] = np.mean(fire_freq_slice)

                    # Adjust the fire frequency distribution
                    max_fif = fire_freq_slice - fire_freqs_sorted[slice_left_min]
                    # Skip if computing the no change scenario
                    if sub_sample_i < len(sub_freq_means):
                        # First create array of replacement frequencies
                        replacement_freqs = np.ones(n_cell)
                        xsresource_filt = (fif > max_fif)
                        replacement_freqs[xsresource_filt] = (fire_freq_slice - max_fif)[xsresource_filt]
                        replacement_freqs[(fif < max_fif)] = (fire_freq_slice - fif)[(fif < max_fif)]
                        # Now replace them in the full array of frequencies
                        fire_freqs_sorted[slice_left:slice_left+n_cell] = replacement_freqs 
                        if metric == metrics[0]:
                            # Store the mean value of excess resources, keep at nan if no excess
                            xsresources = (fif - max_fif)[xsresource_filt]
                            if len(xsresources) > 0:
                                sub_xs_means[fire_sample_i-sub_start] = np.mean(xsresources)

                    # Get new probability distribution across fire return interval
                    fris = 1/fire_freqs_sorted
                    fri_hist = np.histogram(fris, bins=50, density=True);
                    P_fri_xo = scipy.stats.rv_histogram((fri_hist[0], fri_hist[1]))

                    # Get expected value of metric
                    metric_expect = 0
                    if metric != "Nf":
                        metric_hist = metric_data[metric]['metric_hist']
                        dm = (max(metric_hist[2]) - min(metric_hist[2])) / metric_integrand_ratio
                        metric_vals = np.arange(min(metric_hist[2]), max(metric_hist[2])+dm, dm)
                    for fri_i in range(len(fri_edges) - 1):
                        # Get the expected values in this fri bin
                        fri_slice = fris[(fris >= fri_edges[fri_i]) & (fris < fri_edges[fri_i+1])]
                        if len(fri_slice) == 0: continue #can skip if zero fire probability in bin

                        # First get the probability of being in the fri bin
                        fri_vals = np.arange(fri_edges[fri_i], fri_edges[fri_i+1], dfri)
                        P_dfri = np.trapz(y=P_fri_xo.pdf(fri_vals), x=fri_vals)
                        #P_dfri = P_fri_xo.expect(func=lambda x: dfri, lb=fri_edges[fri_i], ub=fri_edges[fri_i+1])

                        # Now get <metric>
                        if metric != "Nf":
                            P_metric_fri = scipy.stats.rv_histogram((metric_hist[0][fri_i], metric_hist[2]))
                            metric_expect_fri = np.trapz(y=P_metric_fri.pdf(metric_vals)*metric_vals, x=metric_vals)
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
                    if sub_sample_i < len(sub_freq_means):
                        sub_metric_expect[fire_sample_i-sub_start] = metric_expect

                        # Also update spatial representation of metric
                        #mapindices_slice = mapindices[freq_argsort][slice_left:slice_left+n_cell]
                        #map_mask = np.zeros(maps_filt.shape, dtype=bool)
                        #map_mask[mapindices_slice[:,0], mapindices_slice[:,1]] = True
                        #sub_cellcounts[map_mask] += 1
                        #sub_celltotals[map_mask] += metric_expect
                    # Otherwise save no change scenario to file
                    elif my_rank == 0:
                        print(f"Not adding sample with index {fire_sample_i} / {fire_sample_i-sub_start} on rank {my_rank}, instead saving as nochange")
                        metric_nochange = metric_expect
                        with open(data_root + "/nochange.json", "w") as handle:
                            json.dump({f'{metric}_expect_nochange': metric_nochange}, handle)
                        if not os.path.isdir(figs_root + f"/const_{constraint}"):
                            os.makedirs(figs_root + f"/const_{constraint}")

                # Collect data across ranks
                comm_world.Gatherv(sub_freq_means, sampled_freq_means, root=0)
                comm_world.Gatherv(sub_metric_expect, sampled_metric_expect, root=0)
                if metric == metrics[0]:
                    comm_world.Gatherv(sub_xs_means, sampled_xs_means, root=0)
                cellcounts = np.zeros(maps_filt.shape)
                comm_world.Allreduce(sub_cellcounts, cellcounts, op=MPI.SUM)
                celltotals = np.zeros(maps_filt.shape)
                comm_world.Allreduce(sub_celltotals, celltotals, op=MPI.SUM)

                if my_rank == 0:
                    # Bin results into final phase matricies
                    for freq_i, freq_left in enumerate(freq_bin_edges):
                        freq_filt = (sampled_freq_means > freq_left) & (sampled_freq_means < freq_left+freq_bw)
                        metric_expect_slice = sampled_metric_expect[freq_filt]
                        if len(metric_expect_slice) > 0:
                            phase_space[len(freq_bin_edges)-1-freq_i, fif_i] = np.mean(metric_expect_slice)
                        # Populate excess resources if on first metric
                        if metric == metrics[0]:
                            xs_means_slice = sampled_xs_means[freq_filt]
                            if not np.all(np.isnan(xs_means_slice)):
                                phase_space_xs[len(freq_bin_edges)-1-freq_i, fif_i] = np.nanmean(xs_means_slice)   
                    # Also store data on max(<r>) at (constraint, n_cell, delta_fri)
                    phase_slice = phase_space[:, fif_i]
                    delta_fri_phase[constraint_i, fif_i, delta_fri_i] = max(phase_slice[phase_slice != 0])

                    # Save spatial representation for this n_cell value
                    # <metric> per cell, assume at baseline if cell  
                    #metric_map = celltotals / cellcounts
                    ## Wherever cellcounts==0 -> nan, now replace within habitat cells to no_change
                    #metric_map[maps_filt & np.isnan(metric_map)] = metric_nochange
                    #fn = data_root + f"/map_ncell_{n_cell}"
                    #np.save(fn, metric_map)

            if my_rank == 0:
                elapsed = timeit.default_timer() - start_time
                print('{} seconds to run metric {}'.format(elapsed, metric))
                # Save phase mats to files
                phase_fn = data_root + "/phase.npy"
                with open(phase_fn, 'wb') as handle:
                    np.save(handle, phase_space)
                # Plot phase
                phase_fig_fn = figs_root + f"/const_{constraint}" + "/phase.png"
                plot_phase(phase_space, metric, metric_nochange, freq_bin_cntrs, n_cell_vec, phase_fig_fn, fif_vec)
                if metric == metrics[0]:
                    phase_fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/deltafri_{delta_fri}/const_{constraint}/phase_xs.npy"
                    with open(phase_fn, 'wb') as handle:
                        np.save(handle, phase_space_xs)
                    phase_fig_fn = figs_root + f"/const_{constraint}" + "/phase_xs.png"
                    plot_phase(phase_space_xs, 'xs', 0, freq_bin_cntrs, n_cell_vec, phase_fig_fn, fif_vec)

                # Plot geographical representations
                # First, get the global max across n_cell values for the colorbar limit
                #vmaxes = []
                #for n_cell in n_cell_vec:
                #    metric_map = np.load(data_root + f"/map_ncell_{n_cell}.npy")
                #    vmaxes.append(np.max(metric_map[np.isnan(metric_map) == False]))
                ## Now actually plot
                #for n_cell in n_cell_vec:
                #    fig, ax = plt.subplots(figsize=(12,12))
                #    cmap = copy.copy(matplotlib.cm.plasma)
                #    cmap.set_bad(alpha=0)
                #    metric_map = np.load(data_root + f"/map_ncell_{n_cell}.npy")
                #    im = ax.imshow(metric_map, vmin=metric_nochange, vmax=max(vmaxes), cmap=cmap)
                #    cbar = ax.figure.colorbar(im, ax=ax, location="right", shrink=0.6)
                #    cbar.ax.set_ylabel(r'$<{}>$'.format(metric), rotation=-90, fontsize=10, labelpad=20)
                #    fig_fn = figs_root + f"/const_{constraint}/map_ncell_{n_cell}.png"
                #    fig.savefig(fig_fn, bbox_inches='tight')
                #    plt.close(fig)
if my_rank == 0:
    fn = f"data/Aeff_{Aeff}/tfinal_{t_final}/delta_fri_phase.npy"
    np.save(fn, delta_fri_phase)
