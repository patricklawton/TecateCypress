import signac as sg
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy.special import gamma
import scipy
import os
import json
from tqdm.auto import tqdm
#from tqdm import tqdm
from mpi4py import MPI
import timeit
import pickle

# Some constants
metrics = ['r', 'Nf', 'g']
overwrite_metrics = True
metric_thresh = 0.98
metric_bw_ratio = 50
c = 1.42
Aeff = 7.29
t_final = 400
ul_coord = [1500, 2800]
lr_coord = [2723, 3905]
max_fri = 66
A_cell = 270**2 / 1e6 #km^2
fif_baseline = 1
metric_integrand_ratio = 800
dfri = 0.01
dNf_ratio = 1_000
n_cell_step = 3_000
num_samples_ratio = 500
progress = False

def adjustmaps(maps):
    dim_len = []
    for dim in range(2):
        dim_len.append(min([m.shape[dim] for m in maps]))
    for mi, m in enumerate(maps):
        maps[mi] = m[0:dim_len[0], 0:dim_len[1]]
    return maps

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

# Handle data reading on rank 0 alone
if my_rank == 0:
    project = sg.get_project()
    with sg.H5Store('shared_data.h5').open(mode='r') as sd:
        b_vec = np.array(sd['b_vec'])
    fri_vec = b_vec * gamma(1+1/c)
    fri_step = (b_vec[1]-b_vec[0]) * gamma(1+1/c)
    fri_edges = np.concatenate(([0], np.arange(fri_step/2, fri_vec[-1]+fri_step, fri_step)))

    jobs = project.find_jobs({'doc.simulated': True, 'Aeff': Aeff, 't_final': t_final})
    all_fri = np.tile(fri_vec, len(jobs))
    if not os.path.isdir('aggregate_data/Aeff_{}'.format(Aeff)):
        os.makedirs('aggregate_data/Aeff_{}'.format(Aeff))
    np.save(f"aggregate_data/Aeff_{Aeff}/all_fri_{t_final}.npy", all_fri)

    fn = f"aggregate_data/Aeff_{Aeff}/metric_data_{t_final}.pkl"
    if (not os.path.isfile(fn)) or overwrite_metrics:
        metric_data = {m: {} for m in metrics}
        for metric in metrics:#[m for m in metrics if m != 'Nf']:
            if metric == 'r': metric_label = 'fractional_change'
            elif metric == 'Nf': metric_label = metric
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
            metric_edges = np.arange(metric_min, metric_max + metric_bw, metric_bw)

            fig, ax = plt.subplots(figsize=(13,8))
            metric_hist = ax.hist2d(all_fri, all_metric, bins=[fri_edges, metric_edges], 
                             norm=matplotlib.colors.LogNorm(vmax=int(len(all_metric)/len(b_vec))))
            cbar = ax.figure.colorbar(metric_hist[-1], ax=ax, location="right")
            cbar.ax.set_ylabel('demographic robustness', rotation=-90, fontsize=10, labelpad=20)
            ax.set_xlabel('<FRI>')
            ax.set_ylabel(metric)
            if not os.path.isdir('figs/Aeff_{}'.format(Aeff)):
                os.makedirs('figs/Aeff_{}'.format(Aeff))
            fig.savefig('figs/Aeff_{}/sensitvity_{}_tfinal_{}.png'.format(Aeff, metric, t_final), bbox_inches='tight')

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

    # Flatten FDM & SDM
    fri_sub = fri_raster[(sdm > 0) & (fdm > 0)] # Why are there any zeros in FDM at all?
    fri_flat = fri_sub.flatten()
    sdm_sub = sdm[(sdm > 0) & (fdm > 0)]
    sdm_flat = sdm_sub.flatten()
    sdm_flat = sdm_flat[fri_flat < max(fri_vec)]

    with open("../model_fitting/mortality/map.json", "r") as handle:
        mort_params = json.load(handle)
    K_adult = mort_params['K_adult']
metric_data = comm_world.bcast(metric_data)
K_adult = comm_world.bcast(K_adult)
all_fri = comm_world.bcast(all_fri)
fri_edges = comm_world.bcast(fri_edges)
fri_vec = comm_world.bcast(fri_vec)
fri_flat = comm_world.bcast(fri_flat)
sdm_flat = comm_world.bcast(sdm_flat)

# Sort fire frequency and habitat suitability data
fire_freqs = 1 / fri_flat[fri_flat < max(fri_vec)] #ignore fri above what we simulated, only a small amount
freq_argsort = np.argsort(fire_freqs)
fire_freqs_sorted = fire_freqs[freq_argsort]
sdm_sorted = sdm_flat[freq_argsort]

# Get bins of initial fire frequency for phase data
slice_left_min = np.nonzero(fire_freqs_sorted > (1/max_fri))[0][0]
freq_range = fire_freqs_sorted[-1] - fire_freqs_sorted[slice_left_min]
freq_bw = freq_range/20
freq_bin_edges = np.arange(fire_freqs_sorted[slice_left_min], fire_freqs_sorted[-1], freq_bw)
freq_bin_cntrs = np.array([edge+freq_bw/2 for edge in freq_bin_edges])

# Loop over different resource constraint values
#baseline_areas = np.array([10]) #km
baseline_area = 10

# Generate resource allocation scenarios
n_cell_baseline = round(baseline_area/A_cell)
constraint = n_cell_baseline * fif_baseline
n_cell_vec = np.arange(n_cell_baseline, len(fire_freqs)-slice_left_min, n_cell_step)
#n_cell_vec = np.arange(100, 300, 100)
fif_vec = np.array([constraint/n_cell for n_cell in n_cell_vec])

# Save a few more things
if my_rank == 0:
    np.save(f"aggregate_data/freq_bin_cntrs.npy", freq_bin_cntrs)
    np.save(f"aggregate_data/Aeff_{Aeff}/n_cell_vec_{constraint}.npy", n_cell_vec)
    start_time = timeit.default_timer()

# Draw all freq slice random starting points for this rank
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
        print(f"adding nochange to rank {my_rank} for {sub_samples} total iterations at fif {fif}")

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

    # Sample random slices of init fire freqs for each intervention freq
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

        # Initialize data to store sample means across all ranks
        sampled_freq_means = None
        sampled_metric_expect = None
        if my_rank == 0:
            sampled_freq_means = np.empty(num_samples)
            sampled_metric_expect = np.empty(num_samples)        

        # Initialize data for this rank's chunk of samples
        sub_freq_means = np.empty(sub_samples)
        sub_metric_expect = np.empty(sub_samples)

        # Add one sample for computing the no change scenario
        if (fif==max(fif_vec)) and (my_rank==0):
            sub_samples += 1
            print(f"adding nochange to rank {my_rank} for {sub_samples} total iterations at fif {fif}")

        # Loop over random realizations of this fire alteration strategy
        for sub_sample_i, fire_sample_i in enumerate(tqdm(range(sub_start, sub_start+sub_samples), disable=(not progress))):
            # Re-sort fire frequencies and get slice for this rank
            fire_freqs_sorted = np.array(sorted(fire_freqs))
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
                fire_freqs_sorted[slice_left:slice_left+n_cell] = np.where(fif < max_fif, fire_freq_slice - fif, fire_freq_slice - max_fif)

            # Get new probability distribution across fire return interval
            fris = 1/fire_freqs_sorted
            fri_hist = np.histogram(fris, bins=50, density=True);
            P_fri_x0 = scipy.stats.rv_histogram((fri_hist[0], fri_hist[1]))

            # Get the expected values of metrics
            metric_expect = 0
            metric_hist = metric_data[metric]['metric_hist']
            dm = (max(metric_hist[2]) - min(metric_hist[2])) / metric_integrand_ratio
            metric_vals = np.arange(min(metric_hist[2]), max(metric_hist[2])+dm, dm)
            for fri_i in range(len(fri_edges) - 1):
                # Get the expected values in this fri bin
                fri_slice = fris[(fris >= fri_edges[fri_i]) & (fris < fri_edges[fri_i+1])]
                if len(fri_slice) == 0: continue #can skip if zero fire probability in bin

                # First get the probability of being in the fri bin
                fri_vals = np.arange(fri_edges[fri_i], fri_edges[fri_i+1], dfri)
                P_dfri = np.trapz(y=P_fri_x0.pdf(fri_vals), x=fri_vals)

                P_metric_fri = scipy.stats.rv_histogram((metric_hist[0][fri_i], metric_hist[2]))
                metric_expect_fri = np.trapz(y=P_metric_fri.pdf(metric_vals)*metric_vals, x=metric_vals)
                metric_expect += metric_expect_fri * P_dfri

            # Add sample to list if not computing no change scenario
            if sub_sample_i < len(sub_freq_means):
                sub_metric_expect[fire_sample_i-sub_start] = metric_expect
            # Otherwise save no change scenario to file
            elif my_rank == 0:
                print(f"Not adding sample with index {fire_sample_i} / {fire_sample_i-sub_start} on rank {my_rank}, instead saving as nochange")
                with open("aggregate_data/Aeff_{}/{}_expect_nochange_{}.json".format(Aeff, metric,  t_final), "w") as handle:
                    json.dump({f'{metric}_expect_nochange': metric_expect}, handle)
        # Collect data across ranks
        comm_world.Gatherv(sub_freq_means, sampled_freq_means, root=0)
        comm_world.Gatherv(sub_metric_expect, sampled_metric_expect, root=0)

        if my_rank == 0:
            # Bin results into final phase matricies
            fif_i = np.nonzero(fif_vec == fif)[0][0]
            for freq_i, freq_left in enumerate(freq_bin_edges):
                freq_filt = (sampled_freq_means > freq_left) & (sampled_freq_means < freq_left+freq_bw)
                metric_expect_slice = sampled_metric_expect[freq_filt]
                if len(metric_expect_slice) > 0:
                    phase_space[len(freq_bin_edges)-1-freq_i, fif_i] = np.mean(metric_expect_slice)

    if my_rank == 0:
        elapsed = timeit.default_timer() - start_time
        print('{} seconds to run metric {}'.format(elapsed, metric))
        # Save phase mats to files
        if not os.path.isdir('phase_mats/Aeff_{}'.format(Aeff)):
            os.makedirs('phase_mats/Aeff_{}'.format(Aeff))
        phase_fn = 'phase_mats/Aeff_{}/phase_{}_{}_{}.npy'.format(Aeff, metric, round(constraint), t_final)
        with open(phase_fn, 'wb') as handle:
            np.save(handle, phase_space)
