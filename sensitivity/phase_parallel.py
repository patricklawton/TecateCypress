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

# Some constants
c = 1.42
Aeff = 2.38 #7.29
t_final = 400
r_bw=0.1
ul_coord = [1500, 2800]
lr_coord = [2723, 3905]
max_fri = 66
A_cell = 270**2 / 1e6 #km^2
fif_baseline = 1
dr = 0.01
dfri = 0.01

# Init data to be read on rank 0
all_fri = None
all_r = None
fri_edges = None
fri_vec = None
hist = None
K_adult = None
fri_flat = None
sdm_flat = None

# Handle data reading on rank 0 alone
if my_rank == 0:
    project = sg.get_project()
    with sg.H5Store('shared_data.h5').open(mode='r') as sd:
        b_vec = np.array(sd['b_vec'])
    fri_vec = b_vec * gamma(1+1/c)

    jobs = project.find_jobs({'doc.simulated': True, 'Aeff': Aeff, 't_final': t_final})
    #jobs = np.ones(500)
    all_fri = np.tile(fri_vec, len(jobs))
    if not os.path.isdir('aggregate_data/Aeff_{}'.format(Aeff)):
        os.makedirs('aggregate_data/Aeff_{}'.format(Aeff))
    np.save(f"aggregate_data/Aeff_{Aeff}/all_fri_{t_final}.npy", all_fri)
    fn = "aggregate_data/Aeff_{}/all_r_{}.npy".format(Aeff, t_final)
    if not os.path.isfile(fn):
        all_r = np.array([])
        for job_i, job in enumerate(jobs):
            with job.data as data:
                frac_change_vec = []
                for b in b_vec:
                    frac_change_vec.append(float(data['fractional_change/{}'.format(b)]))
            all_r = np.append(all_r, frac_change_vec)
        with open(fn, 'wb') as handle:
            np.save(handle, all_r)
    else:
        with open(fn, 'rb') as handle:
            all_r = np.load(handle)
            
    r_edges = np.arange(-1, 6+r_bw, r_bw)
    fri_step = (b_vec[1]-b_vec[0]) * gamma(1+1/c)
    fri_edges = np.concatenate(([0], np.arange(fri_step/2, fri_vec[-1]+fri_step, fri_step)))
    fig, ax = plt.subplots(figsize=(13,8))
    hist = ax.hist2d(all_fri, all_r, bins=[fri_edges, r_edges], 
                     norm=matplotlib.colors.LogNorm(vmax=int(len(all_r)/len(b_vec))))
    cbar = ax.figure.colorbar(hist[-1], ax=ax, location="right")
    cbar.ax.set_ylabel('demographic robustness', rotation=-90, fontsize=10, labelpad=20)
    ax.set_xlabel('<FRI>')
    ax.set_ylabel(r'$\frac{\Delta \text{N}}{\text{N}_1(0)}$')
    if not os.path.isdir('figs/Aeff_{}'.format(Aeff)):
        os.makedirs('figs/Aeff_{}'.format(Aeff))
    fig.savefig('figs/Aeff_{}/sensitvity_tfinal_{}.png'.format(Aeff, t_final), bbox_inches='tight')

    # Read in FDM
    usecols = np.arange(ul_coord[0],lr_coord[0])
    fdmfn = '../shared_maps/FDE_current_allregions.asc'
    if fdmfn[-3:] == 'txt':
        fdm = np.loadtxt(fdmfn)
    else:
        # Assume these are uncropped .asc maps
        # fdm = np.loadtxt(fdmfn, skiprows=6)
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
    #fri_sub = fri_raster[(patchmap > 0) & (fdm > 0)] # Why are there any zeros in FDM at all?
    fri_sub = fri_raster[(sdm > 0) & (fdm > 0)] # Why are there any zeros in FDM at all?
    fri_flat = fri_sub.flatten()
    #sdm_sub = sdm[(patchmap > 0) & (fdm > 0)]
    sdm_sub = sdm[(sdm > 0) & (fdm > 0)]
    sdm_flat = sdm_sub.flatten()
    sdm_flat = sdm_flat[fri_flat < max(fri_vec)]

    with open("../model_fitting/mortality/map.json", "r") as handle:
        mort_params = json.load(handle)
    K_adult = mort_params['K_adult']
all_fri = comm_world.bcast(all_fri)
all_r = comm_world.bcast(all_r)
fri_edges = comm_world.bcast(fri_edges)
fri_vec = comm_world.bcast(fri_vec)
hist = comm_world.bcast(hist)
K_adult = comm_world.bcast(K_adult)
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
n_cell_vec = np.arange(n_cell_baseline, len(fire_freqs)-slice_left_min, 3000)
#n_cell_vec = np.arange(100, 300, 100)
fif_vec = np.array([constraint/n_cell for n_cell in n_cell_vec])

# Save a few more things
if my_rank == 0:
    #np.save(f"aggregate_data/fire_freqs_sorted.npy", fire_freqs_sorted)
    np.save(f"aggregate_data/freq_bin_cntrs.npy", freq_bin_cntrs)
    np.save(f"aggregate_data/Aeff_{Aeff}/n_cell_vec_{constraint}.npy", n_cell_vec)

# Initialize final data matricies
phase_space_r = np.zeros((len(freq_bin_edges), len(fif_vec)))
phase_space_Nf = np.zeros((len(freq_bin_edges), len(fif_vec)))
for fif in tqdm(fif_vec):
    fif_i = np.nonzero(fif_vec == fif)[0][0]
    # Sample randomly placed slices of the fire frequency distribution

    # Set number of slices to generate for fire freq slices of this size
    n_cell = n_cell_vec[np.nonzero(fif_vec == fif)[0][0]]
    slice_left_max = len(fire_freqs) - n_cell - 1 #slice needs to fit
    num_samples = round((slice_left_max-slice_left_min)/200)

    # Initialize data to store sample means across all ranks
    sampled_freq_means = None
    sampled_r_expect = None
    sampled_Nf_expect = None
    if my_rank == 0:
        sampled_freq_means = np.empty(num_samples)
        sampled_r_expect = np.empty(num_samples) #np.ones(num_samples)*1000
        sampled_Nf_expect = np.empty(num_samples)        

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

    # Initialize data for this rank's chunk of samples
    sub_freq_means = np.empty(sub_samples)
    sub_r_expect = np.ones(sub_samples)*float(1e200) #np.empty(sub_samples)
    sub_Nf_expect = np.empty(sub_samples)

    # Add one sample for computing the no change scenario
    if (fif==max(fif_vec)) and (my_rank==0):
        sub_samples += 1
        print(f"adding nochange to rank {my_rank} for {sub_samples} total iterations at fif {fif}")

    for sub_sample_i, fire_sample_i in enumerate(tqdm(range(sub_start, sub_start+sub_samples))):
        # Re-sort fire frequencies and get slice
        fire_freqs_sorted = np.array(sorted(fire_freqs))
        freq_left = np.random.uniform(fire_freqs_sorted[slice_left_min], fire_freqs_sorted[slice_left_max])
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
        r_expect = 0
        r_vals = np.arange(min(hist[2]), max(hist[2])+dr, dr) #for the integrand
        Nf_expect = 0
        for fri_i in range(len(hist[0])):
            # Get the expected values in this fri bin
            fri_slice = fris[(fris >= fri_edges[fri_i]) & (fris < fri_edges[fri_i+1])]
            if len(fri_slice) == 0: continue #can skip if zero fire probability in bin

            # First get the probability of being in the fri bin
            fri_vals = np.arange(fri_edges[fri_i], fri_edges[fri_i+1], dfri)
            P_dfri = np.trapz(y=P_fri_x0.pdf(fri_vals), x=fri_vals)

            # New get <r>
            P_r_fri = scipy.stats.rv_histogram((hist[0][fri_i], hist[2]))
            r_expect_fri = np.trapz(y=P_r_fri.pdf(r_vals)*r_vals, x=r_vals)
            r_expect += r_expect_fri * P_dfri

            # Now get <Nf>
            sdm_slice = sdm_sorted[(fris >= fri_edges[fri_i]) & (fris < fri_edges[fri_i+1])]
            r_slice = all_r[all_fri == fri_vec[fri_i]]
            Nf_slice_agg = (sdm_slice[...,None] * np.tile(K_adult*(1 + r_slice), (len(sdm_slice), 1))).flatten()
            hist_limit = np.quantile(Nf_slice_agg, 0.965)
            Nf_slice_hist = np.histogram(Nf_slice_agg[Nf_slice_agg < hist_limit], bins=50)
            P_Nf_fri = scipy.stats.rv_histogram((Nf_slice_hist[0], Nf_slice_hist[1]))
            dNf = (max(Nf_slice_hist[0])-min(Nf_slice_hist[0]))/1_000
            Nf_vals = np.arange(min(Nf_slice_hist[0]), max(Nf_slice_hist[0])+dNf, dNf)
            Nf_expect_fri = np.trapz(y=P_Nf_fri.pdf(Nf_vals)*Nf_vals, x=Nf_vals)
            Nf_expect += Nf_expect_fri * P_dfri
        # Add sample to list if not computing no change scenario
        if sub_sample_i < len(sub_freq_means):
            sub_r_expect[fire_sample_i-sub_start] = r_expect
            sub_Nf_expect[fire_sample_i-sub_start] = Nf_expect
        # Otherwise save no change scenario to file
        elif my_rank == 0:
            print(f"Not adding sample with index {fire_sample_i} / {fire_sample_i-sub_start} on rank {my_rank}, instead saving as nochange")
            with open("aggregate_data/Aeff_{}/r_expect_nochange_{}.json".format(Aeff, t_final), "w") as handle:
                json.dump({'r_expect_nochange': r_expect}, handle)
            with open("aggregate_data/Aeff_{}/Nf_expect_nochange_{}.json".format(Aeff, t_final), "w") as handle:
                json.dump({'Nf_expect_nochange': Nf_expect}, handle)
    # Collect data across ranks
    comm_world.Gatherv(sub_freq_means, sampled_freq_means, root=0)
    comm_world.Gatherv(sub_r_expect, sampled_r_expect, root=0)
    comm_world.Gatherv(sub_Nf_expect, sampled_Nf_expect, root=0)

    if my_rank == 0:
        # Bin results into final phase matricies
        fif_i = np.nonzero(fif_vec == fif)[0][0]
        for freq_i, freq_left in enumerate(freq_bin_edges):
            freq_filt = (sampled_freq_means > freq_left) & (sampled_freq_means < freq_left+freq_bw)
            r_expect_slice = sampled_r_expect[freq_filt]
            Nf_expect_slice = sampled_Nf_expect[freq_filt]
            if len(r_expect_slice) > 0:
                phase_space_r[len(freq_bin_edges)-1-freq_i, fif_i] = np.mean(r_expect_slice)
                if np.mean(r_expect_slice) > 0.4:
                    print(f"mean r_expect_slice of {np.mean(r_expect_slice)} at fif of {fif}")
                    print(f"r_expect_slice contains {r_expect_slice}\n")
            if len(Nf_expect_slice) > 0:
                phase_space_Nf[len(freq_bin_edges)-1-freq_i, fif_i] = np.mean(Nf_expect_slice)

if my_rank == 0:
    # Save phase mats to files
    if not os.path.isdir('phase_mats/Aeff_{}'.format(Aeff)):
        os.makedirs('phase_mats/Aeff_{}'.format(Aeff))
    phase_fn = 'phase_mats/Aeff_{}/phase_r_{}_{}.npy'.format(Aeff, round(constraint), t_final)
    with open(phase_fn, 'wb') as handle:
        np.save(handle, phase_space_r)
    phase_fn = 'phase_mats/Aeff_{}/phase_Nf_{}_{}.npy'.format(Aeff, round(constraint), t_final)
    with open(phase_fn, 'wb') as handle:
        np.save(handle, phase_space_Nf)
