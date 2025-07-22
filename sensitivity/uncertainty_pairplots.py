import numpy as np
from matplotlib import colors, cm, rc
from matplotlib import pyplot as plt
import pickle
import signac as sg
from scipy.special import gamma
import copy as copy
from global_functions import adjustmaps
import h5py
from itertools import product, combinations

# Define/load things non-specific to a given set of results
metric = 'lambda_s'
Aeff = 7.29
t_final = 300
ncell_tot = 87_993
c = 1.42
with sg.H5Store('shared_data.h5').open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])
tau_vec = b_vec * gamma(1+1/c)
tauc_methods = ["flat"]
results_pre = 'gte_thresh'

# Update global plotting parameters
rc('axes', labelsize=21)  # Font size for x and y labels
rc('axes', titlesize=16)
rc('xtick', labelsize=19)  # Font size for x-axis tick labels
rc('ytick', labelsize=19)  # Font size for y-axis tick labels
rc('lines', markersize=15)
rc('lines', linewidth=5.5)
rc('legend', fontsize=19)
rc('font', family='sans-serif')
rc('font', serif=['Computer Modern Sans Serif'] + plt.rcParams['font.serif'])
rc('font', weight='light')
histlw = 5.5
cbar_lpad = 30
dpi = 40

# Function to read in things specific to given results as global variables
def set_globals(results_pre):
    if metric == 'lambda_s':
        globals()['metric_lab'] = r'$S$'
        globals()['rob_metric_lab'] = r'$S^*$'
        globals()['mean_metric_lab'] = r'$\bar{\lambda}(\tau)$'
    if metric == 'P_s':
        globals()['metric_lab'] = r'$S_{meta}$'
        globals()['rob_metric_lab'] = r'$\S_{meta}^*$'
        globals()['mean_metric_lab'] = r'$<P_s>$'
    if metric == 's':
        globals()['metric_lab'] = r'$s_{meta}$'
        globals()['rob_metric_lab'] = r'$\s_{meta}^*$'
        globals()['mean_metric_lab'] = r'$<s>$'
    globals()['fn_prefix'] = f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/"
    globals()['fig_prefix'] = f"{results_pre}/figs/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/"
    #globals()['fig_prefix'] = os.path.join('/','Volumes', 'Macintosh HD', 'Users', 'patrick',
    #                                       'Google Drive', 'My Drive', 'Research', 'Regan', 'Figs/')

# Read in maps and convert fdm to tau, used by multiple plots below
ul_coord = [1500, 2800]
lr_coord = [2723, 3905]
usecols = np.arange(ul_coord[0],lr_coord[0])
sdmfn = "../shared_maps/SDM_1995.asc"
sdm = np.loadtxt(sdmfn,skiprows=6+ul_coord[1],
                         max_rows=lr_coord[1], usecols=usecols)
fdmfn = '../shared_maps/FDE_current_allregions.asc'
fdm = np.loadtxt(fdmfn,skiprows=6+ul_coord[1],
                         max_rows=lr_coord[1], usecols=usecols)
sdm, fdm = adjustmaps([sdm, fdm])
delta_t = 30
b_raster = delta_t / np.power(-np.log(1-fdm), 1/c)
tau_raster = b_raster * gamma(1+1/c)
maps_filt = (sdm > 0) & (fdm > 0)
tau_flat = tau_raster[maps_filt]
mapindices = np.argwhere(maps_filt)
tau_argsort = np.argsort(tau_flat)
tau_sorted = tau_flat[tau_argsort]

# Define keys and labels for parameters
uncertain_params = ['mu_tau', 'sigm_tau', 'mu_tauc', 'sigm_tauc', 'mean_lam_diff']
param_labels = [r'$\mu_{\tau}$', r'$\sigma_{\tau}$', r'$\mu_{\hat{\tau}}$', 
                r'$\sigma_{\hat{\tau}}$', r'$<\lambda_m - \bar{\lambda}>$']

# Get all possible parameter pairs
param_pairs = [pair for pair in combinations(range(len(uncertain_params)), 2)]

# Read in data of S samples at optimal decisions, as well as some other things
set_globals(results_pre)
phase = h5py.File(fn_prefix + '/phase_optdecisions.h5', 'r')
meta_metric_nochange = float(np.load(fn_prefix + 'meta_metric_nochange.npy'))
rob_thresh_vec = np.load(fn_prefix + "rob_thresh_vec.npy")

# Read in optimal S under baseline conditions
S_opt_baseline = np.load(fn_prefix + 'S_opt_baseline.npy')
# I don't trust interpolation of robustness between Sstar values, 
# so use the closest one we sampled for interpolator fitting
'''This probably isnt necessary'''
Sstar_i = np.argmin(np.abs(rob_thresh_vec - S_opt_baseline) )
S_opt_baseline = rob_thresh_vec[Sstar_i]

# Read in optimal decision params at baseline and uncertain conditions
q_vec = np.load(fn_prefix + 'q_vec.npy')
Sstar_i_optdecisions = np.load(fn_prefix + 'Sstar_i_optdecisions.npy')
ncell_vec = np.load(fn_prefix + 'ncell_vec_optdecisions.npy')
slice_left_all = np.load(fn_prefix + 'slice_left_all_optdecisions.npy')
C_vec = np.load(fn_prefix + 'C_vec_optdecisions.npy')

# Read in all splined interpolations of metric(tau)
with open(fn_prefix + "/metric_spl_all.pkl", "rb") as handle:
    metric_spl_all = pickle.load(handle)

# Summarize uncertainty in demography by average difference in lambda_m from baseline lambda
demographic_i_vec = np.arange(0, len(metric_spl_all), 1).astype(int)
'''Hardcoding in 2 as min tau to match computations, update if needed'''
tau_samples = np.linspace(2, tau_vec.max(), 100)
baseline_lam = metric_spl_all[0](tau_samples)
lam_diff_vec = np.full(demographic_i_vec.size, np.nan)
for demographic_i in demographic_i_vec:
    # Compute the average difference in this sample's lambda from baseline across tau_samples
    lam_diff = np.mean(metric_spl_all[demographic_i](tau_samples) - baseline_lam)
    lam_diff_vec[demographic_i] = lam_diff

def get_pair_results(C_i, n_i, l_i, Sstar, num_param_bins):
    # Get the slice of range-wide stability at the specified decision parameters
    idx = ".".join(str(i) for i in [C_i, n_i, l_i])
    S_slice = np.array(phase[idx])
    x_uncertain = np.array(phase[idx + 'uncertainty_samples'])
    
    # Compute robustness
    counts = np.count_nonzero(S_slice >= Sstar)
    robustness = counts / len(S_slice)
    print(f'robustness={robustness}')
    
    # Compute a normalizing factor to compare parameter bins
    norm_factor = robustness * num_param_bins**2

    # Replace demographic samples in x_uncertain with our summary
    x_uncertain[:, 4] = lam_diff_vec[x_uncertain[:,4].astype(int)]

    # Preallocate filters for samples within bins of each parameter
    bin_filts = {i: np.full((num_param_bins, x_uncertain.shape[0]), False) for i in range(x_uncertain.shape[-1])}
    param_cntrs = {i: np.empty(num_param_bins) for i in range(x_uncertain.shape[-1])}

    for param_i in range(x_uncertain.shape[-1]):
        # Bin the parameter along its sampled range
        if uncertain_params[param_i] == 'mean_lam_diff':
            '''There's an outlier making lam diff right skewed, take percentiles for now'''
            param_low = np.percentile(lam_diff_vec, 0.1)
            param_high = np.percentile(lam_diff_vec, 99.9)
        else:
            param_low = x_uncertain[:, param_i].min()
            param_high = x_uncertain[:, param_i].max()
        param_edges, step = np.linspace(param_low, param_high, num_param_bins + 1, retstep=True)

        # Store bin centers for use in plotting
        _param_cntrs = param_edges[:-1] + step/2
        param_cntrs[param_i] = _param_cntrs
        for bin_i, edge in enumerate(param_edges[:-1]):
            _filt = (x_uncertain[:, param_i] > edge) & (x_uncertain[:, param_i] <= edge + step)
            bin_filts[param_i][bin_i] = _filt

    # Initialize matrix to store results in
    results = np.full((len(param_pairs), num_param_bins, num_param_bins), np.nan)

    # Loop over each bin for each parameter pair and compute a statistic on S 
    for pair_i, (param_i, param_j) in enumerate(param_pairs):
        # Loop over bin combinations
        bin_combinations = product(range(num_param_bins), range(num_param_bins))
        for i, j in bin_combinations:
            # Make filter for being within each bin
            joint_filt = bin_filts[param_i][i] & bin_filts[param_j][j]

            # Compute fraction of samples at or above Sstar (i.e. the probability of our target being met) 
            # within each joint parameter bin use that to compute bin's contribution to the robustness
            if np.any(joint_filt):
                P_targetmet = np.count_nonzero(S_slice[joint_filt] >= Sstar) / np.count_nonzero(joint_filt)
                # We need to normalize to make these comparisons
                results[pair_i, i, j] = P_targetmet / norm_factor
                '''Or just look at P_targetmet'''
                # results[pair_i, i, j] = P_targetmet
                '''Or just divide counts in this bin by total counts'''
                results[pair_i, i, j] = np.count_nonzero(S_slice[joint_filt] >= Sstar) / counts

    return results, param_cntrs

# Specify number of bins for each uncertainty parameter
num_param_bins = 13

# Specify resource constraint
C = 10 * ncell_tot
assert C in C_vec
C_i = np.argmin(np.abs(C_vec - C))
print(f'C/n_tot={C_vec[C_i]/ncell_tot}')

# Get results under baseline conditions
all_results = {}
results, param_cntrs = get_pair_results(0, 0, 0, S_opt_baseline, num_param_bins)
all_results['baseline'] = results

# Specify what percent decrease in optimal S baseline we want to look at
for q_i in range(q_vec.size):
    # Get corresponding Sstar
    Sstar_i = Sstar_i_optdecisions[q_i]
    Sstar = rob_thresh_vec[Sstar_i]

    # Get results at the specified q value for comparison to baseline
    results, param_cntrs = get_pair_results(C_i, q_i+1, q_i+1, Sstar, num_param_bins)
    all_results['uncertain'] = results

    # Take difference between results
    all_results['uncertain-baseline'] = all_results['uncertain'] - all_results['baseline']

    for condition_key in ['baseline', 'uncertain', 'uncertain-baseline']:
        # Plot them
        figdim = np.array([5,4])
        fig, axes = plt.subplots(len(uncertain_params)-1, len(uncertain_params)-1, figsize=figdim*6)

        for pair_i, (param_i, param_j) in enumerate(param_pairs):
            results_pair = all_results[condition_key][pair_i]

            if condition_key == 'uncertain-baseline':
                cmap = 'PuOr_r'
                # Get extreme value of result for colorbar limits
                extreme = max([np.abs(np.nanmin(results_pair)), np.nanmax(results_pair)])
                norm = colors.TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme)
            else:
                cmap = 'viridis'
                norm = colors.Normalize(vmin=np.nanmin(results_pair), vmax=np.nanmax(results_pair))

            if param_i in [0, 2, 4]:
                origin = 'upper'
            else:
                origin = 'lower'

            if param_j  in [0, 2, 4]:
                results_pair = np.fliplr(results_pair.copy())
                x_axis = np.flip(param_cntrs[param_j])
            else:
                x_axis = param_cntrs[param_j]

            im = axes[param_i,param_j-1].imshow(results_pair, origin=origin, norm=norm, cmap=cmap)
            if param_j == len(uncertain_params) - 1:
                # label = r'$\Delta P(S \geq S^*)$' if condition_key == 'uncertain-baseline' else r'$P(S \geq S^*)$'
                label = r'$\Delta$ contribution to $\omega$' if condition_key == 'uncertain-baseline' else r'contribution to $\omega$'
                cbar = fig.colorbar(im, shrink=0.75, label=label)
            else:
                cbar = fig.colorbar(im, shrink=0.75)
            cbar.ax.tick_params(labelsize=plt.rcParams['axes.labelsize'] * 0.5)
            tick_spacing = 1 if num_param_bins <= 5 else 2
            axes[param_i,param_j-1].set_xlabel(param_labels[param_j], fontsize=plt.rcParams['axes.titlesize']*1.75)
            axes[param_i,param_j-1].set_xticks(np.arange(num_param_bins)[::tick_spacing],
                                               np.round(x_axis, 2)[::tick_spacing],
                                               size=plt.rcParams['axes.labelsize']*0.25)
            axes[param_i,param_j-1].set_ylabel(param_labels[param_i], fontsize=plt.rcParams['axes.titlesize']*1.75)
            axes[param_i,param_j-1].set_yticks(np.arange(num_param_bins)[::tick_spacing], np.round(param_cntrs[param_i], 2)[::tick_spacing])
            axes[param_i,param_j-1].tick_params(axis='both', labelsize=plt.rcParams['axes.labelsize'] * 0.75)

        if condition_key == 'uncertain-baseline':
            fig.savefig(fig_prefix + f'uncertainty_pairs_{condition_key}_{np.round(q_vec[q_i],2)}.png', bbox_inches='tight', dpi=dpi)
        plt.close(fig)
