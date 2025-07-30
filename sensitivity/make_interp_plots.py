import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc,  cm, colors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import signac as sg
from scipy.special import gamma
import copy as copy
import scipy
from global_functions import adjustmaps
import h5py
from scipy.interpolate import make_lsq_spline
from itertools import product
import os

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
dpi = 50
# dpi = 200
custom_colors = ['lightgrey', 'coral', 'orchid', 'blueviolet']

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
    # globals()['fig_prefix'] = os.path.join('/','Volumes', 'Macintosh HD', 'Users', 'patrick', 
    #                                        'Google Drive', 'My Drive', 'Research', 'Regan', 'Figs/')

    # Load things saved specific to these results
    globals()['metric_all'] = np.load(f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/metric_all.npy")
    globals()['tau_all'] = np.load(f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/tau_all.npy")
    globals()['C_vec'] = np.load(fn_prefix + "C_vec.npy")
    globals()['C_i_vec'] = np.arange(len(C_vec))[::2]
    globals()['ncell_vec'] = np.load(fn_prefix + "ncell_vec.npy")
    globals()['slice_left_all'] = np.load(fn_prefix + "slice_left_all.npy")
    eps_axes = {}
    with h5py.File(fn_prefix + "eps_axes.h5", "r") as handle:
        for key in handle.keys():
            eps_axes.update({key: handle[key][()]})
    globals()['eps_axes'] = eps_axes

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

set_globals(results_pre)
decision_indices = np.load(fn_prefix + '/decision_indices.npy')
S_opt_baseline = np.load(fn_prefix + '/S_opt_baseline.npy')
n_opt_baseline, l_opt_baseline = np.load(fn_prefix + '/decision_opt_baseline.npy')

meta_metric_nochange = float(np.load(fn_prefix + 'meta_metric_nochange.npy'))

maxrob = np.load(fn_prefix + "maxrob.npy")
argmaxrob = np.load(fn_prefix + "argmaxrob.npy")
rob_thresh_vec = np.load(fn_prefix + "rob_thresh_vec.npy")
rob_all = np.load(fn_prefix + "rob_all.npy")
decision_opt_uncertain = np.load(fn_prefix + 'decision_opt_uncertain.npy')
n_opt_interp = decision_opt_uncertain[:,0]
l_opt_interp = decision_opt_uncertain[:,1]

q_vec = np.arange(0.0, 1.0, 0.05)

delta_taul_interp = np.full(q_vec.size, np.nan)
delta_tauh_interp = np.full(q_vec.size, np.nan)
taul_interp = np.full(q_vec.size, np.nan)
tauh_interp = np.full(q_vec.size, np.nan)

for q_i, q in enumerate(q_vec):
    # Now get the optimal decisions for (1-q) * optimal S baseline
    Sstar_i = np.argmin(np.abs(rob_thresh_vec - ((1 - q) * S_opt_baseline)) )
    n_opt_rob = int(n_opt_interp[Sstar_i])
    l_opt_rob = int(l_opt_interp[Sstar_i])
    
    # Replace this q value with the closest one we have available
    Sstar = rob_thresh_vec[Sstar_i]
    q_vec[q_i] = 1 - (Sstar / S_opt_baseline)
        
    delta_taul_interp[q_i] = tau_sorted[l_opt_rob] - tau_sorted[l_opt_baseline]
    delta_tauh_interp[q_i] = tau_sorted[l_opt_rob+n_opt_rob] - tau_sorted[l_opt_baseline+n_opt_baseline]
    taul_interp[q_i] = tau_sorted[l_opt_rob]
    tauh_interp[q_i] = tau_sorted[l_opt_rob+n_opt_rob]

fig, ax = plt.subplots(figsize=np.array([8,6])*1)

# shapes = ['x', 'o', 's']
ls = ['-', '--', '-.', ':']
for i, q in enumerate([0.0, 0.2, 0.5]):
    if q == 0:
        label = r'baseline $S_\text{opt}$'
        ax.axvline(S_opt_baseline, ls=ls[i], c='limegreen', label=label, lw=3)
        plt.scatter(S_opt_baseline, n_opt_baseline, marker='o', c='limegreen', label=r'baseline $n_\text{opt}$')
        plt.scatter(S_opt_baseline, l_opt_baseline, marker='x', c='limegreen', label=r'baseline $l_\text{opt}$')
    else:
        # Set the S^* value we're plotting
        q_i = np.argmin(np.abs(q_vec - q))
        q = q_vec[q_i]
        # Get the optimal decision at this {S^*, R} combination
        Sstar_i = np.argmin(np.abs(rob_thresh_vec - ((1 - q) * S_opt_baseline)) )
        Sstar = rob_thresh_vec[Sstar_i]
        label = rf'{np.round(100*q, 1)}% sacrificed'
        ax.axvline(Sstar, ls=ls[i], c='k', label=label, lw=3)

spacing = 5
plt_filt = rob_thresh_vec <= S_opt_baseline
# plt_filt = rob_thresh_vec <= 1
ax.scatter(rob_thresh_vec[plt_filt][::spacing], n_opt_interp[plt_filt][::spacing], label=r'robust $n_\text{opt}$', c='k', marker='o')
ax.scatter(rob_thresh_vec[plt_filt][::spacing], l_opt_interp[plt_filt][::spacing], label=r'robust $l_\text{opt}$', c='k', marker='x')
closest_i_baseline = np.argmin(np.abs(rob_thresh_vec - S_opt_baseline))
ax.scatter(rob_thresh_vec[closest_i_baseline], n_opt_interp[closest_i_baseline], c='k', marker='o')
ax.scatter(rob_thresh_vec[closest_i_baseline], l_opt_interp[closest_i_baseline], c='k', marker='x')
        
# ax.legend(fontsize='12')
fig_legend = plt.figure(figsize=(1, 0.5)) # Adjust size as needed for your legend
ax_legend = fig_legend.add_subplot(111)
handles, labels = ax.get_legend_handles_labels()
ax_legend.legend(handles, labels, loc='center', frameon=False) # Customize loc and frameon as desired
ax_legend.axis('off') # Hide the axes

ax.set_xlabel(r'$S^*$')
# ax.set_xlim(meta_metric_nochange, 1.0);
ax.set_xlim(meta_metric_nochange, S_opt_baseline*1.025);
ax.set_ylim(0, ncell_tot)

print('saving fig')
fig.savefig(fig_prefix + '/nandloptvsSstar.png', bbox_inches='tight', dpi=dpi)

# Restrict the range of plotting to a desired q value
q_lim = 0.75
q_mask = q_vec <= q_lim

fig, ax = plt.subplots(figsize=np.array([7,5])*1.)

ax.scatter(q_vec[q_mask]*100, delta_taul_interp[q_mask], label=r'lowest $\tau$')
ax.axhline(0, ls='--', c='k', lw=1)

ax.scatter(q_vec[q_mask]*100, delta_tauh_interp[q_mask], label=r'highest $\tau$')
ax.axhline(0, ls='--', c='k', lw=1)

## Get the points where before and after crossing baseline and color them differently
#alpha = 0.6
#
## First handle lower bound of optimal tau slice
#lte_baseline_q = q_vec[delta_taul_interp[q_mask & (delta_taul_interp <= 0)].argmax() + 1]
#lte_baseline_mask = (q_vec < lte_baseline_q)
##ax.scatter(q_vec[q_mask]*100, taul_interp[q_mask], c='k', marker='v', label=r'lowest $\tau$')
#marker_x_positions = np.linspace(min(q_vec[q_mask]), max(q_vec[q_mask]), 50)
#marker_y_positions = np.full_like(marker_x_positions, tau_sorted[l_opt_baseline])
#ax.plot(marker_x_positions*100, marker_y_positions, '--', markersize=8, color='k', label='baseline')
#y2 = np.full(np.count_nonzero(q_mask & lte_baseline_mask), tau_sorted[l_opt_baseline])
#ax.fill_between( # First handle less than baseline
#    q_vec[q_mask & lte_baseline_mask]*100,
#    taul_interp[q_mask & lte_baseline_mask],
#    y2,
#    color=custom_colors[3],
#    alpha=alpha,
#    zorder=-1
#)
#y1 = np.full(np.count_nonzero(q_mask & ~lte_baseline_mask), tau_sorted[l_opt_baseline])
#ax.fill_between( # Now handle greater than baseline
#    q_vec[q_mask & ~lte_baseline_mask]*100,
#    y1,
#    taul_interp[q_mask & ~lte_baseline_mask],
#    color=custom_colors[1],
#    alpha=alpha,
#    zorder=-1
#)
#
## Now handle upper bound of optimal tau slice
#lte_baseline_q = q_vec[delta_tauh_interp[q_mask & (delta_tauh_interp <= 0)].argmax() + 1]
#lte_baseline_mask = (q_vec < lte_baseline_q)
##ax.scatter(q_vec[q_mask]*100, tauh_interp[q_mask], label=r'highest $\tau$', c='k', marker='^')
#marker_y_positions = np.full_like(marker_x_positions, tau_sorted[l_opt_baseline+n_opt_baseline])
#ax.plot(marker_x_positions*100, marker_y_positions, '--', markersize=8, color='k')
#y2 = np.full(np.count_nonzero(q_mask & lte_baseline_mask), tau_sorted[l_opt_baseline + n_opt_baseline])
#ax.fill_between( # First handle less than baseline
#    q_vec[q_mask & lte_baseline_mask]*100,
#    tauh_interp[q_mask & lte_baseline_mask],
#    y2,
#    color=custom_colors[1],
#    alpha=alpha,
#    zorder=-1
#)
#y1 = np.full(np.count_nonzero(q_mask & ~lte_baseline_mask), tau_sorted[l_opt_baseline + n_opt_baseline])
#ax.fill_between( # Now handle greater than baseline
#    q_vec[q_mask & ~lte_baseline_mask]*100,
#    y1,
#    tauh_interp[q_mask & ~lte_baseline_mask],
#    color=custom_colors[3],
#    alpha=alpha,
#    zorder=-1
#)

ax.set_xlabel(r'% of $\text{max}(S_{baseline})$ sacrificed')
ax.set_ylabel(r'$\Delta \tau$ relative to baseline')
#ax.set_ylabel(r'optimal $\tau$')
ax.legend()
#ax.set_ylim(-5,25)
fig.savefig(fig_prefix + '/tau_minmax_shift.png', bbox_inches='tight', dpi=dpi+40)
#fig.savefig(fig_prefix + '/tauopt_minmax.png', bbox_inches='tight', dpi=dpi+40)

# Define reference indices for per population tau
tau_indices = np.arange(tau_sorted.size)

q_samples = [0.0, 0.3, 0.6]
for i, q in enumerate(q_samples):
    # Set the S^* value we're plotting
    q_i = np.argmin(np.abs(q_vec - q))
    q = q_vec[q_i]

    # Get the optimal decision at this {S^*, R} combination
    Sstar_rob_i = np.argmin(np.abs(rob_thresh_vec - ((1 - q) * S_opt_baseline)) )
    n_opt = int(n_opt_interp[Sstar_rob_i])
    l_opt = int(l_opt_interp[Sstar_rob_i])

    # Define results vector
    results_vector = np.full(tau_sorted.size, np.nan)
    # where each population is given a number to indicate optimality under:
    #   1 -> baseline condiitons only
    #   2 -> baseline and uncertain conditions (risk aversion)
    #   3 -> uncertain conditions only
    #   0 -> everything else

    # Create relevant masks
    baseline_mask = (tau_indices > l_opt_baseline) & (tau_indices < l_opt_baseline + n_opt_baseline)
    uncertain_mask = (tau_indices > l_opt) & (tau_indices < l_opt + n_opt)
    baseline_only_mask = baseline_mask & (~uncertain_mask)
    uncertain_only_mask = uncertain_mask & (~baseline_mask)
    both_mask = baseline_mask & uncertain_mask
    neither_mask = ~(baseline_mask | uncertain_mask)

    # Use masks to assign values to each population
    results_vector[neither_mask] = 0
    results_vector[baseline_only_mask] = 1
    results_vector[both_mask] = 2
    results_vector[uncertain_only_mask] = 3

    # Define colormaping for categories
    labels = ['neither', 'baseline only', 'both', 'uncertain only']
    cmap = colors.ListedColormap(custom_colors)
    vmin = 0; vmax = len(custom_colors) - 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    ### TAU DISTRIBUTION VIZ ###

    stack_data = [tau_sorted[results_vector == i] for i in range(len(custom_colors))]

    bins = np.linspace(min(tau_flat), 50, 80)

    # Plot the stacked histogram
    fig, ax = plt.subplots(figsize=np.array([8,5])*0.75)
    ax.hist(
        stack_data,
        bins=bins,
        stacked=True,
        color=custom_colors,
        label=[labels[i] for i in range(len(custom_colors))]
    )

    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'baseline $\tau$ frequency')
    ax.set_title(f'$q=${np.round(q, 3)}')
    ax.legend()
    fig.savefig(fig_prefix + f'/shiftmap_q{q_samples[i]}.png', bbox_inches='tight', dpi=dpi)

    ### GEOGRAPHICAL MAP ###

    mapi_sorted = mapindices[tau_argsort].T

    colored_data = np.ones(maps_filt.shape + (4,)) * np.nan #colors in rgba
    colored_data[mapi_sorted[0], mapi_sorted[1]] = cmap(norm(results_vector))
    # Color background
    colored_data[maps_filt == False] = colors.to_rgba('black', alpha=0.3)
    # Crop out border where all nans
    nonzero_indices = np.nonzero(maps_filt)
    row_min, row_max = nonzero_indices[0].min(), nonzero_indices[0].max()
    col_min, col_max = nonzero_indices[1].min(), nonzero_indices[1].max()
    colored_data = colored_data[row_min:row_max + 1, col_min:col_max + 1]

    fig, ax = plt.subplots(figsize=np.array([10,10])*1)
    im = ax.imshow(colored_data)#, aspect='auto')
    ax.set_yticks([])
    ax.set_xticks([])
    fig.savefig(fig_prefix + f'/shiftdist_q{q_samples[i]}.png', bbox_inches='tight', dpi=dpi)

