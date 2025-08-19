import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc,  cm, colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib as mpl
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
#dpi = 50
dpi = 200
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
    #globals()['fig_prefix'] = f"{results_pre}/figs/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/"
    globals()['fig_prefix'] = os.path.join('/','Volumes', 'Macintosh HD', 'Users', 'patrick', 
                                           'Google Drive', 'My Drive', 'Research', 'Regan', 'Figs/')

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

xlens = np.array([1.0433, 1.0433]) # left to right
# xlens = np.array([1., 1.]) # left to right
#ylens = np.array([2.25/3, 2/3, 1]) # top to bottom
ylens = np.array([2.25/3, 0.2, 0.55, 1])  # top, legend row, mid, bottom
scale = 5
xlens *= scale; ylens *= scale
figsize = np.array([np.sum(xlens), np.sum(ylens)])
#fig, axd = plt.subplot_mosaic([['top', 'top'],
#                               ['mid left', 'mid right'],
#                               ['bottom left', 'bottom right']],
#                              figsize=figsize, layout="constrained",
#                              height_ratios=ylens)
# Define subplot mosaic
fig, axd = plt.subplot_mosaic(
    [['top', 'top'],
     ['legend', 'legend'],   
     ['mid left', 'mid right'],
     ['bottom left', 'bottom right']],
    figsize=figsize,
    layout="constrained",
    height_ratios=ylens
)

# Add subplot labels
labels = {
    'top': 'a',
    'mid left': 'b',
    'mid right': 'c',
    'bottom left': 'd',
    'bottom right': 'e'
}

# Draw once to get renderer measurements
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

for key, label in labels.items():
    ax = axd[key]
    bbox = ax.get_position()

    # Get y-label bounding box in figure coordinates
    ylab = ax.yaxis.get_label()
    ylab_bbox_disp = ylab.get_window_extent(renderer=renderer)
    ylab_bbox_fig = ylab_bbox_disp.transformed(fig.transFigure.inverted())

    # Left offset: start just left of y-label
    x_pos = ylab_bbox_fig.x0 + 0.05 # small gap

    # Vertical position: slightly above plot box
    y_pos = bbox.y1 - 0.01

    fig.text(
        x_pos,
        y_pos,
        f"({label})",
        fontsize=25,
        fontweight='bold',
        ha='right',
        va='bottom'
    )

# Hide the legend axes frame and ticks
axd['legend'].axis('off')

# Make legend handles
custom_colors_forlegend = ['coral', 'blueviolet', 'orchid', 'lightgrey']
legend_labels = ['baseline only', 'uncertain only', 'both', 'neither']
handles = [Patch(facecolor=c, edgecolor='black', label=l)
           for c, l in zip(custom_colors_forlegend, legend_labels)]

# Place legend in the 'legend' subplot
axd['legend'].axis('off')
axd['legend'].legend(
    handles=handles,
    loc='center',
    frameon=False,
    ncol=len(handles),   # all in one row
    handlelength=2.
)

# Restrict the range of plotting to a desired q value
q_lim = 0.51
q_mask = q_vec <= q_lim

# Get the points where before and after crossing baseline and color them differently
alpha = 0.6

# First handle lower bound of optimal tau slice
lte_baseline_q = q_vec[delta_taul_interp[q_mask & (delta_taul_interp <= 0)].argmax() + 1]
lte_baseline_mask = (q_vec < lte_baseline_q)
#axd['top'].scatter(q_vec[q_mask]*100, taul_interp[q_mask], c='k', marker='v', label=r'lowest $\tau$')
marker_x_positions = np.linspace(min(q_vec[q_mask]), max(q_vec[q_mask]), 50)
marker_y_positions = np.full_like(marker_x_positions, tau_sorted[l_opt_baseline])
axd['top'].plot(marker_x_positions*100, marker_y_positions, '--', markersize=8, color='k', label='baseline, $R~/~n_{tot}=6.0$')
y2 = np.full(np.count_nonzero(q_mask & lte_baseline_mask), tau_sorted[l_opt_baseline])
axd['top'].fill_between( # First handle less than baseline
    q_vec[q_mask & lte_baseline_mask]*100,
    taul_interp[q_mask & lte_baseline_mask],
    y2,
    color=custom_colors[3],
    alpha=alpha,
    zorder=-1
)
y1 = np.full(np.count_nonzero(q_mask & ~lte_baseline_mask), tau_sorted[l_opt_baseline])
axd['top'].fill_between( # Now handle greater than baseline
    q_vec[q_mask & ~lte_baseline_mask]*100,
    y1,
    taul_interp[q_mask & ~lte_baseline_mask],
    color=custom_colors[1],
    alpha=alpha,
    zorder=-1
)

# Now handle upper bound of optimal tau slice
lte_baseline_q = q_vec[delta_tauh_interp[q_mask & (delta_tauh_interp <= 0)].argmax() + 1]
lte_baseline_mask = (q_vec < lte_baseline_q)
#axd['top'].scatter(q_vec[q_mask]*100, tauh_interp[q_mask], label=r'highest $\tau$', c='k', marker='^')
marker_y_positions = np.full_like(marker_x_positions, tau_sorted[l_opt_baseline+n_opt_baseline])
axd['top'].plot(marker_x_positions*100, marker_y_positions, '--', markersize=8, color='k')
y2 = np.full(np.count_nonzero(q_mask & lte_baseline_mask), tau_sorted[l_opt_baseline + n_opt_baseline])
axd['top'].fill_between( # First handle less than baseline
    q_vec[q_mask & lte_baseline_mask]*100,
    tauh_interp[q_mask & lte_baseline_mask],
    y2,
    color=custom_colors[1],
    alpha=alpha,
    zorder=-1
)
y1 = np.full(np.count_nonzero(q_mask & ~lte_baseline_mask), tau_sorted[l_opt_baseline + n_opt_baseline])
axd['top'].fill_between( # Now handle greater than baseline
    q_vec[q_mask & ~lte_baseline_mask]*100,
    y1,
    tauh_interp[q_mask & ~lte_baseline_mask],
    color=custom_colors[3],
    alpha=alpha,
    zorder=-1
)

axd['top'].set_xlabel(r'percent decrease from baseline $\text{max}(S)$ to target stability, $S^*$')
axd['top'].set_xlim(-1, 50)
axd['top'].set_ylabel(r'optimal $\hat{\tau}$')
axd['top'].legend()

# Define reference indices for per population tau
tau_indices = np.arange(tau_sorted.size)

q_samples = [0.0, 0.5]
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
    ax_label = 'mid left' if i == 0 else 'mid right'
    axd[ax_label].hist(
        stack_data,
        bins=bins,
        stacked=True,
        color=custom_colors,
        label=[labels[i] for i in range(len(custom_colors))]
    )

    axd[ax_label].set_xlabel(r'$\tau$')
    axd[ax_label].set_ylabel(r'$\hat{\tau}$ frequency')
    axd[ax_label].set_yticks([])
    # axd[ax_label].legend()

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

    ax_label = 'bottom left' if i == 0 else 'bottom right'
    im = axd[ax_label].imshow(colored_data)
    axd[ax_label].set_yticks([])
    axd[ax_label].set_xticks([])

fig.savefig(fig_prefix + 'fig4_pre.png', bbox_inches='tight')
