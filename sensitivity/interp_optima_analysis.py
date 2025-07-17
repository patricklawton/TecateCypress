import numpy as np
from matplotlib import colors
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import rc
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
x_all = np.load(fn_prefix + '/x_all.npy')
meta_metric_all = np.load(fn_prefix + '/meta_metric_all.npy')
meta_metric_all = meta_metric_all[:,0]

meta_metric_nochange = float(np.load(fn_prefix + 'meta_metric_nochange.npy'))

# Create a filter for the baseline scenario
zero_eps_mask = np.all(x_all[:, 3:] == 0, axis=1)

maxrob = np.load(fn_prefix + "maxrob.npy")
argmaxrob = np.load(fn_prefix + "argmaxrob.npy")
rob_thresh_vec = np.load(fn_prefix + "rob_thresh_vec.npy")
rob_all = np.load(fn_prefix + "rob_all.npy")

# Class to rescale inputs and outputs to [0,1] for numerical stability
# Also store descalers to interpret interpolated values
class Rescaler:
    def __init__(self, mins, maxes):
        """
        mins: vector of minima
        maxes: vector of maxima
        """
        self.mins = mins
        self.maxes = maxes

    def rescale(self, x):
        return (x - self.mins) / (self.maxes - self.mins)

    def descale(self, x):
        return (x * (self.maxes - self.mins)) + self.mins

from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize

NN_ROB = 150
NN_S = 100
# SMOOTHING = 0.15
SMOOTHING = 0.01

C = 9
assert np.any(np.isclose(C_vec/ncell_tot, C))
C_i = np.isclose(C_vec/ncell_tot, C).argmax()

# First, filter robustness values for this R value
rob_all_filtered = rob_all[:, 0, ...]

# Unravel the filtered robustness values into y_obs
y_obs = rob_all_filtered.flatten()

# Place the corresponding decision parameter values into x_obs
indices = np.unravel_index(np.arange(y_obs.size), rob_all_filtered.shape)
x_obs = np.full((y_obs.size, len(rob_all_filtered.shape)), np.nan)
x_obs[:, 0] = rob_thresh_vec[indices[0]]
x_obs[:, 1] = ncell_vec[indices[1]]
x_obs[:, 2] = slice_left_all[indices[2]]

# Filter out any invalid param combinations from both x and y
nan_filt = np.isnan(y_obs)
# y_obs = y_obs[~nan_filt]
# x_obs = x_obs[~nan_filt, :]
'''Instead try putting zeros in y_obs for nan'''
y_obs[nan_filt] = 0.0

# Rescale the y values (i.e. the robustness values)
y_rescaler = Rescaler(y_obs.min(axis=0), y_obs.max(axis=0))
y_obs = y_rescaler.rescale(y_obs) 
    
# Rescale the x values (i.e. the decision parameters)
x_rescaler = Rescaler(x_obs.min(axis=0), x_obs.max(axis=0))
x_obs = x_rescaler.rescale(x_obs)

# Create interpolator for robustness(S^*, n, l) given R
interp = RBFInterpolator(x_obs, y_obs, neighbors=NN_ROB, smoothing=SMOOTHING)

# Get parameters and S values under baseline condintions
C_mask = (x_all[:, 0] == (C*ncell_tot))
baseline_mask = np.all(x_all[:, 3:] == 0, axis=1)
x_obs_baseline = x_all[C_mask & baseline_mask, 1:3]
y_obs_baseline = meta_metric_all[C_mask & baseline_mask]

# Rescale inputs and outputs
x_rescaler_baseline = Rescaler(x_obs_baseline.min(axis=0), x_obs_baseline.max(axis=0))
x_obs_baseline = x_rescaler_baseline.rescale(x_obs_baseline)
y_rescaler_baseline = Rescaler(y_obs_baseline.min(axis=0), y_obs_baseline.max(axis=0))
y_obs_baseline = y_rescaler_baseline.rescale(y_obs_baseline) 

# Interpolate S(n, l) given R under baseline conditions for reference during optimization
interp_baseline = RBFInterpolator(x_obs_baseline, y_obs_baseline, neighbors=NN_S, smoothing=0.0)

# First optimize decision under baseline conditions
def objective_baseline(decision_params):
    S = interp_baseline([decision_params])
    return -S

# Use optimal decision from exisiting samples as starting point
zeroeps_filt = np.all(x_all[:, 3:] == 0, axis=1)
_filt = zeroeps_filt & (x_all[:,0] == C*ncell_tot)
argmax = np.nanargmax(meta_metric_all[_filt])
n0, l0 = x_all[_filt,:][argmax][1:3].astype(int)
n0, l0 = x_rescaler_baseline.rescale([n0, l0])

# Optimize using scipy
x0 = np.array([n0, l0])
bounds = ((0, 1), (0, 1)) # Remeber, we rescaled the training data
cons = [{'type': 'ineq', 'fun': lambda x:  1 - x[1] - x[0]}] # Constrain l < (n_tot - n)
res = minimize(objective_baseline, x0, method='COBYLA', bounds=bounds, constraints=cons)
n_opt_baseline, l_opt_baseline = x_rescaler_baseline.descale(res.x).astype(int)
S_opt_baseline = y_rescaler_baseline.descale(-res.fun)
np.save(fn_prefix + 'decision_opt_baseline.npy', np.array([n_opt_baseline, l_opt_baseline]))
np.save(fn_prefix + 'S_opt_baseline.npy', S_opt_baseline)
'''Fudge factor'''
S_opt_baseline *= 1.

NUM_RESTARTS = 5

# Define objective function (i.e. robustness) to be optimized
def objective(decision_params, *args):
    '''This should get descaled before proceeding'''
    Sstar = args[0]
    # Sstar_descaled = x_rescaler.descale([Sstar, decision_params[0], decision_params[1]])[0]

    # S_baseline = interp_baseline([decision_params])
    # S_baseline = y_rescaler_baseline.descale(S_baseline)
    # if S_baseline < Sstar_descaled:
    #     # Only consider robustness to be nonzero if S^* met under baseline
    #     robustness = 0
    # else:
    if True:
        # Take robustness value from interpolation
        x = np.full(len(decision_params)+1, np.nan)
        x[0] = Sstar
        x[1:] = decision_params
        try:
            robustness = interp([x])
        except:
            # print(x)
            # import sys; sys.exit()
            robustness = 0
        # print(robustness)

    return -robustness # Negate bc using minimization algorithm

# Now step through S^* values and find decisions that optimize robustness
n_opt_interp = np.full(rob_thresh_vec.size, np.nan)
l_opt_interp = np.full(rob_thresh_vec.size, np.nan)
for Sstar_i, Sstar in enumerate(rob_thresh_vec):
    # Use optimal decisions from exisiting samples as starting points
    argsort = np.argsort(rob_all_filtered[Sstar_i, :], axis=None)
    nan_filt = np.isnan(rob_all_filtered[Sstar_i, :].ravel()[argsort])
    argsort = argsort[~nan_filt]

    n_opt_samples = []
    l_opt_samples = []

    for x0_position in argsort[-NUM_RESTARTS:]:
        n0_i, l0_i = np.unravel_index(x0_position, rob_all_filtered.shape[1:])
        n0, l0 = (ncell_vec[n0_i], slice_left_all[l0_i])

        # Rescale to interpolation scale
        Sstar, n0, l0 = x_rescaler.rescale([Sstar, n0, l0])

        # Use an optimizer that can handle some noise in the objective
        x0 = np.array([n0, l0])
        # bounds = ((0, 1), (0, 1)) # Remeber, we rescaled the training data
        # cons = [{'type': 'ineq', 'fun': lambda x:  1 - x[1] - x[0]}] # Constrain l < (n_tot - n)
        # res = minimize(objective, x0, args=(Sstar), method='COBYLA', bounds=bounds, constraints=cons)
        cons = [
            {'type': 'ineq', 'fun': lambda x: x[0]},          # x[0] >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[0]},      # x[0] <= 1
            {'type': 'ineq', 'fun': lambda x: x[1]},          # x[1] >= 0
            {'type': 'ineq', 'fun': lambda x: 1 - x[1]},      # x[1] <= 1
            {'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]}  # your original constraint: l < (1 - n)
        ]
        res = minimize(objective, x0, args=(Sstar,), method='COBYLA', constraints=cons)

        _, n_opt, l_opt = x_rescaler.descale([Sstar, res.x[0], res.x[1]])
        n_opt_samples.append(n_opt)
        l_opt_samples.append(l_opt)

    # Take the mean over multiple optimization runs
    n_opt_interp[Sstar_i] = np.mean(n_opt_samples)
    l_opt_interp[Sstar_i] = np.mean(l_opt_samples)

decision_opt_uncertain = np.full((rob_thresh_vec.size, 2), np.nan)
decision_opt_uncertain[:, 0] = n_opt_interp
decision_opt_uncertain[:, 1] = l_opt_interp
np.save(fn_prefix + 'decision_opt_uncertain.npy', decision_opt_uncertain.astype(int))

# q_vec = np.arange(0.0, 0.85, 0.05)
q_vec = np.arange(0.0, 1.0, 0.05)

delta_taul_interp = np.full(q_vec.size, np.nan)
delta_tauh_interp = np.full(q_vec.size, np.nan)

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

fig, ax = plt.subplots(figsize=(8,6))

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

fig.savefig(fig_prefix + '/nandloptvsSstar.png', bbox_inches='tight')

# Restrict the range of plotting to a desired q value
q_lim = 0.85
# q_lim = 1.
q_mask = q_vec <= q_lim

fig, ax = plt.subplots(figsize=(7,5))

ax.scatter(q_vec[q_mask]*100, delta_taul_interp[q_mask], label=r'lowest $\tau$')
ax.axhline(0, ls='--', c='k', lw=1)

ax.scatter(q_vec[q_mask]*100, delta_tauh_interp[q_mask], label=r'highest $\tau$')
ax.axhline(0, ls='--', c='k', lw=1)

ax.set_xlabel(r'% of $\text{max}(S_{baseline})$ sacrificed')
ax.set_ylabel(r'$\Delta \tau$ relative to baseline')
ax.legend()
ax.set_ylim(-5,10)
fig.savefig(fig_prefix + '/tau_minmax_shift.png', bbox_inches='tight')

# Define reference indices for per population tau
tau_indices = np.arange(tau_sorted.size)

q_samples = [0.0, 0.225, 0.4, 0.6, 0.85]
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
    custom_colors = ['lightgrey', 'coral', 'orchid', 'blueviolet']
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
    fig.savefig(fig_prefix + f'/shiftmap_q{q_samples[i]}.png', bbox_inches='tight')

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

    fig, ax = plt.subplots(figsize=np.array([10,10])*0.7)
    im = ax.imshow(colored_data)#, aspect='auto')
    ax.set_yticks([])
    ax.set_xticks([])
    fig.savefig(fig_prefix + f'/shiftdist_q{q_samples[i]}.png', bbox_inches='tight')
