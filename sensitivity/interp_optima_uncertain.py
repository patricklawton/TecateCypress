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
#phase = h5py.File(fn_prefix + '/phase.h5', 'r')
decision_indices = np.load(fn_prefix + '/decision_indices.npy')
S_opt_baseline = np.load(fn_prefix + '/S_opt_baseline.npy')
#decision_opt_baseline = np.load(fn_prefix + '/decision_opt_baseline.npy')
n_opt_baseline, l_opt_baseline = np.load(fn_prefix + '/decision_opt_baseline.npy')

meta_metric_nochange = float(np.load(fn_prefix + 'meta_metric_nochange.npy'))

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
SMOOTHING = 0.01

C = 6
assert np.any(np.isclose(C_vec/ncell_tot, C))
C_i = np.isclose(C_vec/ncell_tot, C).argmax()

# First, filter robustness values for this R value
rob_all_filtered = rob_all[:, C_i, ...]

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

NUM_RESTARTS = 5

# Define objective function (i.e. robustness) to be optimized
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
