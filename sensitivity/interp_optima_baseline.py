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
C_vec = np.load(fn_prefix + '/C_vec_baseline.npy')
x_all = np.load(fn_prefix + '/x_all_baseline.npy')
meta_metric_all = np.load(fn_prefix + '/meta_metric_all_baseline.npy')
meta_metric_all = meta_metric_all[:,0]

meta_metric_nochange = float(np.load(fn_prefix + 'meta_metric_nochange.npy'))

# Create a filter for the baseline scenario
zero_eps_mask = np.all(x_all[:, 3:] == 0, axis=1)

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

C = 6
assert np.any(np.isclose(C_vec/ncell_tot, C))
C_i = np.isclose(C_vec/ncell_tot, C).argmax()

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
'''Fudge factor; comparison to robust optima can get messed up if baseline optimal S is even slightly too big'''
S_opt_baseline *= 1.

# Save results to file
np.save(fn_prefix + 'decision_opt_baseline.npy', np.array([n_opt_baseline, l_opt_baseline]))
np.save(fn_prefix + 'S_opt_baseline.npy', S_opt_baseline)
