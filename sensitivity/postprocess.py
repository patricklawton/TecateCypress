from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import numpy as np
import pickle
import signac as sg
from scipy.special import gamma
import scipy
from global_functions import adjustmaps
import h5py
from itertools import product
import os

# Define/load things non-specific to a given set of results
metric = "P_s"
Aeff = 7.29
t_final = 300
ncell_tot = 87_993
c = 1.42
with sg.H5Store('shared_data.h5').open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])
tau_vec = b_vec * gamma(1+1/c)
tau_step = np.diff(tau_vec)[0] / 2
tau_edges = np.concatenate(([0], np.arange(tau_step/2, tau_vec[-1]+tau_step, tau_step)))
tauc_methods = ["flat"]
C_i_vec = [1,2] # For generation of cell metric data
overwrite_robustness = True
overwrite_cellmetric = True

# Function to read in things specific to given results as global variables
def set_globals(results_pre):
    globals()['metric_lab'] = f'$S_{{meta}}$'
    globals()['fn_prefix'] = f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/"
    globals()['fig_prefix'] = f"{results_pre}/figs/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/"

    # Load things saved specific to these results
    with open(fn_prefix + 'metric_data.pkl', 'rb') as handle:
        globals()['metric_data'] = pickle.load(handle)
    globals()['all_metric'] = metric_data['all_metric']
    globals()['all_tau'] = np.load(f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/all_tau.npy")
    globals()['C_vec'] = np.load(fn_prefix + "C_vec.npy")
    globals()['ncell_vec'] = np.load(fn_prefix + "ncell_vec.npy")
    globals()['slice_left_all'] = np.load(fn_prefix + "slice_left_all.npy")
    eps_axes = {}
    with h5py.File(fn_prefix + "/eps_axes.h5", "r") as handle:
        for key in handle.keys():
            eps_axes.update({key: handle[key][()]})
    globals()['eps_axes'] = eps_axes

#### POSTPROCESS ROBUSTNESS ####
# for results_pre in ['distribution_avg']:
for results_pre in ['gte_thresh']:
    if not overwrite_robustness: continue
    # Load things saved specific to these results
    set_globals(results_pre)

    # Collect <metric> across all state variables and uncertainty parameterizations
    shape = [len(eps_axes[key]) for key in eps_axes.keys()] 
    shape += [len(C_vec), len(ncell_vec), len(slice_left_all)]
    phase_full = np.ones((shape)) * np.nan
    with h5py.File(fn_prefix + "phase_flat.h5", "r") as phase_handle:
        for eps_params in product(*eps_axes.values()):
            # Create a dictionary of indices along with values for the current combination
            eps_params_dict = {
                key: (index, np.round(value, 3)) for key, values in zip(eps_axes.keys(), eps_axes.values())
                for index, value in enumerate(values) if value == eps_params[list(eps_axes.keys()).index(key)]
            }

            # Get phase slice at this epsilon parameterization
            data_key = f"{eps_params_dict['mu_tau'][1]}/{eps_params_dict['sigm_tau'][1]}/"
            data_key += f"{eps_params_dict['mu_tauc'][1]}/{eps_params_dict['sigm_tauc'][1]}/phase"
            phase_slice = phase_handle[data_key][:]

            # Add them to collective phase_all
            eps_indices = [val[0] for val in eps_params_dict.values()]
            index_tuple = tuple(eps_indices) + (slice(None), slice(None), slice(None))
            phase_full[index_tuple] = phase_slice
    # Save full phase matrix to file
    np.save(fn_prefix + "phase_full.npy", phase_full)

    rob_thresh_vec = np.linspace(min(phase_full.flatten()), max(phase_full.flatten()), 100)
    np.save(fn_prefix + "rob_thresh_vec.npy", rob_thresh_vec)
    allrob = np.ones((rob_thresh_vec.size, C_vec.size, ncell_vec.size, slice_left_all.size)) * np.nan
    maxrob = np.ones((len(rob_thresh_vec), len(C_vec))) * np.nan
    argmaxrob = np.ones((len(rob_thresh_vec), len(C_vec), 2)) * np.nan
    tot_eps_samples = np.cumprod([len(axis) for axis in eps_axes.values()])[-1]
    zero_eps_i = [np.argwhere(ax == 0)[0][0] for ax in eps_axes.values()]
    for (thresh_i, thresh), (C_i, C) in product(enumerate(rob_thresh_vec), 
                                                enumerate(C_vec)):
        rob_slice = np.ones((len(ncell_vec), len(slice_left_all))) * np.nan
        for (ncell_i, ncell), (sl_i, sl) in product(enumerate(ncell_vec),
                                                    enumerate(slice_left_all)):
            # First, check that this result is feasible with zero uncertainty
            # Skip and keep at nan if not feasible
            metric_zero_eps = phase_full[tuple(zero_eps_i + [C_i, ncell_i, sl_i])]
            if np.isnan(metric_zero_eps) or (metric_zero_eps < thresh): continue
            # Now, get the robstness at this (C,ncell,slice_left) coordinate and store
            counts = np.count_nonzero(phase_full[..., C_i, ncell_i, sl_i] >= thresh)
            robustness = counts / tot_eps_samples
            rob_slice[ncell_i, sl_i] = robustness
        allrob[thresh_i, C_i] = rob_slice
        if np.any(~np.isnan(rob_slice)):
            # Store the max robustness at this (thresh, C) coordinate
            maxrob[thresh_i, C_i] = np.nanmax(rob_slice)
            # Also store the optimal param indices
            optimal_param_i = np.unravel_index(np.nanargmax(rob_slice, axis=None), rob_slice.shape)
            argmaxrob[thresh_i, C_i] = optimal_param_i
    # Save maxrob and argmaxrob to files
    np.save(fn_prefix + "maxrob.npy", maxrob)
    np.save(fn_prefix + "argmaxrob.npy", argmaxrob)

#### COMPUTE AND STORE PER CELL METRIC ####
# Read in maps and convert fdm to tau
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
tau_argsort_ref = np.argsort(tau_flat)
tau_sorted = tau_flat[tau_argsort_ref]

# Define functions used for determining best step fit
'''Note these actually are fit for the indices of allowed values, not the y values themselves'''
def step_function1(x, threshold, value_low, value_high):
    return np.where(x < threshold, value_low, value_high)
def step_function2(x, threshold1, threshold2, value_btwn, value_out):
    return np.where((x >= threshold1) & (x < threshold2), value_btwn, value_out)

def stepfit1(X, y, y0_i):
    def mse_loss(params):
        """Mean squared error for the step function fit."""
        threshold = params[0]
        value_low = allowed_values[int(round(params[1]))]
        value_high = allowed_values[int(round(params[2]))]
        y_pred = step_function1(X, threshold, value_low, value_high)
        return np.mean((y - y_pred) ** 2)
    initial_guess = [0.5, 0.5, 0.5]
    bounds = [(X.min(), X.max()), (y0_i, y0_i), (0, allowed_values.size - 1)]
    result = scipy.optimize.differential_evolution(mse_loss, bounds)
    return result
def stepfit2(X, y, y0_i):
    def mse_loss(params):
        threshold1 = params[0]
        threshold2 = params[1]
        value_btwn = allowed_values[int(round(params[2]))]
        value_out = allowed_values[int(round(params[3]))]
        y_pred = step_function2(X, threshold1, threshold2, value_btwn, value_out)
        return np.mean((y - y_pred) ** 2)
    initial_guess = [0.5, 0.5, 0.5, 0.5]
    bounds = [(X.min(), X.max()), (X.min(), X.max()), (0, allowed_values.size - 1), (y0_i, y0_i)]
    result = scipy.optimize.differential_evolution(mse_loss, bounds)
    return result

step_funcs = [step_function1, step_function2]
allowed_values = np.array([0, 1])

# First, populate matrix recording which cells are in optima for a range of omega requirement
omega_samples = np.linspace(0, 1, 50)
# for res_i, results_pre in enumerate(['distribution_avg']):
for res_i, results_pre in enumerate(['gte_thresh']):
    if not overwrite_cellmetric: continue
    set_globals(results_pre)
    maxrob = np.load(fn_prefix + "maxrob.npy")
    argmaxrob = np.load(fn_prefix + "argmaxrob.npy")
    rob_thresh_vec = np.load(fn_prefix + "rob_thresh_vec.npy")
    for C_i in C_i_vec:
        if os.path.isdir(fn_prefix + f"/{C_i}") == False:
            os.makedirs(fn_prefix + f"/{C_i}")

        # Select metric threshold indices as close as possible to desired omega samples
        closest_thresh_i = np.array([np.abs(maxrob[:,C_i] - val).argmin() for val in omega_samples])
        closest_thresh_i = []
        for i, omega in enumerate(omega_samples):
            closest_i = np.nanargmin(np.abs(maxrob[:,C_i] - omega))
            if maxrob[closest_i, C_i] == 1: continue
            closest_thresh_i.append(closest_i)
        closest_thresh_i = np.array(closest_thresh_i)
        # Filter out any repeats
        rpt_filt = np.concatenate(((np.diff(closest_thresh_i) == 0), [False]))
        closest_thresh_i = closest_thresh_i[~rpt_filt]

        # For each omega sample, record which cells are included in optima
        inoptima_vec = np.zeros((tau_flat.size, closest_thresh_i.size)).astype('bool')
        for omega_sample_i, metric_thresh_i in enumerate(closest_thresh_i):
            opt_ncell_i, opt_sl_i = argmaxrob[metric_thresh_i, C_i].astype(int)
            opt_ncell = ncell_vec[opt_ncell_i]
            opt_sl = slice_left_all[opt_sl_i]
            '''Note that we assume the tau axis of inoptima_vec is in sorted order'''
            inoptima_vec[opt_sl:opt_sl+opt_ncell, omega_sample_i] = True
        np.save(fn_prefix + f"/{C_i}/inoptima_vec.npy", inoptima_vec)

        # Now, fit a step function to each cell's data and store information based on that
        stepfit_T1 = np.ones(tau_flat.size) * np.nan
        stepfit_T2 = np.ones(tau_flat.size) * np.nan
        total_inoptima = np.ones(tau_flat.size) * np.nan
        for k in tqdm(range(tau_flat.size)):
        #for k in tqdm(range(100)):
            # Retrieve and relabel cell k's data for fitting
            X = maxrob[:, C_i][closest_thresh_i]
            y = inoptima_vec[k]
            y0_i = np.argwhere(y[0] == allowed_values)[0][0]
            y0 = allowed_values[y0_i]
            # Check if always in/ex-cluded from optima
            if np.all(y == y0):
                # Encode 2 -> always excluded, 3 -> always included
                stepfit_T1[k] = y0 + 2
                total_inoptima[k] = y0*y.size
            # Perform fitting otherwise
            else:
                # Determine the best fit
                best_fit = stepfit1(X,y,y0_i); best_fit_i = 0
                # Use y0 to determine the sign
                sign = 1 if y0 == 0 else -1
                # Store the threshold values with direction from sign
                stepfit_T1[k] = sign * best_fit.x[0]
                if best_fit_i == 1:
                    stepfit_T2[k] = sign * best_fit.x[1]
                total_inoptima[k] = np.count_nonzero(y == 1)
        np.save(fn_prefix + f"/{C_i}/stepfit_T1.npy", stepfit_T1)
        np.save(fn_prefix + f"/{C_i}/stepfit_T2.npy", stepfit_T2)
        np.save(fn_prefix + f"/{C_i}/total_inoptima.npy", total_inoptima)
        np.save(fn_prefix + f"/{C_i}/closest_thresh_i.npy", closest_thresh_i)
