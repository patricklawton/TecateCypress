import numpy as np

def adjustmaps(maps):
    '''For making SDM and FDM the same shape'''
    dim_len = []
    for dim in range(2):
        dim_len.append(min([m.shape[dim] for m in maps]))
    for mi, m in enumerate(maps):
        maps[mi] = m[0:dim_len[0], 0:dim_len[1]]
    return maps

def lambda_s(N_tot, compressed, valid_timesteps=None, ext_mask=None,  burn_in_end_i=0):
    if compressed:
        # Add extinction threshold
        eps = 1
        valid_timesteps[ext_mask] = valid_timesteps[ext_mask] + 1
        N_tot[ext_mask, 1] = eps
        N_slice = N_tot
        final_N = N_tot[:, 1]
    else:
        # Indices for slicing
        start_i = burn_in_end_i  # Start after burn-in
        final_i = N_tot.shape[1]  # Last valid timestep
        N_slice = N_tot[:, start_i:final_i]  # Slice N_tot by all post burn in timesteps

        # Mask where N_slice > 0 (avoiding inf values)
        valid_mask = N_slice > 0
        masked_N_slice = np.where(valid_mask, N_slice, np.nan)  # Replace zeros with NaN (ignored in np.nanprod)

        # Just use the first and final timesteps
        valid_timesteps = np.sum(valid_mask, axis=1) - 1
        final_N = np.take_along_axis(N_slice, valid_timesteps[..., None], axis=1)[:, 0]

        # Add extinction threshold
        eps = 1
        ext_mask = valid_timesteps < (N_slice.shape[1] - 1)
        valid_timesteps[ext_mask] = valid_timesteps[ext_mask] + 1
        final_N[ext_mask] = eps
    # Compute lambda
    growthrates = final_N / N_slice[:, 0]
    lam_s_all = np.full(N_tot.shape[0], np.nan)  # Default to NaN
    lam_s_all = growthrates ** (1 / valid_timesteps)
    return lam_s_all

#import pickle
#import numpy as np
#import h5py
#
## Function to read in things specific to given results as global variables
#def set_globals(results_pre, Aeff, t_final, metric):
#    globals()['metric_lab'] = f'$S_{{meta}}$'
#    globals()['fn_prefix'] = f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/"
#    globals()['fig_prefix'] = f"{results_pre}/figs/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/"
#
#    # Load things saved specific to these results
#    with open(fn_prefix + 'metric_data.pkl', 'rb') as handle:
#        globals()['metric_data'] = pickle.load(handle)
#    globals()['all_metric'] = metric_data['all_metric']
#    globals()['all_tau'] = np.load(f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/all_tau.npy")
#    globals()['C_vec'] = np.load(fn_prefix + "C_vec.npy")
#    globals()['ncell_vec'] = np.load(fn_prefix + "ncell_vec.npy")
#    globals()['slice_left_all'] = np.load(fn_prefix + "slice_left_all.npy")
#    eps_axes = {}
#    with h5py.File(fn_prefix + "/eps_axes.h5", "r") as handle:
#        for key in handle.keys():
#            eps_axes.update({key: handle[key][()]})
#    globals()['eps_axes'] = eps_axes

