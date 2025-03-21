def adjustmaps(maps):
    '''For making SDM and FDM the same shape'''
    dim_len = []
    for dim in range(2):
        dim_len.append(min([m.shape[dim] for m in maps]))
    for mi, m in enumerate(maps):
        maps[mi] = m[0:dim_len[0], 0:dim_len[1]]
    return maps

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

