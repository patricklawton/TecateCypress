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
    eps = 1
    if compressed:
        # Add extinction threshold
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
        '''Note the - 1 actually makes this the last valid timestep *index*t'''
        valid_timesteps = np.sum(valid_mask, axis=1) - 1
        final_N = np.take_along_axis(N_slice, valid_timesteps[..., None], axis=1)[:, 0]

        # Add extinction threshold
        ext_mask = valid_timesteps < (N_slice.shape[1] - 1)
        valid_timesteps[ext_mask] = valid_timesteps[ext_mask] + 1
        final_N[ext_mask] = eps
    # Compute lambda
    growthrates = final_N / N_slice[:, 0]
    lam_s_all = np.full(N_tot.shape[0], np.nan)  # Default to NaN
    lam_s_all = growthrates ** (1 / valid_timesteps)
    return lam_s_all

def s(N_tot, compressed, valid_timesteps=None, ext_mask=None,  burn_in_end_i=0):
    if compressed:
        # Place zeros for extirpated replicas' final abundance
        valid_timesteps[ext_mask] = valid_timesteps[ext_mask] + 1
        N_tot[ext_mask, 1] = 0
        N_slice = N_tot
        final_N = N_tot[:, 1]
    else:
        # Indices for slicing
        start_i = burn_in_end_i  # Start after burn-in
        final_i = N_tot.shape[1]  # Last valid timestep
        N_slice = N_tot[:, start_i:final_i]  # Slice N_tot by all post burn in timesteps

        # Mask where N_slice > 0 (avoiding inf values)
        valid_mask = N_slice > 0

        # Just use the first and final timesteps
        valid_timesteps = np.sum(valid_mask, axis=1) - 1
        final_N = np.take_along_axis(N_slice, valid_timesteps[..., None], axis=1)[:, 0]

    # Compute s
    fractional_change = (final_N / N_slice[:, 0]) - 1
    # Take abs bc we only want the real part of the following root
    s_all = np.abs(fractional_change) ** (1 / valid_timesteps) 
    # Put signs back in
    s_all = s_all * np.sign(fractional_change)
    return s_all

#class Numpy_dataloader()
def batch_data(data, batch_size):
    num_batches = (len(data) + batch_size - 1) // batch_size  # Ceiling division
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(data))
        yield data[start:end]

def shuffle_data(data):
    np.random.shuffle(data)
    return data

def transform_data(batch):
    # Example: Scale data to range [0, 1]
    return (batch - np.min(batch)) / (np.max(batch) - np.min(batch)) if np.max(batch) > np.min(batch) else batch
 
'''Not handling remainder batches correctly''' 
def numpy_dataloader(data, batch_size, shuffle=True, transform=None):
    if shuffle:
        data = shuffle_data(data.copy())
    for batch in batch_data(data, batch_size):
        if transform:
            batch = transform_data(batch)
        yield batch
