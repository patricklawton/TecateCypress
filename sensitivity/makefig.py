import numpy as np
from matplotlib import colors
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec
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
C_i_vec = [1, 2]
results_pre = 'gte_thresh' 
#results_pre = 'distribution_avg' 

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

################################################################

# Initialize figure
#fig, axes = plt.subplots(2, 2, figsize=np.array([7.,5])*2.5)
fig = plt.figure(figsize=np.array([7.75, 5])*2.5)
gs = GridSpec(3, 2, figure=fig, width_ratios=[1,1], height_ratios=[1.5,1,1])
ax1 = fig.add_subplot(gs[0, 0]) 
ax2 = fig.add_subplot(gs[0, 1])  
ax3 = fig.add_subplot(gs[1:, 0])
ax4 = fig.add_subplot(gs[1:, 1])
axes = np.array([[ax1,ax2],
                 [ax3,ax4]
                ])
gs.update(hspace=0.4)

#### VIZ OF METAPOP METRIC ####
set_globals(results_pre)
# Make interpolation function for <metric> wrt tau
metric_expect_vec = np.ones(tau_vec.size) * np.nan
for tau_i, tau in enumerate(tau_vec):
    tau_filt = (all_tau == tau)
    metric_slice = all_metric[tau_filt]
    metric_expect_vec[tau_i] = np.mean(metric_slice)
t = tau_vec[2:-2:1] 
k = 3
t = np.r_[(0,)*(k+1), t, (tau_vec[-1],)*(k+1)]
spl = make_lsq_spline(tau_vec[1:], metric_expect_vec[1:], t, k)

# Plot <metric> distribution outline
tau_max = max(tau_edges)
bin_step = 0.01
bin_edges = np.concatenate(([0], np.arange(bin_step/2, 1+bin_step, bin_step)))
color = 'limegreen'
mlab = "$<S>$"
metric_interp = spl(tau_flat[tau_flat < tau_max])
counts, bin_edges, hist = axes[0,0].hist(metric_interp, bins=bin_edges, color=color, histtype='step', lw=histlw);
if np.any(counts[5:] == 0):
    xmax_i = np.min(np.nonzero(counts[5:] == 0)[0]) + 5
    xmax = bin_edges[xmax_i + 1] 
else:
    xmax = 1
# Fill in area gte thresh
thresh = 0.5
bin_i = np.argmin(np.abs(bin_edges - thresh))
axes[0,0].hist(metric_interp[metric_interp >= bin_edges[bin_i]], bins=bin_edges, color=color, 
               histtype='bar', alpha=0.6);
closest_bin_i = np.argmin(np.abs(thresh - bin_edges))
axes[0,0].axvline(bin_edges[closest_bin_i], ls='--', color='darkgreen', lw=6.5)
# Labels
axes[0,0].set_yticks([])
#axes[0,0].set_ylabel(rf"{mlab} frequency within range")
axes[0,0].set_ylabel(rf"{mlab} frequency")
xticks = [0, 0.25, 0.5, 0.75, 1]
axes[0,0].set_xticks(xticks, labels=xticks)
axes[0,0].set_xlabel(rf"average survival probabiltiy {mlab}")
axes[0,0].set_xlim(-0.01, xmax)
########

#### FIRE REGIME SHIFT EXAMPLES ####
set_globals(results_pre)
tau_step = np.diff(tau_vec)[0] / 2
tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])
tau_i_samples = [22, 24]
tau_f_samples = [24, 35]
tau_current = tau_flat.copy()
current_counts, _ = np.histogram(tau_current, bins=tau_edges)
mask = np.ones(tau_current.size, dtype=bool)
C_i = np.argmin(np.abs((C_vec/ncell_tot) - 10))
C = C_vec[C_i]

vmin = 0; vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
colormap = copy.copy(cm.RdPu_r)
bins = 50
density = False
alpha = 1

for tau_i, tau_f in zip(tau_i_samples, tau_f_samples):
    bin_i = np.argmin(np.abs(tau_edges - tau_i))
    sl = np.count_nonzero(tau_flat <= tau_edges[bin_i]); sl
    bin_f = np.argmin(np.abs(tau_edges - tau_f))
    ncell = np.count_nonzero(tau_flat < tau_edges[bin_f]) - sl
    color = colormap(norm(ncell/ncell_tot))
    tauc = C / ncell
    mask[tau_argsort[sl:sl+ncell]] = False

    axes[0,1].hist(tau_flat[(tau_flat >= tau_edges[bin_i]) & (tau_flat < tau_edges[bin_f])],
            bins=tau_edges, color=color, alpha=alpha*0.6, density=density);

    tau_slice = tau_flat[tau_argsort][sl:sl+ncell]
    tau_current = tau_flat.copy()
    '''inflate the last bin for any gt max, as we do in actual calcs'''
    tau_current[tau_current >= max(tau_edges)] = tau_edges[-2]
    future_pos_filt = (tau_flat >= min(tau_slice)+tauc) & (tau_flat < max(tau_slice)+tauc)
    current_future_slice = tau_current[future_pos_filt]
    post_shift = np.concatenate((tau_slice+tauc, current_future_slice))
    axes[0,1].hist(post_shift, bins=tau_edges, color=color, alpha=alpha*0.6,
            density=density, histtype='stepfilled');
    # Find where post shift hist is nonzero
    post_bin_i = np.argmin(np.abs(tau_edges - min(post_shift)))
    axes[0,1].hist(post_shift, bins=tau_edges[post_bin_i:], color=color, alpha=alpha*1,
            density=density, histtype='step', linewidth=histlw, zorder=-1);

    tau_shifted = tau_flat.copy()
    tau_shifted[tau_argsort[sl:sl+ncell]] += tauc
    '''inflate the last bin for any gt max, as we do in actual calcs'''
    tau_shifted[tau_shifted >= max(tau_edges)] = tau_edges[-2]

axes[0,1].hist(tau_shifted[mask,...], bins=tau_edges, color='white', density=density);
axes[0,1].hist(tau_current, bins=tau_edges, color='black', histtype='step', lw=histlw, density=density);

axes[0,1].set_yticks([])
#axes[0,1].set_ylabel(r"$\tau$ frequency within range")
axes[0,1].set_ylabel(r"$\tau$ frequency")
axes[0,1].set_xticks(np.arange(20,100,20).astype(int))
axes[0,1].set_xlabel(r"average fire return interval $\tau$")
axes[0,1].set_xlim(15, 81);
########

#### RESULTS UNDER ZERO UNCERTAINTY ####
width = 0.875
set_globals(results_pre)
with h5py.File(fn_prefix + "/phase_flat.h5", "r") as phase_handle:
    data_key = "0.0/0.0/0.0/0.0/phase"
    phase_slice_zeroeps = phase_handle[data_key][:]
    nochange_zeroeps = phase_handle["0.0/0.0/0.0/0.0/metric_nochange"][()]

plot_vec = np.ones_like(C_vec) * np.nan
c_vec = np.ones_like(C_vec) * np.nan
for C_i, C in enumerate(C_vec):
    argmax = np.nanargmax(phase_slice_zeroeps[C_i, :, :])
    optimal_param_i = np.unravel_index(argmax, phase_slice_zeroeps.shape[1:])
    plot_vec[C_i] = phase_slice_zeroeps[C_i, optimal_param_i[0], optimal_param_i[1]]
    c_vec[C_i] = ncell_vec[optimal_param_i[0]]
c_vec = c_vec / ncell_tot

vmin = 0; vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
colormap = copy.copy(cm.RdPu_r)
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
bar_colors = colormap(norm(c_vec))
bar = axes[1,0].bar(np.arange(C_vec.size), plot_vec, color=bar_colors, width=width)
axes[1,0].set_ylim(nochange_zeroeps, 1.02*np.max(plot_vec))
yticks = np.arange(0., 1.2, 0.2)
axes[1,0].set_yticks(yticks[yticks >= nochange_zeroeps])
axes[1,0].set_ylabel(fr"maximum {metric_lab}")
xtick_spacing = 2
#xticks = np.arange(1, len(C_vec)+1, xtick_spacing)
if len(C_vec) % 2 == 0:
    xticks = np.arange(1, len(C_vec)+1, xtick_spacing)
    xtick_labels = np.round((C_vec/(ncell_tot))[1::xtick_spacing], 1)
else:
    xticks = np.arange(0, len(C_vec), xtick_spacing)
    xtick_labels = np.round((C_vec/(ncell_tot))[0::xtick_spacing], 1)
#xtick_labels = np.round((C_vec/(ncell_tot))[0::xtick_spacing], 1)
axes[1,0].set_xticks(xticks, labels=xtick_labels);
axes[1,0].set_xlabel(r"$\hat{\tau}$ if spread over entire range ${C}~/~n_{tot}$")
axes[1,0].set_xlim(-(width/2)*1.4, len(C_vec)-1+((width/2)*1.4))
########

#### OPTIMA ACROSS ROBUSTNESS REQUIREMENTS ####
maxrob = np.load(fn_prefix + "maxrob.npy")
argmaxrob = np.load(fn_prefix + "argmaxrob.npy")
rob_thresh_vec = np.load(fn_prefix + "rob_thresh_vec.npy")

colormap = copy.copy(cm.RdPu_r)
vmax = 1; vmin = 0
normalize = colors.Normalize(vmin=vmin, vmax=vmax)
all_markers = ['o','^','D','s','H','*']
all_linestyles = ['dotted', 'dashdot', 'dashed', 'solid']
#C_i_samples = [1,5,9]
#C_i_samples = [1,3,6,9,11]
C_i_samples = [0,1,2,3,4]
for line_i, C_i in enumerate(C_i_samples):
    plot_vec = np.ones(len(rob_thresh_vec)) * np.nan
    c_vec = np.ones(len(rob_thresh_vec)) * np.nan
    for thresh_i, thresh in enumerate(rob_thresh_vec):
        #if thresh_i % 2 == 0: continue
        # Get the maximum robustness across (ncell, sl) at this C
        if maxrob[thresh_i, C_i] < 1:
            plot_vec[thresh_i] = maxrob[thresh_i, C_i]
        if not np.isnan(plot_vec[thresh_i]):
            c_vec[thresh_i] = ncell_vec[int(argmaxrob[thresh_i, C_i][0])] / ncell_tot
    # Filter out some samples for clarity
    samp_spacing = 1
    scatter = axes[1,1].scatter(plot_vec[::samp_spacing], rob_thresh_vec[::samp_spacing], cmap=colormap, norm=normalize,
                        c=c_vec[::samp_spacing], marker=all_markers[line_i])#, s=60)
    axes[1,1].scatter([], [], label=fr"$C~/~n_{{tot}}=${np.round(C_vec[C_i]/ncell_tot, 1)}",
               c='black', marker=all_markers[line_i])
axes[1,1].set_xlabel(fr"required robustness $\omega$")
axes[1,1].set_ylabel(fr"maximum $S_{{meta}}^*$")
handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(handles[::-1], labels[::-1])
########

# Add colorbar on separate axis
cbar_ax = fig.add_axes([0.2, 0.009, 0.6, 0.025])  # [left, bottom, width, height]
fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal', 
    label=r"optimal range fraction for intervention $n~/~n_{tot}$")

fig.savefig(f'{results_pre}/figs/fig2_pre.png', bbox_inches='tight', dpi=dpi)
fig.savefig(f'{results_pre}/figs/fig2_pre.svg', bbox_inches='tight')

################################################################

fig = plt.figure(figsize=np.array([12, 6])*2.)
gs = GridSpec(4, 4, figure=fig, width_ratios=[1,1,1,1], height_ratios=[1,1,1,1])
ax1 = fig.add_subplot(gs[:, :2])  # Left panel for FDM
ax2 = fig.add_subplot(gs[0, 2:])  # Top right for tau hist
ax3 = fig.add_subplot(gs[1:, 2:])  # Bottom right for P(S) vs tau

#### FDM map ####
# Color data
#cmap = copy.copy(cm.YlOrRd)
#vmin = 0; vmax = 1
cmap = copy.copy(cm.YlOrRd_r)
vmin = 15; vmax = 45
norm = colors.Normalize(vmin=vmin, vmax=vmax)
colored_data = cmap(norm(tau_raster))
# Color background
colored_data[maps_filt == False] = colors.to_rgba('black', alpha=0.3)
# Crop out border where all nans
nonzero_indices = np.nonzero(maps_filt)
row_min, row_max = nonzero_indices[0].min(), nonzero_indices[0].max()
col_min, col_max = nonzero_indices[1].min(), nonzero_indices[1].max()
colored_data = colored_data[row_min:row_max + 1, col_min:col_max + 1]

im1 = ax1.imshow(colored_data)
# Add the colorbar to inset axis
cbar_ax = inset_axes(ax1, width="5%", height="50%", loc='center',
                     bbox_to_anchor=(-0.1, -0.15, 0.65, 0.9),  # Centered on the plot,
                     bbox_transform=ax1.transAxes, borderpad=0)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")#, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar_ticks = cbar.get_ticks()
cbar.set_ticks(cbar_ticks)
cbar_ticklabels = [rf'$\leq${cbar_ticks[0]}'] + [t for t in cbar_ticks[1:-1]] + [rf'$\geq${cbar_ticks[-1]}']
cbar.set_ticklabels(cbar_ticklabels)
#cbar.set_label('30 year fire probability', color='white', rotation=-90, labelpad=cbar_lpad)
cbar.set_label(r'average fire return interval $\tau$', color='white', rotation=-90, labelpad=cbar_lpad)
cbar.ax.tick_params(labelcolor='white', color='white')
ax1.set_xticks([])
ax1.set_yticks([])
########

#### P(S) vs tau ####
set_globals(results_pre)
metric_edges = np.linspace(0, 1, 50)
metric_hist = np.histogram2d(all_tau, all_metric, bins=[tau_edges[::2], metric_edges], density=False)
cmap = copy.copy(cm.YlGn)
cmap.set_bad(cmap(0.0))
imshow_mat = np.flip(np.transpose(metric_hist[0]), axis=0)
samples_per_tau = int(len(all_metric)/len(b_vec))
imshow_mat = imshow_mat / samples_per_tau
# smoothed = scipy.ndimage.gaussian_filter(imshow_mat, sigma=0.125)
im = ax3.imshow(imshow_mat, cmap=cmap,
              norm=colors.LogNorm(vmax=1), interpolation="nearest")

# Add the colorbar to inset axis
cbar_ax = inset_axes(ax3, width="5%", height="50%", loc='center',
                     bbox_to_anchor=(0.5, -0.1, 0.55, 1.1),
                     bbox_transform=ax3.transAxes, borderpad=0)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", ticks=[0, 0.25, 0.5, 0.75, 1.])
cbar.set_label(rf'frequency of $S$ given ${{\tau}}$', rotation=-90, labelpad=cbar_lpad)

S_samples = np.linspace(0, 1, 5)
yticks = (imshow_mat.shape[0] - 1) * S_samples
ax3.set_yticks(yticks, labels=np.flip(S_samples));
ax3.set_ylabel(rf'simulated survival probability $S$')
ax3.set_ylim(np.array([imshow_mat.shape[0], 0]) - 0.5)
tau_samples = np.arange(20, 140, 20).astype(int)
xticks = tau_samples / (tau_step*2)
ax3.set_xticks(xticks, labels=tau_samples);
ax3.set_xlabel(r"average fire return interval $\tau$")
ax3.set_xlim(1, imshow_mat.shape[1]-1)

# Make interpolation function
metric_expect_vec = np.ones(tau_vec.size) * np.nan
for tau_i, tau in enumerate(tau_vec):
    tau_filt = (all_tau == tau)
    metric_slice = all_metric[tau_filt]
    metric_expect_vec[tau_i] = np.mean(metric_slice)
t = tau_vec[2:-2:1]
k = 3
t = np.r_[(0,)*(k+1), t, (tau_vec[-1],)*(k+1)]
spl = make_lsq_spline(tau_vec[1:], metric_expect_vec[1:], t, k)
# Overplot interpolated means
tau_samples = np.arange(5, 140, 2)
mean_points = (imshow_mat.shape[0]-1) * (1 - spl(tau_samples))
x_samples = tau_samples / (tau_step*2)
ax3.plot(x_samples, mean_points, color='k')
dot_spacing = 5
ax3.scatter(x_samples[::dot_spacing], mean_points[::dot_spacing], color='k')
ax3.plot([], [], marker='o', color='k', label=r'$<S>$')
ax3.legend(bbox_to_anchor=(0.2, -0.05, 0.5, 0.5), fontsize=21)
########

#### INITIAL TAU DISTRIBUTION ####
tau_flat = tau_raster[maps_filt]
tau_argsort = np.argsort(tau_flat)
ax2.hist(tau_flat, bins=tau_edges, color='black', histtype='step', lw=histlw, density=True);
ax2.set_yticks([])
ax2.set_ylabel(r"$\tau$ frequency")
# ax2.set_xlabel(r"average fire return interval $\tau$")
ax2.set_xticks([])
# Use the boundaries of the P(S) imshow plot for the tau limits
ax2.set_xlim(metric_hist[1][1], metric_hist[1][imshow_mat.shape[1]-1])
gs.update(hspace=-0.21)  # Adjust vertical spacing
gs.update(wspace=0.3)
########

fig.savefig(f'{results_pre}/figs/fig1_pre.png', bbox_inches='tight', dpi=dpi)
fig.savefig(f'{results_pre}/figs/fig1_pre.svg', bbox_inches='tight')

################################################################

cmap = copy.copy(cm.twilight_shifted)
vmin = -1; vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for C_i in C_i_vec:
    fig = plt.figure(figsize=np.array([6.8, 6])*2.)
    gs = GridSpec(3, 4, figure=fig, width_ratios=[2, 2, 1.1, 1.1], height_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0, :2])  # Top-left subplot
    ax2 = fig.add_subplot(gs[1:, :2])  # Bottom-left subplot
    ax3 = fig.add_subplot(gs[:, 2])  # Right subplot
    ax4 = fig.add_subplot(gs[:, 3])  # Right subplot

    # Read in / set constants
    set_globals(results_pre)
    stepfit_T1 = np.load(fn_prefix + f"/{C_i}/stepfit_T1.npy")
    total_inoptima = np.load(fn_prefix + f"/{C_i}/total_inoptima.npy")
    closest_thresh_i = np.load(fn_prefix + f"/{C_i}/closest_thresh_i.npy")

    mapi_sorted = mapindices[tau_argsort].T
    # Initialize array for sensitivity metric; no changes needed to case of inclusions (T1 > 0)
    '''NOTE sampling is only approximately even across omega, could be throwing results off'''
    sens_metric = total_inoptima / closest_thresh_i.size
    inclusion_filt = (stepfit_T1 > 0) & (stepfit_T1 <= 1)
    # Flip sign and adjust value for cells with exclusions
    exclusion_filt = (stepfit_T1 < 0)
    sens_metric[exclusion_filt] = -1 * sens_metric[exclusion_filt]

    alwaysex_filt = (stepfit_T1 == 2)
    sens_metric[alwaysex_filt] = 0
    '''could set this to -1, doesn't matter'''
    '''Should split the quantity thats == 1 on top & bottom in the tau plot?'''
    alwaysin_filt = (stepfit_T1 == 3)
    sens_metric[alwaysin_filt] =-1

    ### INITIAL TAU DISTRIBUTION VIZ ###
    colorbar_samples = np.linspace(-1, 1, 51)
    line_colors = cmap(norm(colorbar_samples))
    tau_edges = np.linspace(min(tau_flat), 44, 80)
    tau_step = np.diff(tau_edges)[0]
    tau_cntrs = tau_edges + tau_step
    all_heights = np.ones((colorbar_samples.size, tau_edges.size)) * np.nan
    for tau_edge_i, tau_edge in enumerate(tau_edges):
        if tau_edge_i < tau_edges.size - 1:
            tau_filt = (tau_sorted >= tau_edge) & (tau_sorted < tau_edges[tau_edge_i+1])
        else:
            tau_filt = (tau_sorted >= tau_edge)
        sensm_slice = sens_metric[tau_filt]
        heights = np.ones(colorbar_samples.size) * np.nan
        for c_samp_i, c_samp in enumerate(colorbar_samples):
            heights[c_samp_i] = np.count_nonzero(sensm_slice <= c_samp) / sensm_slice.size
        all_heights[:, tau_edge_i] = heights
    '''
    density hist initial tau
    for each sens metric sample, in reverse order
        overplot histogram of corresponding heights multiplied by that bin's density of initial tau, density=False, with corresponding color
    '''
    tau_densities, _, _ = ax1.hist(tau_flat, bins=tau_edges, color='black', histtype='step', lw=3.5, density=True);
    '''Above I include all counts gte the last bin edge in the final bin
       Numpy/matplotlib histogram doesnt do that, so I'll just cut off the last bin from above'''
    for i, c_samp_i in enumerate(np.flip(np.arange(colorbar_samples.size))):
        bar_heights = (all_heights[c_samp_i,:-1] * tau_densities)
        bar_colors = cmap(norm(colorbar_samples[c_samp_i]))
        ax1.bar(tau_cntrs[:-1], bar_heights, width=np.diff(tau_cntrs[:-1])[0], color=bar_colors)
    ax1.set_yticks([])
    # ax1.set_ylabel(r"$\tau$ prevalence within range")
    ax1.set_xlabel(r"$\tau$")
    ax1.set_xlim(tau_edges[0], tau_edges[-2])
    ######

    ### GEOGRAPHICAL MAP ###
    colored_data = np.ones(maps_filt.shape + (4,)) * np.nan #colors in rgba
    colored_data[mapi_sorted[0], mapi_sorted[1]] = cmap(norm(sens_metric))
    # Color background
    colored_data[maps_filt == False] = colors.to_rgba('black', alpha=0.3)
    # Crop out border where all nans
    nonzero_indices = np.nonzero(maps_filt)
    row_min, row_max = nonzero_indices[0].min(), nonzero_indices[0].max()
    col_min, col_max = nonzero_indices[1].min(), nonzero_indices[1].max()
    colored_data = colored_data[row_min:row_max + 1, col_min:col_max + 1]

    im = ax2.imshow(colored_data)#, aspect='auto')
    ax2.set_yticks([])
    ax2.set_xticks([])
    ######

    ### COLORMAP ###
    cbar_pos = ax3.get_position()  # Get original position of the colorbar
    new_cbar_pos = [
        cbar_pos.x0 + 0.13,  # Shift right to create left white space
        cbar_pos.y0 + 0.04,         # Keep vertical position
        cbar_pos.width * 0.25,  # Narrow the width
        cbar_pos.height * 0.9
    ]
    ax3.set_position(new_cbar_pos)  # Apply new position
    sm = cm.ScalarMappable(cmap=copy.copy(cm.twilight), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax3, aspect=60, ticks=[])

    ax4.axis('off')
    ######

    fn = f'{results_pre}/figs/fig3_pre_{C_i}.png'
    fig.savefig(fn, bbox_inches='tight', dpi=dpi)
    fn = f'{results_pre}/figs/fig3_pre_{C_i}.svg'
    fig.savefig(fn, bbox_inches='tight', dpi=dpi)
