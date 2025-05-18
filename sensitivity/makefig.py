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
import os

# Define/load things non-specific to a given set of results
#metric = "P_s"
metric = 'lambda_s'
#metric = 's'
Aeff = 7.29
t_final = 300
ncell_tot = 87_993
c = 1.42
with sg.H5Store('shared_data.h5').open(mode='r') as sd:
    b_vec = np.array(sd['b_vec'])
tau_vec = b_vec * gamma(1+1/c)
#tau_step = np.diff(tau_vec)[0] / 2
#tau_edges = np.concatenate(([0], np.arange(tau_step/2, tau_vec[-1]+tau_step, tau_step)))
tauc_methods = ["flat"]
#C_i_vec = [0,1,2,4,6]
#C_i_vec = [0,2,4,6]
C_i_vec = [0, 1]
results_pre = 'gte_thresh' 
#results_pre = 'distribution_avg' 

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

# Function to read in things specific to given results as global variables
def set_globals(results_pre):
    if metric == 'lambda_s':
        globals()['metric_lab'] = r'$\lambda_{meta}$'
        globals()['rob_metric_lab'] = r'$\lambda_{meta}^*$'
        globals()['mean_metric_lab'] = r'$<\lambda>$'
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
    #globals()['fig_prefix'] = os.path.join('/','Volumes', 'Macintosh HD', 'Users', 'patrick', 
    #                                       'Google Drive', 'My Drive', 'Research', 'Regan', 'Figs/')

    # Load things saved specific to these results
    globals()['all_metric'] = np.load(f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/metric_{metric}/all_metric.npy")
    globals()['all_tau'] = np.load(f"{results_pre}/data/Aeff_{Aeff}/tfinal_{t_final}/all_tau.npy")
    globals()['C_vec'] = np.load(fn_prefix + "C_vec.npy")
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

################################################################

#fig = plt.figure(figsize=np.array([12, 6])*2.)
fig = plt.figure(figsize=np.array([14, 6])*2.)
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
tau_diffs = np.diff(tau_vec)
tau_step = tau_diffs[1]
tau_edges = np.concatenate((
                [tau_vec[0]],
                [tau_diffs[0]/2],
                np.arange(tau_vec[1]+tau_step/2, tau_vec[-1]+tau_step, tau_step)
                           ))
min_edge_i = 2
metric_min = min(all_metric[(all_tau >= tau_edges[min_edge_i]) & (all_tau < tau_edges[min_edge_i+1])])
metric_edges = np.linspace(metric_min, all_metric.max()*1.005, 50)
cmap = copy.copy(cm.YlGn)
#cmap.set_bad(cmap(0.0))
im = ax3.hist2d(all_tau, all_metric, bins=[tau_edges, metric_edges],
                 norm=colors.LogNorm(vmax=int(len(all_metric)/len(b_vec))),
                 density=False,
                cmap=cmap)

# Add the colorbar to inset axis
cbar_ax = inset_axes(ax3, width="5%", height="50%", loc='center',
                     bbox_to_anchor=(0.5, -0.2, 0.55, 1.1),
                     bbox_transform=ax3.transAxes, borderpad=0)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", ticks=[0, 0.25, 0.5, 0.75, 1.])

# Make interpolation function for <metric> wrt tau
metric_expect_vec = np.ones(tau_vec.size) * np.nan
for tau_i, tau in enumerate(tau_vec):
    tau_filt = (all_tau == tau)
    metric_slice = all_metric[tau_filt]
    metric_expect_vec[tau_i] = np.mean(metric_slice)
t = tau_vec[2:-2:2] 
k = 3
t = np.r_[(tau_vec[1],)*(k+1), t, (tau_vec[-1],)*(k+1)]
spl = make_lsq_spline(tau_vec[1:], metric_expect_vec[1:], t, k)

# Overplot spline and label axes, etc
tau_samples = np.arange(0, 140, 2)
ax3.plot(tau_samples, spl(tau_samples), color='k')
dot_spacing = 2
ax3.scatter(tau_vec[1::dot_spacing], spl(tau_vec[1::dot_spacing]), color='k')
if metric == 'P_s':
    cbar.set_label(rf'frequency of $S$ given ${{\tau}}$', rotation=-90, labelpad=cbar_lpad)
    ax3.set_ylabel(rf'simulated survival probability $S$')
elif metric == 'lambda_s':
    cbar.set_label(rf'frequency of $\lambda$ given ${{\tau}}$', rotation=-90, labelpad=cbar_lpad)
    ax3.set_ylabel(rf'simulated growth rate $\lambda$')
elif metric == 's':
    ax3.set_ylabel(rf's')
ax3.plot([], [], marker='o', color='k', label=mean_metric_lab)
ax3.legend(bbox_to_anchor=(0.2, -0.05, 0.5, 0.5), fontsize=21)
ax3.set_ylim(metric_edges[np.nonzero(im[0][min_edge_i])[0].min()], max(metric_edges))
ax3.set_xlim(tau_edges[min_edge_i], max(tau_edges))
ax3.set_xlabel(r'average fire return interval $\tau$')
########

#### INITIAL TAU DISTRIBUTION ####
#tau_edges = np.arange(0, int(max(tau_vec)+0.5)+1, 1)
tau_edges = np.arange(0, int(max(tau_edges)+0.5)+1, 1)
tau_flat = tau_raster[maps_filt]
tau_argsort = np.argsort(tau_flat)
ax2.hist(tau_flat, bins=tau_edges, color='black', histtype='step', lw=histlw, density=True);
ax2.set_yticks([])
ax2.set_ylabel(r"$\tau$ frequency")
ax2.set_xticks([])
# Use the boundaries of the P(S) imshow plot for the tau limits
ax2.set_xlim(ax3.get_xlim())
#gs.update(hspace=-0.21)  # Adjust vertical spacing
gs.update(hspace=0.01)  # Adjust vertical spacing
gs.update(wspace=0.3)
########

fig.savefig(fig_prefix + 'fig1_pre.png', bbox_inches='tight', dpi=dpi)
fig.savefig(fig_prefix + 'fig1_pre.svg', bbox_inches='tight')

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

# Plot <metric> distribution outline
tau_max = max(tau_edges)
color = 'limegreen'
metric_interp = spl(tau_flat[tau_flat < tau_max])
bin_step = 0.0025
bin_edges = np.concatenate(([0], np.arange(bin_step/2, 1+bin_step, bin_step)))
counts, bin_edges, hist = axes[0,0].hist(metric_interp, bins=bin_edges, color=color, histtype='step', lw=histlw);
if metric == 'P_s':
    if np.any(counts[5:] == 0):
        xmax_i = np.min(np.nonzero(counts[5:] == 0)[0]) + 5
        xmax = bin_edges[xmax_i + 1] 
    else:
        xmax = 1
    xticks = [0, 0.25, 0.5, 0.75, 1]
    axes[0,0].set_xlabel(rf"average survival probabiltiy {mean_metric_lab}")
    axes[0,0].set_xlim(-0.01, xmax)
else:
    xticks = [0.9, 0.95, 1]
    #xticks = np.round(np.arange(0.9, 1.02, 0.02), 3)
    axes[0,0].set_xlim(0.89,1.0+bin_step)
    axes[0,0].set_xlabel(rf"average stochastic growth rate {mean_metric_lab}")
# Fill in area gte thresh
if metric == 'P_s':
    thresh = 0.5
elif metric == 'lambda_s':
    thresh = 0.975
elif metric == 's':
    thresh = 0.975
bin_i = np.argmin(np.abs(bin_edges - thresh))
axes[0,0].hist(metric_interp[metric_interp >= bin_edges[bin_i]], bins=bin_edges, color=color, 
               histtype='bar', alpha=0.6);
closest_bin_i = np.argmin(np.abs(thresh - bin_edges))
axes[0,0].axvline(bin_edges[closest_bin_i], ls='--', color='darkgreen', lw=6.5)
# Labels
axes[0,0].set_yticks([])
axes[0,0].set_ylabel(rf"{mean_metric_lab} frequency")
axes[0,0].set_xticks(xticks, labels=xticks)
########

#### FIRE REGIME SHIFT EXAMPLES ####
set_globals(results_pre)
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
    #tau_current[tau_current >= max(tau_edges)] = tau_edges[-2]
    tau_current[tau_current >= max(tau_vec)] = max(tau_vec)
    future_pos_filt = (tau_flat >= min(tau_slice)+tauc) & (tau_flat < max(tau_slice)+tauc)
    current_future_slice = tau_current[future_pos_filt]
    if tau_i == min(tau_i_samples):
        xmax = max(current_future_slice)+2
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
#axes[0,1].set_xlim(15, 81);
axes[0,1].set_xlim(15, xmax);
########

#### RESULTS UNDER ZERO UNCERTAINTY ####
# Load all results
set_globals(results_pre)
x_all = np.load(fn_prefix + '/x_all.npy')
meta_metric_all = np.load(fn_prefix + '/meta_metric_all.npy')
meta_metric_all = meta_metric_all[:,0]
meta_metric_nochange = float(np.load(fn_prefix + 'meta_metric_nochange.npy'))

plot_vec = np.ones_like(C_vec) * np.nan
c_vec = np.ones_like(C_vec) * np.nan
for C_i, C in enumerate(C_vec):
    #argmax = np.nanargmax(phase_slice_zeroeps[C_i, :, :])
    #optimal_param_i = np.unravel_index(argmax, phase_slice_zeroeps.shape[1:])
    #plot_vec[C_i] = phase_slice_zeroeps[C_i, optimal_param_i[0], optimal_param_i[1]]
    #c_vec[C_i] = ncell_vec[optimal_param_i[0]]
    zeroeps_filt = np.all(x_all[:, 3:] == 0, axis=1)
    _filt = zeroeps_filt & (x_all[:,0] == C)
    argmax = np.nanargmax(meta_metric_all[_filt])
    plot_vec[C_i] = meta_metric_all[_filt][argmax]
    c_vec[C_i] = x_all[_filt,:][argmax][1]
c_vec = c_vec / ncell_tot

width = 0.875
vmin = 0; vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
colormap = copy.copy(cm.RdPu_r)
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
bar_colors = colormap(norm(c_vec))
bar = axes[1,0].bar(np.arange(C_vec.size), plot_vec, color=bar_colors, width=width)
axes[1,0].set_ylim(meta_metric_nochange, 1.02*np.max(plot_vec))
yticks = np.arange(0., 1.2, 0.2)
axes[1,0].set_yticks(yticks[yticks >= meta_metric_nochange])
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
#C_i_samples = [0,2,4,6]
C_i_samples = C_i_vec.copy()
#C_i_samples = [i for i in range(C_vec.size)][::1]
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
    #scatter = axes[1,1].scatter(rob_thresh_vec[::samp_spacing], plot_vec[::samp_spacing], cmap=colormap, norm=normalize,
    #                    c=c_vec[::samp_spacing], marker=all_markers[line_i])#, s=60)
    axes[1,1].scatter([], [], label=fr"$C~/~n_{{tot}}=${np.round(C_vec[C_i]/ncell_tot, 1)}",
               c='black', marker=all_markers[line_i])
axes[1,1].set_xlabel(fr"required robustness $\omega$")
axes[1,1].set_ylabel(f"maximum {rob_metric_lab}")
handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(handles[::-1], labels[::-1])
########

# Add colorbar on separate axis
cbar_ax = fig.add_axes([0.2, 0.009, 0.6, 0.025])  # [left, bottom, width, height]
fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal', 
    label=r"optimal range fraction for intervention $n~/~n_{tot}$")

fig.savefig(fig_prefix + 'fig2_pre.png', bbox_inches='tight', dpi=dpi)
fig.savefig(fig_prefix + 'fig2_pre.svg', bbox_inches='tight')

################################################################

cmap = copy.copy(cm.twilight_shifted)
vmin = -1; vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for C_i in C_i_vec:
    tauhat_min = np.round(C_vec[C_i]/ncell_tot, 2)
    fig = plt.figure(figsize=np.array([6.8, 6])*2.)
    gs = GridSpec(3, 4, figure=fig, width_ratios=[2, 2, 1.1, 1.1], height_ratios=[1, 1, 1])
    ax1 = fig.add_subplot(gs[0, :2])  # Top-left subplot
    ax2 = fig.add_subplot(gs[1:, :2])  # Bottom-left subplot
    ax3 = fig.add_subplot(gs[:, 2])  # Right subplot
    ax4 = fig.add_subplot(gs[:, 3])  # Right subplot
    gs.update(hspace=0.28)  # Adjust vertical spacing

    # Read in / set constants
    set_globals(results_pre)
    #stepfit_T1 = np.load(fn_prefix + f"{tauhat_min}/stepfit_T1.npy")
    total_inoptima = np.load(fn_prefix + f"{tauhat_min}/total_inoptima.npy")
    closest_thresh_i = np.load(fn_prefix + f"{tauhat_min}/closest_thresh_i.npy")
    inoptima_vec = np.load(fn_prefix + f"{tauhat_min}/inoptima_vec.npy")

    mapi_sorted = mapindices[tau_argsort].T
    # Initialize array for sensitivity metric; no changes needed to case of inclusions (T1 > 0)
    '''NOTE sampling is only approximately even across omega, could be throwing results off'''
    sens_metric = total_inoptima / closest_thresh_i.size
    '''DONT ACTUALLY NEED TO USE T1'''
    #inclusion_filt = (stepfit_T1 > 0) & (stepfit_T1 <= 1)
    # Flip sign and adjust value for cells with exclusions
    #exclusion_filt = (stepfit_T1 < 0)
    exclusion_filt = (sens_metric != 1) & (inoptima_vec[:, 0] == True)
    sens_metric[exclusion_filt] = -1 * sens_metric[exclusion_filt]

    #alwaysex_filt = (stepfit_T1 == 2)
    #sens_metric[alwaysex_filt] = 0
    '''change always in value to -1 for the sake of plotting'''
    '''Should split the quantity thats == 1 on top & bottom in the tau plot?'''
    #alwaysin_filt = (stepfit_T1 == 3)
    #sens_metric[alwaysin_filt] =-1
    sens_metric[sens_metric == 1] = -1

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
    #cbar = fig.colorbar(sm, cax=ax3, aspect=60, ticks=[])
    ax3.axis('off')

    ax4.axis('off')
    ######

    fn = fig_prefix + f'fig3_pre_{tauhat_min}.png'
    fig.savefig(fn, bbox_inches='tight', dpi=dpi)
    fn = fig_prefix + f'fig3_pre_{tauhat_min}.svg'
    fig.savefig(fn, bbox_inches='tight', dpi=dpi)
