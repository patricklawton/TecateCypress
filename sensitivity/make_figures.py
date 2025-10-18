from project import Phase
import numpy as np
from matplotlib import colors, cm, rc
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import h5py
from scipy.special import gamma
import copy as copy
from itertools import combinations, product

# Update global plotting parameters
rc('axes', labelsize=24)  # Font size for x and y labels
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
#dpi = 200
custom_colors = ['lightgrey', '#e69f00', '#ee6778', '#a04a95'] #pop

# Define constants for a Phase instance to collect existing data
constants = {}
constants['c'] = 1.42
constants['Aeff'] = 7.29
constants['t_final'] = 300
constants['sim_method'] = 'discrete'
constants['ul_coord'] = [1500, 2800]
constants['lr_coord'] = [2723, 3905]
'''Should just set this to min tau_vec I think, which it basically already is'''
constants['min_tau'] = 2
constants['A_cell'] = 270**2 / 1e6 #km^2
constants['ncell_min'] = 2_500
constants['root'] = 0 #For mpi
constants['final_max_tau'] =  np.nan
constants['meta_metric'] = 'gte_thresh'
metric_thresh = 0.975 # Threshold of pop metric value used for calculating meta metric
constants['metric'] = 'lambda_s'
constants['overwrite_metrics'] = False
constants['extra_attributes'] = ['tau_raster', 'maps_filt', 'metric_spl_all']

# Create instance and retrieve exisitng data
pproc = Phase(**constants)
pproc.initialize()
pproc.load_decision_parameters(suffix="_baseline")

#########
# FIG 2 #
#########

fig = plt.figure(figsize=np.array([14, 6])*2.)
gs = GridSpec(4, 4, figure=fig, width_ratios=[1,1,1,1], height_ratios=[1,1,1,1])
ax1 = fig.add_subplot(gs[:, :2])  # Left panel for FDM
ax2 = fig.add_subplot(gs[0, 2:])  # Top right for tau hist
ax3 = fig.add_subplot(gs[1:, 2:])  # Bottom right for P(S) vs tau

#### FDM map ####
cmap = copy.copy(cm.YlOrRd_r)
vmin = 15; vmax = 45
norm = colors.Normalize(vmin=vmin, vmax=vmax)
colored_data = cmap(norm(pproc.tau_raster))
# Color background
colored_data[pproc.maps_filt == False] = colors.to_rgba('black', alpha=0.3)
# Crop out border where all nans
nonzero_indices = np.nonzero(pproc.maps_filt)
row_min, row_max = nonzero_indices[0].min(), nonzero_indices[0].max()
col_min, col_max = nonzero_indices[1].min(), nonzero_indices[1].max()
colored_data = colored_data[row_min:row_max + 1, col_min:col_max + 1]

im1 = ax1.imshow(colored_data)
# Add the colorbar to inset axis
cbar_ax = inset_axes(ax1, width="5%", height="60%", loc='center',
                     bbox_to_anchor=(-0.1, -0.15, 0.65, 0.9),  # Centered on the plot,
                     bbox_transform=ax1.transAxes, borderpad=0)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")#, ticks=[0, 0.25, 0.5, 0.75, 1])
cbar_ticks = cbar.get_ticks()
cbar.set_ticks(cbar_ticks)
cbar_ticklabels = [rf'$\leq${cbar_ticks[0]}'] + [t for t in cbar_ticks[1:-1]] + [rf'$\geq${cbar_ticks[-1]}']
cbar.set_ticklabels(cbar_ticklabels)
cbar.set_label(r'baseline fire return interval, $\hat{\tau}_k$', color='white', rotation=-90, labelpad=cbar_lpad)
cbar.ax.tick_params(labelcolor='white', color='white')
ax1.set_xticks([])
ax1.set_yticks([])

#### P(S) vs tau ####
# Define bins for tau axis
tau_plot = np.linspace(pproc.tau_vec[0], pproc.tau_vec[-1], 90)
tau_diffs = np.diff(tau_plot)
tau_step = tau_diffs[1]
tau_edges = np.concatenate((
                [tau_plot[0]],
                [tau_diffs[0]/2],
                np.arange(tau_plot[1]+tau_step/2, tau_plot[-1]+tau_step, tau_step)
                           ))
min_edge_i = np.argmin(np.abs(tau_edges - 10))

# Collect interpolated lambda values at selected tau values
tau_plot_all = np.tile(tau_plot, pproc.num_demographic_samples)
metric_interp_all = np.array([])
for demographic_index in range(pproc.num_demographic_samples):
    metric_spl = pproc.metric_spl_all[demographic_index]
    metric_interp_all = np.append(metric_interp_all, metric_spl(tau_plot))

# Define bins for lambda axis
metric_min = min(metric_interp_all[(tau_plot_all >= tau_edges[min_edge_i]) & (tau_plot_all < tau_edges[min_edge_i+1])])
metric_max = np.quantile(metric_interp_all, 0.99)
metric_edges = np.linspace(metric_min, metric_max*1.015, 60)

# Use hist2d to plot
clrs = ['white', 
        '#edd09b', #this is 50% opacity
        custom_colors[1]]
nodes = [0.0, 0.6, 1.0]
cmap = colors.LinearSegmentedColormap.from_list(
    "white_to_target", list(zip(nodes, clrs))
)
norm = colors.LogNorm(vmin=1, vmax=pproc.num_demographic_samples)
cmap.set_bad('white')
im = ax3.hist2d(tau_plot_all, metric_interp_all, bins=[tau_edges, metric_edges],
                norm=norm,
                density=False,
                cmap=cmap)

# Add the colorbar to inset axis
cbar_ax = inset_axes(ax3, width="5%", height="50%", loc='center',
                     bbox_to_anchor=(0.5, -0.2, 0.55, 1.1),
                     bbox_transform=ax3.transAxes, borderpad=0)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
ticks = np.array([1,10, 100, 500]) * 10
ticklabels = np.round(ticks/pproc.num_demographic_samples, 3)
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", ticks=ticks,
                   format=mticker.FixedFormatter(ticklabels))

# Plot interpolation function for <metric> wrt tau
metric_spl = pproc.metric_spl_all[0]
tau_samples = np.arange(0, 140, 2)
ax3.plot(tau_samples, metric_spl(tau_samples), color='k')

# Labels etc
cbar.set_label(rf'$P(\lambda|\tau_k)$', rotation=-90, labelpad=cbar_lpad)
ax3.set_ylabel(rf'growth rate, $\lambda(\tau_k)$')
ax3.plot([], [], color='k', label=r'baseline: $\hat{\lambda}(\tau_k)$')
ax3.legend(bbox_to_anchor=(0.2, -0.05, 0.5, 0.5), fontsize=24)
ax3.set_ylim(metric_edges[np.nonzero(im[0][min_edge_i])[0].min()], max(metric_edges))
ax3.set_xlim(tau_edges[min_edge_i], max(tau_edges))
ax3.set_xlabel(r'fire return interval, $\tau_k$')

#### INITIAL TAU DISTRIBUTION ####
tau_edges = np.arange(0, int(max(tau_edges)+0.5)+1, 1)
ax2.hist(pproc.tau_flat, bins=tau_edges, color='black', histtype='step', lw=histlw, density=True);
ax2.set_yticks([])
ax2.set_ylabel(r"$\hat{\tau}_k$ frequency")
ax2.set_xticks([])
# Use the boundaries of the P(S) imshow plot for the tau limits
ax2.set_xlim(ax3.get_xlim())
gs.update(hspace=0.01)  # Adjust vertical spacing
gs.update(wspace=0.3)

fig.savefig(pproc.figs_dir + '/fig2_pre.png', bbox_inches='tight', dpi=dpi)
fig.savefig(pproc.figs_dir + '/fig2_pre.svg', bbox_inches='tight')

#########
# FIG 3 #
#########

# Initalize figure
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
tau_max = max(tau_edges)
#color = 'limegreen'
color = custom_colors[1]
metric_spl = pproc.metric_spl_all[0]
metric_interp = metric_spl(pproc.tau_flat[pproc.tau_flat < tau_max])
bin_step = 0.0025
bin_edges = np.concatenate(([0], np.arange(bin_step/2, 1+bin_step, bin_step)))
counts, bin_edges, hist = axes[0,0].hist(metric_interp, bins=bin_edges, color=color, 
                                         histtype='step', lw=histlw);
xticks = [0.9, 0.95, 1]
axes[0,0].set_xlim(min(metric_interp),1.0+bin_step)
axes[0,0].set_xlabel(rf"baseline growth rate, $\hat{{\lambda}}$")
bin_i = np.argmin(np.abs(bin_edges - metric_thresh))
axes[0,0].hist(metric_interp[metric_interp >= bin_edges[bin_i]], 
               bins=bin_edges, color=color,
               histtype='bar', alpha=0.6);
closest_bin_i = np.argmin(np.abs(metric_thresh - bin_edges))
axes[0,0].axvline(bin_edges[closest_bin_i], ls='--', color='k', lw=6.5)
# Labels
axes[0,0].set_yticks([])
axes[0,0].set_ylabel(rf"$\hat{{\lambda}}$ frequency")
axes[0,0].set_xticks(xticks, labels=xticks)

#### FIRE REGIME SHIFT EXAMPLES ####
tau_centers = 0.5 * (tau_edges[:-1] + tau_edges[1:])
tau_i_samples = [22, 24]
tau_f_samples = [24, 35]
tau_current = pproc.tau_flat.copy()
current_counts, _ = np.histogram(tau_current, bins=tau_edges)
mask = np.ones(tau_current.size, dtype=bool)
C_i = np.argmin(np.abs((pproc.C_vec/pproc.ncell_tot) - 10))
C = pproc.C_vec[C_i]

vmin = 0; vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
clrs = [custom_colors[3], 
        custom_colors[2],
        '#EEB7BC', #pink 50% opacity
        'white']
nodes = [0.0, 0.6, 0.85, 1.0]
colormap = colors.LinearSegmentedColormap.from_list(
    "white_to_target", list(zip(nodes, clrs))
)
bins = 50
density = False
alpha = 1

for tau_i, tau_f in zip(tau_i_samples, tau_f_samples):
    bin_i = np.argmin(np.abs(tau_edges - tau_i))
    sl = np.count_nonzero(pproc.tau_flat <= tau_edges[bin_i])
    bin_f = np.argmin(np.abs(tau_edges - tau_f))
    ncell = np.count_nonzero(pproc.tau_flat < tau_edges[bin_f]) - sl
    color = colormap(norm(ncell/pproc.ncell_tot))
    tauc = C / ncell
    mask[pproc.tau_argsort_ref[sl:sl+ncell]] = False

    axes[0,1].hist(pproc.tau_flat[(pproc.tau_flat >= tau_edges[bin_i]) & (pproc.tau_flat < tau_edges[bin_f])],
                   bins=tau_edges, color=color, alpha=alpha, density=density);

    tau_slice = pproc.tau_flat[pproc.tau_argsort_ref][sl:sl+ncell]
    tau_current = pproc.tau_flat.copy()
    '''inflate the last bin for any gt max, as we do in actual calcs'''
    tau_current[tau_current >= max(pproc.tau_vec)] = max(pproc.tau_vec)
    future_pos_filt = (pproc.tau_flat >= min(tau_slice)+tauc) & (pproc.tau_flat < max(tau_slice)+tauc)
    current_future_slice = tau_current[future_pos_filt]
    if tau_i == min(tau_i_samples):
        xmax = max(current_future_slice)+2
    post_shift = np.concatenate((tau_slice+tauc, current_future_slice))
    # Find where post shift hist is nonzero
    post_bin_i = np.argmin(np.abs(tau_edges - min(post_shift)))
    # Mask zero counts in shifted distributions
    counts, edges = np.histogram(post_shift, bins=tau_edges, density=density)
    nonzero = counts > 0
    axes[0,1].hist(
        post_shift,
        bins=tau_edges[:-1][nonzero].tolist() + [tau_edges[1:][nonzero][-1]],
        color=color,
        edgecolor=color,     # solid outline in same color
        alpha=alpha,
        linewidth=histlw,
        density=density,
        histtype='stepfilled',
    )

    tau_shifted = pproc.tau_flat.copy()
    tau_shifted[pproc.tau_argsort_ref[sl:sl+ncell]] += tauc
    '''inflate the last bin for any gt max, as we do in actual calcs'''
    tau_shifted[tau_shifted >= max(tau_edges)] = tau_edges[-2]

axes[0,1].hist(tau_shifted[mask,...], bins=tau_edges, color='white', density=density);
axes[0,1].hist(tau_current, bins=tau_edges, color='black', histtype='step', lw=histlw, density=density);

axes[0,1].set_yticks([])
axes[0,1].set_ylabel(r"$\tau_k$ frequency")
axes[0,1].set_xticks(np.arange(20,100,20).astype(int))
axes[0,1].set_xlabel(r"fire return interval, $\tau_k$")
axes[0,1].set_xlim(15, xmax);

#### RESULTS UNDER ZERO UNCERTAINTY ####
# Load results for baseline
with h5py.File(pproc.data_dir + '/phase_baseline.h5', 'r') as phase:
    decision_samples = np.array(phase['0.0.0decision_samples'])
    meta_metric_all = np.array(phase['0.0.0'])
meta_metric_nochange = float(np.load(pproc.data_dir + '/meta_metric_nochange.npy'))

plot_vec = np.ones_like(pproc.C_vec) * np.nan
c_vec = np.ones_like(pproc.C_vec) * np.nan
for C_i, C in enumerate(pproc.C_vec):
    _filt = decision_samples[:,0] == C
    argmax = np.nanargmax(meta_metric_all[_filt])
    plot_vec[C_i] = meta_metric_all[_filt][argmax]
    c_vec[C_i] = decision_samples[_filt,:][argmax][1]
c_vec = c_vec / pproc.ncell_tot

width = 0.875
vmin = 0; vmax = 1
norm = colors.Normalize(vmin=vmin, vmax=vmax)
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
bar_colors = colormap(norm(c_vec))
bar = axes[1,0].bar(np.arange(pproc.C_vec.size), plot_vec, color=bar_colors, width=width)
axes[1,0].set_ylim(0, 1.02*np.max(plot_vec))
yticks = np.arange(0., 1.2, 0.2)
axes[1,0].set_yticks(yticks)
axes[1,0].set_ylabel(fr"$\text{{max}}~S$")
xtick_spacing = 2
if len(pproc.C_vec) % 2 == 0:
    xticks = np.arange(1, len(pproc.C_vec)+1, xtick_spacing)
    xtick_labels = np.round((pproc.C_vec/(pproc.ncell_tot))[1::xtick_spacing], 1)
else:
    xticks = np.arange(0, len(pproc.C_vec), xtick_spacing)
    xtick_labels = np.round((pproc.C_vec/(pproc.ncell_tot))[0::xtick_spacing], 1)
axes[1,0].set_xticks(xticks, labels=xtick_labels);
axes[1,0].set_xlabel(r"minimum $\hat{\Delta\tau}_k$, $R~/~n_{tot}$")
axes[1,0].set_xlim(-(width/2)*1.4, len(pproc.C_vec)-1+((width/2)*1.4))
# Plot baseline value
axes[1,0].axhline(meta_metric_nochange, ls=':', label=f'no management', c='k')
axes[1,0].legend()

#### OPTIMA ACROSS ROBUSTNESS REQUIREMENTS ####
# Reinitialize phase instance with uncertainty samples 
constants.update({'extra_attributes': None})
pproc = Phase(**constants)
pproc.initialize()
pproc.load_decision_parameters(suffix="_uncertain")

# Load post-processed results
maxrob = np.load(pproc.data_dir + "/maxrob.npy")
argmaxrob = np.load(pproc.data_dir + "/argmaxrob.npy")
Sstar_vec = np.load(pproc.data_dir + "/Sstar_vec.npy")

vmax = 1; vmin = 0
normalize = colors.Normalize(vmin=vmin, vmax=vmax)
all_markers = ['o','^','D','s','H','*']
all_linestyles = ['dotted', 'dashdot', 'dashed', 'solid']
C_i_samples = [i for i in range(pproc.C_vec.size)][::1]

for line_i, C_i in enumerate(C_i_samples):
    plot_vec = np.ones(len(Sstar_vec)) * np.nan
    c_vec = np.ones(len(Sstar_vec)) * np.nan
    for thresh_i, thresh in enumerate(Sstar_vec):
        # Get the maximum robustness across (ncell, sl) at this C
        if maxrob[thresh_i, C_i] < 1:
            plot_vec[thresh_i] = maxrob[thresh_i, C_i]
        if not np.isnan(plot_vec[thresh_i]):
            c_vec[thresh_i] = pproc.ncell_vec[int(argmaxrob[thresh_i, C_i][0])] / pproc.ncell_tot
    # Filter out some samples for clarity
    samp_spacing = 4
    scatter = axes[1,1].scatter(Sstar_vec[::samp_spacing], plot_vec[::samp_spacing], cmap=colormap, norm=normalize,
                        c=c_vec[::samp_spacing], marker=all_markers[line_i])
    axes[1,1].scatter([], [], label=fr"$R~/~n_{{tot}}=${np.round(pproc.C_vec[C_i]/pproc.ncell_tot, 1)}",
               c='black', marker=all_markers[line_i])

    # Get the max robustness under Sstar=S_baseline_nochange for y lim
    if C_i == max(C_i_samples):
        Sstar_i = np.abs(Sstar_vec - meta_metric_nochange).argmin()
        ymax = maxrob[Sstar_i, C_i]
axes[1,1].set_ylabel(fr"$\text{{max}}~P(S \geq S^*)$")
axes[1,1].set_xlabel(fr"target range-wide stability, $S^*$")
handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(handles[::-1], labels[::-1], fontsize=17)
axes[1,1].set_xlim(1, meta_metric_nochange)
xticks = np.arange(0., 1.2, 0.2)
axes[1,1].set_yticks(xticks)
axes[1,1].set_ylim(-0.01, ymax)

# Add colorbar on separate axis
cbar_ax = fig.add_axes([0.2, 0.009, 0.6, 0.025])  # [left, bottom, width, height]
fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal',
    label=r"optimal fraction of populations managed $n~/~n_{tot}$")

fig.savefig(pproc.figs_dir + '/fig3_pre.png', bbox_inches='tight', dpi=dpi)
fig.savefig(pproc.figs_dir + '/fig3_pre.svg', bbox_inches='tight')

#########
# FIG 4 #
#########
xlens = np.array([1.0433, 1.0433]) # left to right
ylens = np.array([2.25/3, 0.2, 0.55, 1])  # top, legend row, mid, bottom
scale = 5
xlens *= scale; ylens *= scale
figsize = np.array([np.sum(xlens), np.sum(ylens)])
#custom_colors = ['lightgrey', 'coral', 'orchid', 'blueviolet']
#custom_colors = ['lightgrey', '#CCBB44', '#EE6778', '#AA3377'] #paul tol bright
#custom_colors = ['lightgrey', '#FFB00D', '#FF5F00', '#DD227D'] #ibm
#custom_colors = ['lightgrey', '#5B8EFD', '#725DEF', '#DD227D'] #ibm 2
#custom_colors = ['lightgrey', '#D55E00', '#009E37', '#AA3377'] #ito 
#custom_colors = ['lightgrey', '#E69F00', '#5CA89A', '#C36a77'] #ito 2
#custom_colors = ['lightgrey', '#e69f00', '#ee6778', '#a04a95'] #pop
#custom_colors = ['lightgrey', '#ff5f00', '#dd227d', '#725def'] #pop2

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
    #if key in ['d', 'e']:
    #    y_pos = bbox.y1 - 0.4
    #else:
    y_pos = bbox.y1 + 0.005

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
#custom_colors_forlegend = ['coral', 'blueviolet', 'orchid', 'lightgrey']
custom_colors_forlegend = np.array(custom_colors)[[1,3,2,0]]
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

# Reinitialize phase instance with samples taken at optimal decisions
constants.update({'extra_attributes': ['maps_filt', 'metric_spl_all']})
pproc = Phase(**constants)
pproc.initialize()
pproc.load_decision_parameters(suffix="_optdecisions")

# Load some other stuff we need
S_opt_baseline = np.load(pproc.data_dir + '/S_opt_baseline.npy')
n_opt_baseline, l_opt_baseline = np.load(pproc.data_dir + '/decision_opt_baseline.npy')
decision_opt_uncertain = np.load(pproc.data_dir + '/decision_opt_uncertain.npy')
n_opt_interp = decision_opt_uncertain[:,0]
l_opt_interp = decision_opt_uncertain[:,1]
rob_all = np.load(pproc.data_dir + "/rob_all.npy")
decision_indices = np.load(pproc.data_dir + '/decision_indices_optdecisions.npy')
tau_sorted = pproc.tau_flat[pproc.tau_argsort_ref]
mapindices = np.argwhere(pproc.maps_filt)

q_vec = np.arange(0.0, 1.0, 0.05)
delta_taul_interp = np.full(q_vec.size, np.nan)
delta_tauh_interp = np.full(q_vec.size, np.nan)
taul_interp = np.full(q_vec.size, np.nan)
tauh_interp = np.full(q_vec.size, np.nan)

for q_i, q in enumerate(q_vec):
    # Now get the optimal decisions for (1-q) * optimal S baseline
    Sstar_i = np.argmin(np.abs(Sstar_vec - ((1 - q) * S_opt_baseline)) )
    n_opt_rob = int(n_opt_interp[Sstar_i])
    l_opt_rob = int(l_opt_interp[Sstar_i])
    
    # Replace this q value with the closest one we have available
    Sstar = Sstar_vec[Sstar_i]
    q_vec[q_i] = 1 - (Sstar / S_opt_baseline)
        
    delta_taul_interp[q_i] = tau_sorted[l_opt_rob] - tau_sorted[l_opt_baseline]
    delta_tauh_interp[q_i] = tau_sorted[l_opt_rob+n_opt_rob] - tau_sorted[l_opt_baseline+n_opt_baseline]
    taul_interp[q_i] = tau_sorted[l_opt_rob]
    tauh_interp[q_i] = tau_sorted[l_opt_rob+n_opt_rob]

# Restrict the range of plotting to a desired q value
q_lim = 0.51
q_mask = q_vec <= q_lim

# Get the points where before and after crossing baseline and color them differently
alpha = 0.85#0.6

# First handle lower bound of optimal tau slice
lte_baseline_q = q_vec[delta_taul_interp[q_mask & (delta_taul_interp <= 0)].argmax() + 1]
lte_baseline_mask = (q_vec < lte_baseline_q)
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

axd['top'].set_xlabel(r'% decrease from baseline $\text{max}(S)$ to target outcome, $S^*$')
axd['top'].set_xlim(-1, 50)
axd['top'].set_ylabel(r'optimal $\hat{\tau}_k$')
axd['top'].legend()

# Define reference indices for per population tau
tau_indices = np.arange(tau_sorted.size)

q_samples = [0.0, 0.5]
for i, q in enumerate(q_samples):
    # Set the S^* value we're plotting
    q_i = np.argmin(np.abs(q_vec - q))
    q = q_vec[q_i]

    # Get the optimal decision at this {S^*, R} combination
    Sstar_rob_i = np.argmin(np.abs(Sstar_vec - ((1 - q) * S_opt_baseline)) )
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

    bins = np.linspace(min(pproc.tau_flat), 50, 80)

    # Plot the stacked histogram
    ax_label = 'mid left' if i == 0 else 'mid right'
    axd[ax_label].hist(
        stack_data,
        bins=bins,
        stacked=True,
        color=custom_colors,
        label=[labels[i] for i in range(len(custom_colors))]
    )

    axd[ax_label].set_xlabel(r'$\hat{\tau}_k$')
    axd[ax_label].set_ylabel(r'$\hat{\tau}_k$ frequency')
    axd[ax_label].set_yticks([])

    ### GEOGRAPHICAL MAP ###

    mapi_sorted = mapindices[pproc.tau_argsort_ref].T

    colored_data = np.ones(pproc.maps_filt.shape + (4,)) * np.nan #colors in rgba
    colored_data[mapi_sorted[0], mapi_sorted[1]] = cmap(norm(results_vector))
    # Color background
    colored_data[pproc.maps_filt == False] = colors.to_rgba('black', alpha=0.3)
    # Crop out border where all nans
    nonzero_indices = np.nonzero(pproc.maps_filt)
    row_min, row_max = nonzero_indices[0].min(), nonzero_indices[0].max()
    col_min, col_max = nonzero_indices[1].min(), nonzero_indices[1].max()
    colored_data = colored_data[row_min:row_max + 1, col_min:col_max + 1]

    ax_label = 'bottom left' if i == 0 else 'bottom right'
    im = axd[ax_label].imshow(colored_data)
    axd[ax_label].set_yticks([])
    axd[ax_label].set_xticks([])

fig.savefig(pproc.figs_dir + '/fig4_pre.png', bbox_inches='tight')

#########
# Conditional probability heatmap #
#########
overwrite_results = True

# Define keys and labels for parameters
uncertain_params = ['tau', 'Deltatau', 'tau_crit']
param_labels = [r'$p_{\tau}$', r'$p_{\Delta\tau}$', r'$\tau^*$']

# Get all possible parameter pairs
param_pairs = [pair for pair in combinations(range(len(uncertain_params)), 2)]

# Read in data run at optimal decisions
phase = h5py.File(pproc.data_dir + '/phase_optdecisions.h5', 'r')
Sstar_i_optdecisions = np.load(pproc.data_dir + '/Sstar_i_optdecisions.npy')

# I don't trust interpolation of robustness between Sstar values,
# so use the closest one we sampled for interpolator fitting
'''This probably isnt necessary'''
Sstar_i_baseline = np.argmin(np.abs(Sstar_vec - S_opt_baseline))
S_opt_baseline = Sstar_vec[Sstar_i_baseline]

# Summarize uncertainty in demography by critical tau values where pops are 'stabilized'
demographic_i_vec = np.arange(0, pproc.num_demographic_samples, 1).astype(int)
tau_crit_vec = np.full(demographic_i_vec.size, np.nan)
tau_samples = np.linspace(pproc.tau_vec[1], pproc.tau_vec.max(), 1_000)
for demographic_i in demographic_i_vec:
    lam_samples = pproc.metric_spl_all[demographic_i](tau_samples)
    if np.any(lam_samples >= metric_thresh):
        if np.all(lam_samples[tau_samples < pproc.tau_vec[3]] > metric_thresh):
            # These are rare cases where lam always greater than threshold
            tau_crit = pproc.tau_vec[0]
        else:
            test_points = pproc.metric_spl_all[demographic_i](pproc.tau_vec[1:4])
            if np.any(np.diff(test_points) < 0):
                # Something weird is happening very rarely at the first tau sample
                tau_samples = np.linspace(pproc.tau_vec[2], pproc.tau_vec.max(), 1_000)
                lam_samples = pproc.metric_spl_all[demographic_i](tau_samples) 
            # Find intersection between line connecting points on either side of metric_thresh
            lam_diffs = lam_samples - metric_thresh
            pos_lam_i = np.nonzero(lam_diffs > 0)[0]
            ydiff = (lam_samples[pos_lam_i[0]+1] - lam_samples[pos_lam_i[0]-1])
            xdiff = (tau_samples[pos_lam_i[0]+1] - tau_samples[pos_lam_i[0]-1])
            slope = ydiff / xdiff
            tau_crit = ((metric_thresh - lam_samples[pos_lam_i[0]-1]) / slope) + tau_samples[pos_lam_i[0]-1]
        tau_crit_vec[demographic_i] = tau_crit
print("ATTENTION: CAPPING tau_crit AT 40 FOR THE SAKE OF PLOTTING")
tau_crit_vec[tau_crit_vec >= 40] = 40

def get_pair_results(C_i, n_i, l_i, Sstar, num_param_bins):
    # Get the slice of range-wide stability at the specified decision parameters
    idx = ".".join(str(i) for i in [C_i, n_i, l_i])
    S_slice = np.array(phase[idx])
    x_uncertain = np.array(phase[idx + 'uncertainty_samples'])

    # Make a temporary version of x_uncertain for summary statistics
    x_uncertain_temp = np.full((x_uncertain.shape[0], len(uncertain_params)), np.nan)

    # Compute robustness
    counts = np.count_nonzero(S_slice >= Sstar)
    robustness = counts / len(S_slice)
    print(f'robustness={robustness}')

    # Replace demographic samples in x_uncertain with our summary
    x_uncertain_temp[:, 2] = tau_crit_vec[x_uncertain[:,4].astype(int)]

    # Just look at p_tau and p_Deltatau
    x_uncertain_temp[:, 0] = x_uncertain[:,0]
    x_uncertain_temp[:, 1] = x_uncertain[:,2]

    # Replace x_uncertain with summary stats version
    x_uncertain = x_uncertain_temp

    # Preallocate filters for samples within bins of each parameter
    bin_filts = {i: np.full((num_param_bins, x_uncertain.shape[0]), False) for i in range(x_uncertain.shape[-1])}
    param_cntrs = {i: np.empty(num_param_bins) for i in range(x_uncertain.shape[-1])}

    for param_i in range(x_uncertain.shape[-1]):
        # Bin the parameter along its sampled range
        if uncertain_params[param_i] == 'tau_crit':
            param_low = np.nanmin(tau_crit_vec)
            param_high = np.nanmax(tau_crit_vec)
        else:
            param_low = x_uncertain[:, param_i].min()
            param_high = x_uncertain[:, param_i].max()
        param_edges, step = np.linspace(param_low, param_high, num_param_bins + 1, retstep=True)

        # Store bin centers for use in plotting
        _param_cntrs = param_edges[:-1] + step/2
        param_cntrs[param_i] = _param_cntrs
        for bin_i, edge in enumerate(param_edges[:-1]):
            _filt = (x_uncertain[:, param_i] > edge) & (x_uncertain[:, param_i] <= edge + step)
            bin_filts[param_i][bin_i] = _filt

    # Initialize matrix to store results in
    results = np.full((len(param_pairs), num_param_bins, num_param_bins), np.nan)

    # Loop over each bin for each parameter pair and compute a statistic on S
    for pair_i, (param_i, param_j) in enumerate(param_pairs):
        # Loop over bin combinations
        bin_combinations = product(range(num_param_bins), range(num_param_bins))
        for i, j in bin_combinations:
            # Make filter for being within each bin
            joint_filt = bin_filts[param_i][i] & bin_filts[param_j][j]

            # Compute fraction of samples at or above Sstar (i.e. probability of target outcome)
            # within each joint parameter bin
            if np.any(joint_filt):
                # Store probability of meeting target outcome
                P_targetmet = np.count_nonzero(S_slice[joint_filt] >= Sstar) / np.count_nonzero(joint_filt)
                results[pair_i, i, j] = P_targetmet

    return results, param_cntrs

# Specify number of bins for each uncertainty parameter
num_param_bins = 13

# Get results under baseline conditions
if overwrite_results:
    all_results = {}
    results, param_cntrs = get_pair_results(0, 0, 0, S_opt_baseline, num_param_bins)
    all_results['baseline'] = results
    all_results['param_cntrs'] = param_cntrs
else:
    with open(pproc.data_dir + '/pair_results.pkl', 'rb') as handle:
        all_results = pickle.load(handle)
        param_cntrs = all_results['param_cntrs']

# Get results at robust optima with max baseline outcome targeted
q_i = 0 #Could select a different q value if desired
C_i = 0 #Assuming only 1 C value was run 
Sstar_i = Sstar_i_optdecisions[q_i]
Sstar = Sstar_vec[Sstar_i]
if overwrite_results:
    results, _ = get_pair_results(C_i, q_i+1, q_i+1, Sstar, num_param_bins)
    all_results['uncertain'] = results

    # Take difference between results
    all_results['uncertain-baseline'] = all_results['uncertain'] - all_results['baseline']

for condition_key in ['baseline', 'uncertain', 'uncertain-baseline']:
    # Plot them
    figdim = np.array([5,4])
    fig, axes = plt.subplots(len(uncertain_params)-1, len(uncertain_params)-1, figsize=figdim*4)

    for pair_i, (param_i, param_j) in enumerate(param_pairs):
        results_pair = all_results[condition_key][pair_i]

        # Get limits of parameters on x and y axes
        x_diff = np.diff(param_cntrs[param_j])[0]
        x_min = param_cntrs[param_j][0] - x_diff/2
        x_max = param_cntrs[param_j][-1] + x_diff/2
        x_bounds = np.array([x_min, x_max])

        y_diff = np.diff(param_cntrs[param_i])[0]
        y_min = param_cntrs[param_i][0] - y_diff/2
        y_max = param_cntrs[param_i][-1] + y_diff/2
        y_bounds = np.array([y_min, y_max])

        if condition_key == 'uncertain-baseline':
            cmap = 'RdPu'
            # Get extreme value of result for colorbar limits
            extreme = max([np.abs(np.nanmin(results_pair)), np.nanmax(results_pair)])
            print("HARDCODING IN COLORBAR LIMIT")
            extreme = 0.4
            #norm = colors.TwoSlopeNorm(vmin=-extreme, vcenter=0, vmax=extreme)
            norm = colors.Normalize(vmin=0, vmax=extreme)
        else:
            cmap = 'viridis'
            norm = colors.Normalize(vmin=np.nanmin(results_pair), vmax=np.nanmax(results_pair))

        if param_i in [0,1]:
            origin = 'upper'
            # Also flip ymin and ymax
            y_bounds = np.flip(y_bounds)
        else:
            origin = 'lower'

        if param_j  in [0,1]:
            results_pair = np.fliplr(results_pair.copy())
            x_axis = np.flip(param_cntrs[param_j])
            # Also flip xmin, xmax
            x_bounds = np.flip(x_bounds)
        else:
            x_axis = param_cntrs[param_j]

        extent = np.concatenate((x_bounds, y_bounds))
        im = axes[param_i,param_j-1].imshow(results_pair, origin=origin, norm=norm, cmap=cmap, extent=extent, aspect='auto')
        axes[param_i, param_j-1].set_box_aspect(1)
        if param_j == len(uncertain_params) - 1:
            label = r'$\Delta P(S \geq S^*)$' if condition_key == 'uncertain-baseline' else r'$P(S \geq S^*)$'
            cbar = fig.colorbar(im, shrink=0.80)
            cbar.set_label(label, fontsize=mpl.rcParams['legend.fontsize']*1.75, labelpad=17)

            # Plot vertical line at critical tau under baseline
            '''Hardcoded for now'''
            axes[param_i, param_j-1].axvline(32.895, ls='--', c='k')
        tick_spacing = 1 if num_param_bins <= 5 else 3
        numround = 1 if param_j-1 == 3 else 2
        axes[param_i,param_j-1].set_xticks(x_axis[::tick_spacing],
                                           np.round(x_axis, numround)[::tick_spacing])
        axes[param_i,param_j-1].set_yticks(param_cntrs[param_i][::tick_spacing], 
                                           np.round(param_cntrs[param_i], 2)[::tick_spacing])
        axes[param_i,param_j-1].tick_params(axis='both', labelsize=plt.rcParams['axes.labelsize'] * 1)
        if param_i == param_j - 1:
            axes[param_i,param_j-1].set_xlabel(param_labels[param_j], fontsize=plt.rcParams['axes.titlesize']*2.5)
            axes[param_i,param_j-1].set_ylabel(param_labels[param_i], fontsize=plt.rcParams['axes.titlesize']*2.5)

    for param_i, param_j in [[1,0]]:
        axes[param_i, param_j].set_axis_off()

    if (q_i == 0) and (condition_key == 'uncertain-baseline'): 
        fig.savefig(pproc.figs_dir + f'/uncertainty_pairs_{condition_key}_{np.round(q_vec[q_i],2)}.png', bbox_inches='tight', dpi=dpi)
    plt.close(fig)

with open(pproc.data_dir + '/pair_results.pkl', 'wb') as handle:
    pickle.dump(all_results, handle)

phase.close()
