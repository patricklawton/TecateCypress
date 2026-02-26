from project import Phase
import numpy as np
from matplotlib import colors, cm, rc
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Polygon
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
from shapely.geometry import box
import cartopy.feature as cfeature
from pyproj import Transformer
import pickle
import h5py
from scipy.special import gamma
from scipy.interpolate import make_splrep
from scipy.optimize import root_scalar
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
rc('font', size=19)
histlw = 5.5
cbar_lpad = 30
#dpi = 50
dpi = 300
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

fig = plt.figure(figsize=np.array([2.6, 1])*11)
nrows = 8
height_ratios = np.ones(nrows)
height_ratios[0] = 0.75
height_ratios[7] = 0.75
gs = GridSpec(nrows=nrows, ncols=3, figure=fig, width_ratios=[0.3,0.3,0.4], 
              height_ratios=height_ratios)
popdyn_ax = fig.add_subplot(gs[2:6, 0])
fmanag_ax = fig.add_subplot(gs[3:6, 1]) 
robdec_ax = fig.add_subplot(gs[2:6, 2])
legend_ax = fig.add_subplot(gs[1, 2])
topgap_ax = fig.add_subplot(gs[0, :])
topgap_ax.axis('off')
if nrows > 6:
    botgap_ax = fig.add_subplot(gs[6:, :])
    botgap_ax.axis('off')

# Make legend handles
custom_colors_forlegend = np.array(custom_colors)[[1,3,2]]
legend_labels = ['baseline only', 'uncertain only', 'both']
handles = [Patch(facecolor=c, edgecolor='black', label=l)
           for c, l in zip(custom_colors_forlegend, legend_labels)]

# Place legend in the 'legend' subplot
legend_ax.axis('off')
legend_ax.legend(
    handles=handles,
    loc='center',
    frameon=False,
    ncol=len(handles),   # all in one row
    handlelength=2.,
    title=r'conditions where $\hat{\tau}_k$ optimal for management',
    title_fontsize=mpl.rcParams['axes.labelsize']*1.15,
    fontsize=mpl.rcParams['axes.labelsize']*0.9
)

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
im = popdyn_ax.hist2d(tau_plot_all, metric_interp_all, bins=[tau_edges, metric_edges],
                norm=norm,
                density=False,
                cmap=cmap)

# Plot interpolation function for <metric> wrt tau
#metric_spl = pproc.metric_spl_all[0]
#tau_samples = np.arange(0, 140, 2)
#popdyn_ax.plot(tau_samples, metric_spl(tau_samples), color='k',
#         label=r'baseline: $\hat{\lambda}(\tau_k)$')

# Add the colorbar to inset axis
cbar_ax = inset_axes(popdyn_ax, width="5%", height="50%", loc='center',
                     bbox_to_anchor=(0.25, -0.15, 0.55, 0.9), #x,y,w,h
                     bbox_transform=popdyn_ax.transAxes, borderpad=0)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#ticks = np.array([1,10, 100, 500]) * 10
#ticklabels = np.round(ticks/pproc.num_demographic_samples, 3)
ticks = []
ticklabels = []
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical", ticks=ticks,
                   format=mticker.FixedFormatter(ticklabels))

# Labels etc
cbar.set_label(rf'$P(\lambda|\tau_k)$', rotation=-90, labelpad=cbar_lpad)
popdyn_ax.set_ylabel(rf'growth rate, $\lambda(\tau_k)$')
#popdyn_ax.legend(bbox_to_anchor=(0.1, -0.05, 0.5, 0.5), fontsize=24)
popdyn_ax.set_ylim(metric_edges[np.nonzero(im[0][min_edge_i])[0].min()], max(metric_edges))
popdyn_ax.set_xlim(tau_edges[min_edge_i], max(tau_edges))
popdyn_ax.set_xlabel(r'fire return interval, $\tau_k$')

#### FIRE REGIME SHIFT EXAMPLES ####
tau_edges = np.arange(0, int(max(tau_edges)+0.5)+1, 1)
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
nodes = [0.0, 0.6, 0.8, 1.0]
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

    fmanag_ax.hist(pproc.tau_flat[(pproc.tau_flat >= tau_edges[bin_i]) & (pproc.tau_flat < tau_edges[bin_f])],
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
    fmanag_ax.hist(
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

    fmanag_ax.hist(tau_shifted[mask,...], bins=tau_edges, color='white', density=density);
    fmanag_ax.hist(tau_current, bins=tau_edges, color='black', histtype='step', lw=histlw, density=density);

    fmanag_ax.set_yticks([])
    fmanag_ax.set_ylabel(r"$\tau_k$ frequency")
    fmanag_ax.set_xticks(np.arange(20,100,20).astype(int))
    fmanag_ax.set_xlabel(r"fire return interval, $\tau_k$")
    fmanag_ax.set_xlim(15, xmax);

#### Robust optima ####

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
Sstar_vec = np.load(pproc.data_dir + "/Sstar_vec.npy")

q_vec = np.arange(0.0, 1.0, 0.05)
delta_taul_interp = np.full(q_vec.size, np.nan)
delta_tauh_interp = np.full(q_vec.size, np.nan)
taul_interp = np.full(q_vec.size, np.nan)
tauh_interp = np.full(q_vec.size, np.nan)
tau_range = [18, 40]

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
q_lim = 0.5

# Get the points where before and after crossing baseline and color them differently
alpha = 0.85

# Interpolate optimal tau over decreasing Sstar
s=len(q_vec)-np.sqrt(2*len(q_vec)) #recommended by scipy doc
tauh_spl = make_splrep(q_vec, tauh_interp, s=s)
taul_spl = make_splrep(q_vec, taul_interp, s=s)

# Find intersections with baseline optima
tauh_baseline = tau_sorted[l_opt_baseline + n_opt_baseline]
qcrit_h = root_scalar(lambda x: tauh_baseline - tauh_spl(x), bracket=[0.1,0.9]).root
taul_baseline = tau_sorted[l_opt_baseline]
qcrit_l = root_scalar(lambda x: taul_baseline - taul_spl(x), bracket=[0.1,0.9]).root

# Make shared, resampled q axis
q1 = np.linspace(0, qcrit_l, 100)
q2 = np.linspace(qcrit_l, qcrit_h, 30)
q3 = np.linspace(qcrit_h, q_lim, 100)
q_new = np.concatenate((q1, q2, q3))

# First handle lower bound of optimal tau slice
# Handle less than baseline
q_sub = q_new[q_new < qcrit_l]
y2 = np.full(len(q_sub), taul_baseline)
robdec_ax.fill_between( 
    q_sub*100, taul_spl(q_sub), y2,
    color=custom_colors[3],
    alpha=alpha,
    zorder=-1
)
# Handle greater than baseline
q_sub = q_new[q_new >= qcrit_l]
y1 = np.full(len(q_sub), taul_baseline)
robdec_ax.fill_between( 
    q_sub*100, y1, taul_spl(q_sub),
    color=custom_colors[1],
    alpha=alpha,
    zorder=-1
)
# Combine things to fill middle later
y_l = np.concatenate((y2, taul_spl(q_sub)))

# Now handle upper bound of optimal tau slice
# Handle less than baseline
q_sub1 = q_new[q_new < qcrit_h]
y2 = np.full(len(q_sub1), tauh_baseline)
robdec_ax.fill_between( 
    q_sub1*100, tauh_spl(q_sub1), y2,
    color=custom_colors[1],
    alpha=alpha,
    zorder=-1
)
# Handle greater than baseline
q_sub2 = q_new[q_new >= qcrit_h]
y1 = np.full(len(q_sub2), tauh_baseline)
robdec_ax.fill_between( 
    q_sub2*100, y1, tauh_spl(q_sub2),
    color=custom_colors[3],
    alpha=alpha,
    zorder=-1
)
# Combine things to fill middle later
y_h = np.concatenate((tauh_spl(q_sub1), y1))

# Finally, fill in the middle (both) section
robdec_ax.fill_between(
    q_new*100, y_h, y_l,
    color=custom_colors[2],
    alpha=alpha,
    zorder=-1
)

#robdec_ax.set_xlabel(r'target range-wide stability, $S^*$')
robdec_ax.set_xlim(-1, 51)
#xticks = np.arange(0, 0.6, 0.1)
#xtick_labels = np.round(S_opt_baseline * (1 - xticks), 2)
xticks = []
xtick_labels = []
robdec_ax.set_xticks(xticks, labels=xtick_labels)
robdec_ax.set_ylabel(r'$\hat{\tau}_k$')
robdec_ax.set_ylim(tau_range[0], tau_range[1])

fig.savefig(pproc.figs_dir + '/graphicalabstract_pre.png', bbox_inches='tight', dpi=dpi)
