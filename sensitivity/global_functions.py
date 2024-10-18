import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import copy

# Define line function to be used for fitting later
def line(x, m, b):
    return m*x + b

def adjustmaps(maps):
    dim_len = []
    for dim in range(2):
        dim_len.append(min([m.shape[dim] for m in maps]))
    for mi, m in enumerate(maps):
        maps[mi] = m[0:dim_len[0], 0:dim_len[1]]
    return maps

def plot_phase(phase_space, metric, metric_nochange, fri_bin_cntrs, n_cell_vec, fig_fn, C, fric_vec=None):
    fig, ax = plt.subplots(figsize=(12,12))
    ncell_tot = 87_993 #should read this in
    axfontsize = 16
    metric_labels = ['$<r>$', '$<\mu>$', '$<\lambda>$', '$<xs>$']
    metrics = np.array(['r', 'mu_s', 'lambda_s', 'xs'])
    metric_i = np.nonzero(metrics == metric)[0][0]
    metric_lab = metric_labels[metric_i]
    #phase_space = np.ma.masked_where(phase_space==0, phase_space)
    phase_space = np.ma.masked_where(np.isnan(phase_space),  phase_space)
    #print(f"phase_space from plot_phase: {phase_space}")
    phase_flat = phase_space.flatten()
    cmap = copy.copy(matplotlib.cm.plasma)
    if len(phase_flat[phase_flat != np.ma.masked]) == 0:
        phase_max = 0
    else:
        '''doing this for now bc some runs are bad'''
        if (metric=='r') or (metric=='g'):
            phase_max = np.quantile(phase_flat[phase_flat != np.ma.masked], 0.98)
        if metric in ['Nf', 'xs', 'mu_s', 'lambda_s']:
            phase_max = max(phase_flat[phase_flat != np.ma.masked])
    cmap.set_bad('white')
    im = ax.imshow(phase_space, norm=matplotlib.colors.Normalize(vmin=metric_nochange, vmax=phase_max), cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, location="right", shrink=0.6)
    cbar.ax.set_ylabel(fr'{metric_lab}', rotation=-90, fontsize=axfontsize, labelpad=20)
    ytick_spacing = 2
    ytick_labels = np.flip(fri_bin_cntrs)[::ytick_spacing]
    yticks = np.arange(0,len(fri_bin_cntrs),ytick_spacing)
    ax.set_yticks(yticks, labels=np.round(ytick_labels, decimals=3));
    ax.set_ylabel(fr'Average initial $\tau$ in area where $\tau$ is changed', fontsize=axfontsize)
    xtick_spacing = 3
    xticks = np.arange(0,len(n_cell_vec),xtick_spacing)
    xtick_labels = np.round(n_cell_vec/ncell_tot, 2)
    ax.set_xticks(xticks, labels=xtick_labels[::xtick_spacing]);
    ax.set_xlabel(r'Fraction of species range where $\tau$ is altered ($A/A_{\text{range}}$)', fontsize=axfontsize)
    #if fric_vec != None:
    if hasattr(fric_vec, "__len__"):
        secax = ax.secondary_xaxis('top')
        secax.set_xticks(xticks, labels=np.round(fric_vec[::xtick_spacing], decimals=3));
        secax.set_xlabel(r'Change in $\tau$ per unit area ($\hat{\tau}$)', fontsize=axfontsize)
    fig.savefig(fig_fn, bbox_inches='tight')
    plt.close(fig)
