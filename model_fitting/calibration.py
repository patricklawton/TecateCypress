import torch
import numpy as np
import pickle
import sbi
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi.diagnostics import check_sbc, run_sbc
from sbi import utils as utils
from sbi import analysis as analysis
from torch.distributions import Uniform
from sbi.utils import MultipleIndependent
from torch import tensor
import json

processes = ['mortality']
for pr in processes:
    if pr == 'mortality':
        from mortality.simulator import simulator, fixed
        labels = ['alph_m', 'beta_m', 'sigm_m', 'K_seedling']#,'alph_nu', 'gamm_m']
    elif pr == 'fecundity':
        from fecundity.simulator import simulator
        fixed = {}
        labels = ['rho_max', 'eta_rho', 'a_mature', 'sigm_max', 'eta_sigm']
    with open(pr+"/prior.pkl", "rb") as handle:
        prior = pickle.load(handle)
    simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)

    with open(pr+"/posterior.pkl", "rb") as handle:
        posterior = pickle.load(handle)

    ### map
    map_dict = {}
    map_dict.update(fixed)
    _map = posterior.map(force_update=True).numpy()[0]
    for lab, val in zip(labels, _map):
        map_dict.update({lab: val})
    with open(pr+'/map.json', 'w') as handle:
        json.dump(map_dict, handle)

    num_simulations = 5_000  # choose a number of sbc runs, should be ~100s or ideally 1000
    # generate ground truth parameters and corresponding simulated observations for SBC.
    thetas, xs = simulate_for_sbi(simulator, proposal=prior, num_simulations=num_simulations, num_workers=8)
    # filter out invalid parameter samples
    thetas = thetas[~torch.any(xs.isnan(),dim=1)]
    xs = xs[~torch.any(xs.isnan(),dim=1)]

    num_posterior_samples = 100_000
    ranks, dap_samples = run_sbc(
        thetas, xs, posterior, num_posterior_samples=num_posterior_samples,
        num_workers=8, reduce_fns='marginals'#posterior.log_prob,
    )
    with open(pr+'/ranks.pkl', 'wb') as handle:
        pickle.dump(ranks, handle)
    check_stats = check_sbc(
        ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples
    )
    with open(pr+'/sbc_stats.pkl', 'wb') as handle:
        pickle.dump(check_stats, handle)
    with open(pr+'/sbc-log.txt', 'w') as handle:
        handle.write(
            f"""kolmogorov-smirnov p-values \n
            check_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()} \n"""
        )
        handle.write(
            f"c2st accuracies \ncheck_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()} \n"
        )
        handle.write(f"check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()} \n")

    f, ax = analysis.plot.sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=num_posterior_samples,
        plot_type="hist",
        parameter_labels=labels,
        num_bins=None,  # by passing None we use a heuristic for the number of bins.
    )
    f.savefig('sbi_figs/{}_rank_hist.png'.format(pr), bbox_inches='tight')

    f, ax = analysis.plot.sbc_rank_plot(
        ranks=ranks, 
        num_posterior_samples=num_posterior_samples, 
        plot_type="cdf",
        parameter_labels=labels
    )
    f.savefig('sbi_figs/{}_rank_cdf.png'.format(pr), bbox_inches='tight')

    ### ppc
    x_o = np.load(pr+'/observations/observations.npy')
    x_o = torch.Tensor(x_o)
    posterior_samples = posterior.sample(sample_shape=(5_000,))
    x_pp = simulator(posterior_samples)
    x_pp = x_pp[~torch.any(x_pp.isnan(),dim=1)]
    with open(pr+'/x_pp.pkl', 'wb') as handle:
        pickle.dump(x_pp, handle)
    mins = x_pp.amin(0)
    maxes = x_pp.amax(0)
    limits = torch.tensor([[mins[i], maxes[i]] for i in range(x_pp.shape[1])])
    if pr == 'mortality':
        limits[9][0] = -0.0005; limits[9][1] = 0.001
        limits[12][0] = -0.0005; limits[12][1] = 0.0008
        limits[13][0] = -0.00005; limits[13][1] = 0.00015
        limits[14][0] = -0.00005; limits[14][1] = 0.0001
    elif pr == 'fecundity':
        limits[0][0] = -0.0005; limits[0][1] = 10
        limits[1][0] = -0.0005; limits[1][1] = 250
        limits[2][0] = -0.0005; limits[2][1] = 2500
        limits[3][0] = -0.0005; limits[3][1] = 10
        limits[4][0] = -0.0005; limits[4][1] = 250
        limits[5][0] = -0.0005; limits[5][1] = 2500
        limits[6][0] = -0.0005; limits[6][1] = 1
    ppc = analysis.pairplot(
        samples=x_pp,
        points=x_o, 
        points_offdiag=dict(marker="+", markersize=10),
        limits=limits,
        figsize=(16,16)
    )
    ppc[0].savefig('sbi_figs/{}_npe_ppc.png'.format(pr), bbox_inches='tight')
