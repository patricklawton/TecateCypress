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
from simulator import simulator

with open("prior.pkl", "rb") as handle:
    prior = pickle.load(handle)
#prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)


with open("posterior.pkl", "rb") as handle:
    posterior = pickle.load(handle)

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
with open('ranks.pkl', 'wb') as handle:
    pickle.dump(ranks, handle)
check_stats = check_sbc(
    ranks, thetas, dap_samples, num_posterior_samples=num_posterior_samples
)
with open('sbc_stats.pkl', 'wb') as handle:
    pickle.dump(check_stats, handle)
with open('sbc-log.txt', 'w') as handle:
    handle.write(
        f"""kolmogorov-smirnov p-values \n
        check_stats['ks_pvals'] = {check_stats['ks_pvals'].numpy()} \n"""
    )
    handle.write(
        f"c2st accuracies \ncheck_stats['c2st_ranks'] = {check_stats['c2st_ranks'].numpy()} \n"
    )
    handle.write(f"check_stats['c2st_dap'] = {check_stats['c2st_dap'].numpy()} \n")

labels = ['alph_m', 'beta_m', 'sigm_m','alph_nu']
f, ax = analysis.plot.sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=num_posterior_samples,
    plot_type="hist",
    parameter_labels=labels,
    num_bins=None,  # by passing None we use a heuristic for the number of bins.
)
f.savefig('sbi_figs/rank_hist.png', bbox_inches='tight')

f, ax = analysis.plot.sbc_rank_plot(
    ranks=ranks, 
    num_posterior_samples=num_posterior_samples, 
    plot_type="cdf",
    parameter_labels=labels
)
f.savefig('sbi_figs/rank_cdf.png', bbox_inches='tight')
