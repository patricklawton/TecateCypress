import torch
import numpy as np
import sbi
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi import utils as utils
from sbi import analysis as analysis
from torch.distributions import Uniform
from sbi.utils import MultipleIndependent, RestrictionEstimator
from torch import tensor
import pickle
import pandas as pd
import os
from simulator import simulator
from scipy.stats import moment

overwrite_observations = False
overwrite_simulations = False
add_simulations = True
overwrite_posterior = False

if (os.path.isfile('observations/observations.npy') == False) or overwrite_observations:
    # First, compute and store summary statistics of observed data
    census_yrs = [2,6,8,11,14]
    fn = 'observations/mortality.csv'
    mortality_o = pd.read_csv(fn, header=None)
    mortality_o[0] = [round(v) for v in mortality_o[0]]
    # Use the first 3 moments
    m1 = []; m2 = []; m3 = []
    for t_i, t in enumerate(census_yrs):
        mort_sub = mortality_o[mortality_o[0]==t][1].to_numpy()
        est_mean = np.mean(mort_sub)
        m1.append(est_mean)
        m2.append(moment(mort_sub, moment=2))
        m3.append(moment(mort_sub, moment=3))
    observations = np.concatenate((m1,m2,m3))
    np.save('observations/observations.npy', observations)
x_o = np.load('observations/observations.npy')
x_o = torch.Tensor(x_o)

ranges = np.array([
                   # alph_m
                   [0.01, 0.6], 
                   # beta_m
                   [0.01, 0.9], 
                   # sigm_m
                   [0.1,1.7], 
                   # alph_nu
                   [0.01,2.]
])
if (os.path.isfile('all_theta.pkl') == False) or overwrite_simulations:
    priors = [Uniform(tensor([rng[0]]), tensor([rng[1]])) for rng in ranges]
    prior = MultipleIndependent(priors)
    prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
    with open("prior.pkl", "wb") as handle:
        pickle.dump(prior, handle)
    simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=50000, num_workers=8)
    restriction_estimator = RestrictionEstimator(prior=prior)
    restriction_estimator.append_simulations(theta, x)
    classifier = restriction_estimator.train()
    restricted_prior = restriction_estimator.restrict_prior()
    with open("restricted_prior.pkl", "wb") as handle:
        pickle.dump(restricted_prior, handle)
    new_theta, new_x = simulate_for_sbi(simulator, restricted_prior, 150000)
    restriction_estimator.append_simulations(
        new_theta, new_x
    )  # Gather the new simulations in the `restriction_estimator`.
    (
        all_theta,
        all_x,
        _,
    ) = restriction_estimator.get_simulations()  # Get all simulations run so far.
    with open("all_theta.pkl", "wb") as handle:
        pickle.dump(all_theta, handle)
    with open("all_x.pkl", "wb") as handle:
        pickle.dump(all_x, handle)
else:
    with open("prior.pkl", "rb") as handle:
        prior = pickle.load(handle)
    with open("restricted_prior.pkl", "rb") as handle:
        restricted_prior = pickle.load(handle)
    with open("all_theta.pkl", "rb") as handle:
        all_theta = pickle.load(handle)
    with open("all_x.pkl", "rb") as handle:
        all_x = pickle.load(handle)
    if add_simulations:
        simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)
        new_theta, new_x = simulate_for_sbi(simulator, restricted_prior, 100_000, num_workers=8)
        all_theta = torch.cat((all_theta, new_theta), 0)
        all_x = torch.cat((all_x, new_x), 0)
        with open("all_theta.pkl", "wb") as handle:
            pickle.dump(all_theta, handle)
        with open("all_x.pkl", "wb") as handle:
            pickle.dump(all_x, handle)

if (os.path.isfile('posterior.pkl') == False) or overwrite_posterior or add_simulations:
    inferer = SNPE(prior, show_progress_bars=True, density_estimator="mdn")
    inferer = inferer.append_simulations(all_theta, all_x)
    density_estimator = inferer.train()
    posterior = inferer.build_posterior(density_estimator)
    posterior.set_default_x(x_o)
    with open("posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)
else:
    with open("posterior.pkl", "rb") as handle:
        posterior = pickle.load(handle)

num_samples = 1_000_000
npe_samples = posterior.sample(sample_shape=(num_samples,))
torch.save(npe_samples, 'posterior_samples.pkl')
labels = ['alph_m', 'beta_m', 'sigm_m','alph_nu']
_ = analysis.pairplot(
    npe_samples, limits=ranges, figsize=(10, 10), labels=labels
)
_[0].savefig('figs/npe_test.png', bbox_inches='tight')
