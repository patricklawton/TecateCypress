import torch
import numpy as np
import sbi
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi import utils as utils
from sbi import analysis as analysis
from torch.distributions import Uniform, Distribution
from sbi.utils import MultipleIndependent, RestrictionEstimator
from torch import tensor
import pickle
import pandas as pd
import os
from scipy.stats import moment

overwrite_observations = True
overwrite_simulations = True
overwrite_posterior = True
add_simulations = False

processes = ['fecundity']
for pr in processes:
    if pr == 'mortality':
        from mortality.simulator import simulator, h_o
        labels = ['alph_m', 'beta_m', 'sigm_m'] 
        ranges = np.array([
                           # alph_m
                           [0.01, 0.7], 
                           # beta_m
                           [0.01, 0.9], 
                           # sigm_m
                           [0.1,1.7],
                           ## alph_nu
                           #[0.01,2.]#,
                           ## beta_nu
                           #[0.01,0.9]
                           ## gamm_nu
                           #[0.001, 0.8],
                           ## K_seedling
                           #[10_000, 120_000],
                           ## kappa
                           #[0.01, 2.5],
                           ## K_adult
                           #[(0.1/h_o)*10_000, (3/h_o)*10_000]
        ])
        restrictor_sims = 10_000
        training_sims = 20_000
        num_samples = 1_000_000 
        allowed_false_negatives = 0.0
    elif pr == 'fecundity':
        from fecundity.simulator import simulator, save_observations
        labels = ['rho_max', 'eta_rho', 'a_mature', 'sigm_max']#, 'eta_sigm']#, 'a_sigm_star']
        #labels = ['eta_rho', 'a_mature', 'sigm_max']
        ranges = np.array([
                           [10, 600],
                           [0.01, 0.8],
                           [15, 80],
                           [0.01, 6]
                           #[0.01, 0.8]
        ])
        restrictor_sims = 20_000
        training_sims = 20_000
        num_samples = 1_000_000 
        allowed_false_negatives = 0.1

    with open(pr+"/param_labels.pkl", "wb") as handle:
        pickle.dump(labels, handle)

    # First, compute and store summary statistics of observed data
    fn = pr + '/observations/observations.npy'
    if ((not os.path.isfile(fn)) or overwrite_observations) and (pr=='mortality'):
        census_yrs = [2,6,8,11,14]
        #fn = pr + '/observations/mortality.csv'
        mortality_o = pd.read_csv(pr + '/observations/mortality.csv', header=None)
        mortality_o[0] = [round(v) for v in mortality_o[0]]
        # Use the first 3 moments
        m1 = []; m2 = []; m3 = []
        for t_i, t in enumerate(census_yrs):
            mort_sub = mortality_o[mortality_o[0]==t][1].to_numpy()
            est_mean = np.mean(mort_sub)
            m1.append(est_mean)
            m2.append(moment(mort_sub, moment=2))
            m3.append(moment(mort_sub, moment=3))
        observations = np.concatenate((m1,m2))
        np.save(fn, observations)
    elif ((not os.path.isfile(fn)) or overwrite_observations) and (pr=='fecundity'):
        save_observations()
    x_o = np.load(fn, allow_pickle=True)
    x_o = torch.Tensor(x_o)

    if (not os.path.isfile(pr+'/all_theta.pkl')) or overwrite_simulations:
        if pr == 'mortality':
            priors = [Uniform(tensor([rng[0]]), tensor([rng[1]])) for rng in ranges]
            if len(priors) > 1:
                prior = MultipleIndependent(priors)
            else:
                prior = priors[0]
        elif pr == 'fecundity':
            custom_bin_edges = np.load(pr + '/observations/custom_bin_edges.npy')
            a_star_target = 20
            a_star_i = np.argmin(np.abs(a_star_target - custom_bin_edges))
            a_star = custom_bin_edges[a_star_i]
            class CustomPrior(Distribution):
                def __init__(self):
                    super().__init__()
                    self.base_dist = Uniform(tensor([rng[0] for rng in ranges]), tensor([rng[1] for rng in ranges]))

                def sample(self, sample_shape=torch.Size()):
                    num_samples = sample_shape[0] if len(sample_shape) > 0 else 1
                    samples = []
                    while len(samples) < num_samples:
                        theta = self.base_dist.sample()
                        constraint_check = theta[2] - ((1 / theta[3]) * torch.log(torch.exp(theta[1]**2 / 2) * theta[0] - 0.25))
                        #constraint_check = theta[2] - ((1 / theta[3]) * torch.log(theta[0] - 1))
                        #if theta[0] + theta[1] < 10:  # Constraint example: θ₁ + θ₂ < 10
                        if a_star <= constraint_check:
                            samples.append(theta)
                    if len(sample_shape) == 0:
                        return samples[0]
                    else:
                        return torch.stack(samples)

                def log_prob(self, theta):
                    sample_shape = theta.shape
                    # Return -inf if constraint is violated, otherwise use base log prob
                    if len(sample_shape) == 1:
                        constraint_check = theta[2] - ((1 / theta[3]) * torch.log(torch.exp(theta[1]**2 / 2) * theta[0] - 0.25))
                        #constraint_check = theta[2] - ((1 / theta[3]) * torch.log(theta[0] - 1))
                        success = (a_star <= constraint_check)
                        if success:
                            log_prob = self.base_dist.log_prob(theta).sum(dim=-1)  # Sum over dimensions
                        else:
                            log_prob = -float("inf")  # Enforce constraint
                        return tensor([log_prob])
                    else:
                        constraint_check = theta[:, 2] - ((1 / theta[:, 3]) * torch.log(torch.exp(theta[:, 1]**2 / 2) * theta[:, 0] - 1))
                        mask = (a_star <= constraint_check)
                        log_prob = self.base_dist.log_prob(theta).sum(dim=-1)  # Sum over dimensions
                        log_prob[~mask] = -float("inf")  # Enforce constraint
                        return log_prob
            prior = CustomPrior()
        prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
        with open(pr+"/prior.pkl", "wb") as handle:
            pickle.dump(prior, handle)
        simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=restrictor_sims, num_workers=8)
        restriction_estimator = RestrictionEstimator(prior=prior)
        restriction_estimator.append_simulations(theta, x)
        classifier = restriction_estimator.train()
        restricted_prior = restriction_estimator.restrict_prior(allowed_false_negatives=allowed_false_negatives)
        with open(pr+"/restricted_prior.pkl", "wb") as handle:
            pickle.dump(restricted_prior, handle)
        new_theta, new_x = simulate_for_sbi(simulator, restricted_prior, training_sims, num_workers=8)
        restriction_estimator.append_simulations(
            new_theta, new_x
        )  # Gather the new simulations in the `restriction_estimator`.
        (
            all_theta,
            all_x,
            _,
        ) = restriction_estimator.get_simulations()  # Get all simulations run so far.
        with open(pr+"/all_theta.pkl", "wb") as handle:
            pickle.dump(all_theta, handle)
        with open(pr+"/all_x.pkl", "wb") as handle:
            pickle.dump(all_x, handle)
    else:
        with open(pr+"/prior.pkl", "rb") as handle:
            prior = pickle.load(handle)
        with open(pr+"/all_theta.pkl", "rb") as handle:
            all_theta = pickle.load(handle)
        with open(pr+"/all_x.pkl", "rb") as handle:
            all_x = pickle.load(handle)
        if add_simulations:
            simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)
            with open(pr+"/restricted_prior.pkl", "rb") as handle:
                restricted_prior = pickle.load(handle)
            new_theta, new_x = simulate_for_sbi(simulator, restricted_prior, 100_000, num_workers=8)
            all_theta = torch.cat((all_theta, new_theta), 0)
            all_x = torch.cat((all_x, new_x), 0)
            with open(pr+"/all_theta.pkl", "wb") as handle:
                pickle.dump(all_theta, handle)
            with open(pr+"/all_x.pkl", "wb") as handle:
                pickle.dump(all_x, handle)

    if (not os.path.isfile(pr+'/posterior.pkl')) or overwrite_posterior or add_simulations:
        inferer = SNPE(prior, show_progress_bars=True, density_estimator="mdn")
        inferer = inferer.append_simulations(all_theta, all_x)
        density_estimator = inferer.train()
        posterior = inferer.build_posterior(density_estimator)
        posterior.set_default_x(x_o)
        with open(pr+"/posterior.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
    else:
        with open(pr+"/posterior.pkl", "rb") as handle:
            posterior = pickle.load(handle)

    npe_samples = posterior.sample(sample_shape=(num_samples,))
    torch.save(npe_samples, pr+'/posterior_samples.pkl')
    _ = analysis.pairplot(
        npe_samples, limits=ranges, figsize=(10, 10), labels=labels
    )
    _[0].savefig('sbi_figs/{}_posterior_pairplot.png'.format(pr), bbox_inches='tight')
