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
from scipy.stats import moment

overwrite_observations = False
overwrite_simulations = False
overwrite_posterior = False
add_simulations = True

processes = ['mortality']
for pr in processes:
    if pr == 'mortality':
        from mortality.simulator import simulator
        labels = ['alph_m', 'beta_m', 'sigm_m', 'gamm_nu'] 
        ranges = np.array([
                           # alph_m
                           [0.01, 0.6], 
                           # beta_m
                           [0.01, 0.9], 
                           # sigm_m
                           [0.1,1.7], 
                           ## alph_nu
                           #[0.01,2.]#,
                           ## beta_nu
                           #[0.01,0.9]
                           # gamm_nu
                           [0.001, 0.8],
                           ## K_seedling
                           #[10_000, 120_000],
                           ## kappa
                           #[0.01, 1.5]
                           ## K_adult
                           #[8000,30000]
        ])
        restrictor_sims = 5_000
        training_sims = 30_000
        num_samples = 1_000_000 
    elif pr == 'fecundity':
        from fecundity.simulator import simulator, save_observations
        labels = ['rho_max', 'eta_rho', 'a_mature', 'sigm_max', 'eta_sigm']#, 'a_sigm_star']
        ranges = np.array([
                           [100, 600],
                           [0.01, 0.8],
                           [15, 80],
                           [0.01, 5],
                           [0.01, 0.8]
        ])
        restrictor_sims = 20_000
        training_sims = 20_000
        num_samples = 1_000_000 

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
        observations = np.concatenate((m1,m2,m3))
        np.save(fn, observations)
    elif ((not os.path.isfile(fn)) or overwrite_observations) and (pr=='fecundity'):
        save_observations()
    x_o = np.load(fn, allow_pickle=True)
    x_o = torch.Tensor(x_o)

    if (not os.path.isfile(pr+'/all_theta.pkl')) or overwrite_simulations:
        priors = [Uniform(tensor([rng[0]]), tensor([rng[1]])) for rng in ranges]
        prior = MultipleIndependent(priors)
        prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
        with open(pr+"/prior.pkl", "wb") as handle:
            pickle.dump(prior, handle)
        simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)
        theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=restrictor_sims, num_workers=8)
        restriction_estimator = RestrictionEstimator(prior=prior)
        restriction_estimator.append_simulations(theta, x)
        classifier = restriction_estimator.train()
        restricted_prior = restriction_estimator.restrict_prior()
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
