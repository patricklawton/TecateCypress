import torch
import numpy as np
import sbi
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi import utils as utils
from sbi import analysis as analysis
from torch.distributions import Uniform
from sbi.utils import MultipleIndependent
from torch import tensor
import pickle
import pandas as pd
import os
from simulator import simulator

overwrite_estimator = True

defaults = np.array([0.2, 0.8, 0.45])
ranges = np.array([[0.01, 0.6], [0.1,0.9], [0.05,0.95]])
priors = [
    # alph_m
    Uniform(tensor([ranges[0][0]]), tensor([ranges[0][1]])),
    # sigm_m
    Uniform(tensor([ranges[1][0]]), tensor([ranges[1][1]])),
    # alph_nu
    Uniform(tensor([ranges[2][0]]), tensor([ranges[2][1]])),
]
prior = MultipleIndependent(priors)

prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)

if (os.path.isfile('likelihood_estimator.pkl') == False) or overwrite_estimator:
    inferer = SNLE(prior, show_progress_bars=True, density_estimator="mdn")
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=2000)
    inferer = inferer.append_simulations(theta, x)
    likelihood_estimator = inferer.train()
    # Write likelihood estimator to file
    with open("likelihood_estimator.pkl", "wb") as handle:
        pickle.dump(likelihood_estimator, handle)
else:
    # Read likelihood estimator from file
    with open("likelihood_estimator.pkl", "rb") as handle:
        likelihood_estimator = pickle.load(handle)

x_o = np.load('observations/observations.npy')
x_o = torch.Tensor(x_o)
potential_fn, parameter_transform = likelihood_estimator_based_potential(
    likelihood_estimator, prior, x_o
)

mcmc_parameters = dict(
    method = "slice_np_vectorized",
    num_chains=20,
    thin=10,
    warmup_steps=50,
    init_strategy="proposal"
    #init_width=0.1
)
posterior = MCMCPosterior(
    potential_fn, proposal=prior,
    #theta_transform=parameter_transform,
    **mcmc_parameters
)
num_samples = 300
nle_samples = posterior.sample(sample_shape=(num_samples,))
torch.save(nle_samples, 'posterior_samples.pkl')
