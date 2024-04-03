import torch
import numpy as np
import pickle
import sbi
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi, SNPE
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi.diagnostics import check_sbc, run_sbc
from sbi import analysis as analysis
from sbi import utils as utils
from torch.distributions import Uniform
from sbi.utils import MultipleIndependent
from torch import tensor
from simulator import simulator

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
priors = [Uniform(tensor([rng[0]]), tensor([rng[1]])) for rng in ranges]
prior = MultipleIndependent(priors)
prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)

# Read likelihood estimator from file
with open("likelihood_estimator_big.pkl", "rb") as handle:
    likelihood_estimator = pickle.load(handle)

x_o = np.load('observations/observations.npy')
x_o = torch.Tensor(x_o)
potential_fn, parameter_transform = likelihood_estimator_based_potential(
    likelihood_estimator, prior, x_o
)
mcmc_parameters = dict(
    method = "slice_np",
    num_chains=20,
    thin=10,
    warmup_steps=50,
    init_strategy="proposal",
    theta_transform=parameter_transform
)
posterior = MCMCPosterior(
    potential_fn, proposal=prior, **mcmc_parameters
)
posterior_samples = posterior.sample(sample_shape=(3000,), num_workers=6)
torch.save(posterior_samples, 'posterior_samples_test.pkl')

labels = ['alph_m', 'beta_m', 'sigm_m','alph_nu']
_ = analysis.pairplot(
    posterior_samples, limits=ranges, figsize=(10, 10), labels=labels
)
_[0].savefig('figs/test.png', bbox_inches='tight')
