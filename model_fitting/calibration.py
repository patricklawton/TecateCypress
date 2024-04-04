import torch
import numpy as np
import pickle
import sbi
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import likelihood_estimator_based_potential, MCMCPosterior
from sbi.diagnostics import check_sbc, run_sbc
from sbi import utils as utils
from torch.distributions import Uniform
from sbi.utils import MultipleIndependent
from torch import tensor
from simulator import simulator

with open("prior.pkl", "rb") as handle:
    prior = pickle.load(handle)
prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)

with open("posterior.pkl", "rb") as handle:
    posterior = pickle.load(handle)

num_simulations = 300  # choose a number of sbc runs, should be ~100s or ideally 1000
# generate ground truth parameters and corresponding simulated observations for SBC.
thetas = prior.sample((num_simulations,))
xs = simulator(thetas)

num_posterior_samples = 1_000
ranks, dap_samples = run_sbc(
    thetas, xs, posterior, num_posterior_samples=num_posterior_samples,
    num_workers=1, reduce_fns=posterior.log_prob,
)
