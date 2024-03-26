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
from scipy.stats import moment

overwrite_observations = False
overwrite_estimator = True

if (os.path.isfile('observations/observations.npy') == False) or overwrite_observations:
    # First, compute and store summary statistics of observed data
    census_yrs = [2,6,8,11,14]
    fn = 'observations/mortality.csv'
    mortality_o = pd.read_csv(fn, header=None)
    mortality_o[0] = [round(v) for v in mortality_o[0]]
    ### avg, min, max
    #mort_o = mortality_o
    #res_len = len(census_yrs)
    #observations = np.empty(res_len*3)
    #observations[0:res_len] = [mort_o[mort_o[0] == yr][1].to_numpy().mean() for yr in census_yrs]
    #observations[res_len:res_len*2] = [mort_o[mort_o[0] == yr][1].to_numpy().min() for yr in census_yrs]
    #observations[res_len*2:res_len*3] = [mort_o[mort_o[0] == yr][1].to_numpy().max() for yr in census_yrs]
    ### first 3 moments
    m1 = []; m2 = []; m3 = []
    for t_i, t in enumerate(census_yrs):
        mort_sub = mortality_o[mortality_o[0]==t][1].to_numpy()
        est_mean = np.mean(mort_sub)
        m1.append(est_mean)
        m2.append(moment(mort_sub, moment=2))
        m3.append(moment(mort_sub, moment=3))
    observations = np.concatenate((m1,m2,m3))
    ### mean, 10th, 90th percentiles
    #res_len = len(census_yrs)
    #observations = np.empty(res_len*3)
    #mort_subs = [mortality_o[mortality_o[0]==t][1].to_numpy() for t in census_yrs]
    #observations[0:res_len] = [np.mean(ms) for ms in mort_subs] 
    #observations[res_len:res_len*2] = [np.percentile(ms, 20) for ms in mort_subs]
    #observations[res_len*2:res_len*3] = [np.percentile(ms, 80) for ms in mort_subs]
    np.save('observations/observations.npy', observations)

defaults = np.array([0.2, 0.8, 0.45])
ranges = np.array([
                   # alph_m
                   [0.01, 0.6], 
                   # beta_m
                   [0.01, 0.9], 
                   # sigm_m
                   [0.1,1.7], 
                   # alph_nu
                   [0.05,3.5]
])
#priors = [
#    # alph_m
#    Uniform(tensor([ranges[0][0]]), tensor([ranges[0][1]])),
#    # beta_m
#    Uniform(tensor([ranges[1][0]]), tensor([ranges[1][1]])),
#    # sigm_m
#    Uniform(tensor([ranges[2][0]]), tensor([ranges[2][1]])),
#    # alph_nu
#    Uniform(tensor([ranges[3][0]]), tensor([ranges[3][1]])),
#]
priors = [Uniform(tensor([rng[0]]), tensor([rng[1]])) for rng in ranges]
prior = MultipleIndependent(priors)

prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)

if (os.path.isfile('likelihood_estimator.pkl') == False) or overwrite_estimator:
    inferer = SNLE(prior, show_progress_bars=True, density_estimator="mdn")
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=200000)
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
num_samples = 10000
nle_samples = posterior.sample(sample_shape=(num_samples,))
torch.save(nle_samples, 'posterior_samples.pkl')
