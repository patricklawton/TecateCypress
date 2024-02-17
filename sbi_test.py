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

priors = [
    # m_a
    Uniform(tensor([0.1]), tensor([1.8])),
    Uniform(tensor([0.1]), tensor([0.8])),
    Uniform(tensor([0.]), tensor([0.1])),
    # epsilon_m
    Uniform(tensor([0.]), tensor([0.6])),
    Uniform(tensor([0.]), tensor([0.1])),
    # nu_a
    Uniform(tensor([0.2]), tensor([1.2])),
    Uniform(tensor([0.05]), tensor([0.5])),
    Uniform(tensor([0.01]), tensor([0.12])),
    # K_a
    Uniform(tensor([4000.]), tensor([10000.])),
    Uniform(tensor([0.15]), tensor([0.45])),
    Uniform(tensor([700.]), tensor([1300.])),
    # eta
    Uniform(tensor([0.01]), tensor([0.15])),
    # r_a
    Uniform(tensor([0.75]), tensor([2.5])),
    # epsilon_r
    Uniform(tensor([0.75]), tensor([2.])),
    Uniform(tensor([0.1]), tensor([2.5]))#,
    # N_1(0)
    #Uniform(tensor([200.]), tensor([16000.]))
]
prior = MultipleIndependent(priors)

def simulator(params):
    # Assign parameter labels
    alph_m = params[0]; beta_m = params[1]; gamm_m = params[2]
    sigm_m = params[3]; tau_m = params[4]
    alph_nu = params[5]; beta_nu = params[6]; gamm_nu = params[7]
    K_seedling = params[8]; kappa = params[9]; K_adult = params[10]
    eta = params[11]
    beta_r = params[12]
    sigm_r = params[13]; tau_r = params[14]
    #N_1_0 = params[15]

    # For generating env stochasticity multipliers
    rng = np.random.default_rng()

    # Initialize empty results array
    results = np.empty(3)
    res_i = 0

    t_vec = np.arange(1,6)
    N_vec = np.zeros(len(t_vec))
    N_1_0 = rng.integers(200,1400)
    N_vec[0] = N_1_0
    N_vec = N_vec.astype(int)
    # print(N_vec, '\n')

    m_a = alph_m * np.exp(-beta_m*t_vec) + gamm_m
    r_tstar = np.exp(-beta_r*t_vec)
    K_a = K_seedling * np.exp(-kappa*t_vec) + K_adult
    nu_a = alph_nu * np.exp(-beta_nu*t_vec) + gamm_nu

    for age_i, t in enumerate(t_vec[:-1]):
        # print('t={}'.format(t))
        # Add density dependent term to mortalities
        dens_dep = ((nu_a)*(1-m_a)) / (1 + np.exp(-eta*K_adult*(np.sum(N_vec/K_a))))
        m_a_N = m_a + dens_dep
        # Draw env. stoch. terms and combine for final survival prob.
        epsilon_m = rng.lognormal(np.zeros_like(t_vec), sigm_m*np.exp(-tau_m*t_vec))
        survival_probs = np.exp(-m_a_N * epsilon_m)
        num_survivors = rng.binomial(N_vec, survival_probs)
        num_survivors = np.roll(num_survivors, 1)
        # print(num_survivors)
        # Get number of non-fire seedlings, including env. stoch.
        epsilon_r = rng.lognormal(t, sigm_r*np.exp(-tau_r*t))
        tstar_i = age_i + 1 #Not true if timestep != age gaps
        num_births = rng.poisson(r_tstar[tstar_i]*epsilon_r*N_1_0)
        # print(num_births)
        # Finally, update abundances
        N_vec = num_survivors
        N_vec[0] = num_births
        # print(N_vec, '\n')
        # Store relevant data to compare with observations
        if t in [1,2,5]:
            results[res_i] = N_vec.sum()
            res_i += 1
    return results

prior, theta_numel, prior_returns_numpy = utils.user_input_checks.process_prior(prior)
simulator = utils.user_input_checks.process_simulator(simulator, prior, is_numpy_simulator=True)

# Read likelihood estimator from file
with open("likelihood_estimator.pkl", "rb") as handle:
    likelihood_estimator = pickle.load(handle)

x_o = np.load('observations.npy')
x_o = torch.Tensor(x_o)
potential_fn, parameter_transform = likelihood_estimator_based_potential(
    likelihood_estimator, prior, x_o
)

mcmc_parameters = dict(
    method = "slice_np_vectorized",
    num_chains=50,
    thin=10,
    warmup_steps=100,
    init_strategy="proposal"#,
    #init_width=0.1
)
posterior = MCMCPosterior(
    potential_fn, proposal=prior, **mcmc_parameters
    #theta_transform=parameter_transform
)
num_samples = 30000
nle_samples = posterior.sample(sample_shape=(num_samples,))
torch.save(nle_samples, 'posterior_samples.pkl')
