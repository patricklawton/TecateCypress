import os
import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qNoisyExpectedImprovement, qSimpleRegret
from botorch.acquisition.risk_measures import VaR
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.input import InputPerturbation
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples, draw_sobol_normal_samples
from botorch.utils.transforms import unnormalize
from botorch.test_functions import SixHumpCamel
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

warnings.filterwarnings("ignore")

#SMOKE_TEST = os.environ.get("SMOKE_TEST")
SMOKE_TEST = True
BATCH_SIZE = 2 if not SMOKE_TEST else 1
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 128 if not SMOKE_TEST else 4
N_W = 16 if not SMOKE_TEST else 2
NUM_ITERATIONS = 28 if not SMOKE_TEST else 2
STD_DEV = 0.05
ALPHA = 0.8
NUM_TRAIN = 8 if not SMOKE_TEST else 3

tkwargs = {"device": "cpu", "dtype": torch.double}

test_function = SixHumpCamel(negate=True)
dim = test_function.dim

def evaluate_function(X: Tensor) -> Tensor:
    return test_function(unnormalize(X, test_function.bounds)).view(*X.shape[:-1], 1)

print(f'test function dim: {dim}')

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(**tkwargs)
train_X = draw_sobol_samples(bounds=bounds, n=NUM_TRAIN, q=1).squeeze(-2).to(**tkwargs)
train_Y = evaluate_function(train_X)
#print(f'train_X shape: {train_X.shape}')
print(f'train_X:\n{train_X}')
print(f'train_Y:\n{train_Y}')
print(f'untransformed function vals:\n{test_function(train_X)}')

def train_model(train_X: Tensor, train_Y: Tensor) -> SingleTaskGP:
    r"""Returns a `SingleTaskGP` model trained on the inputs"""
    perturbation_set = draw_sobol_normal_samples(d=dim, n=N_W, **tkwargs) * STD_DEV
    print(f'perturbation_set:\n{perturbation_set}')
    intf = InputPerturbation(
        perturbation_set=draw_sobol_normal_samples(d=dim, n=N_W, **tkwargs) * STD_DEV,
        bounds=bounds,
    )
    model = SingleTaskGP(
        train_X, train_Y, input_transform=intf, outcome_transform=Standardize(m=1)
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


model = train_model(train_X, train_Y)
import sys; sys.exit()

risk_measure = VaR(alpha=ALPHA, n_w=N_W)


def optimize_acqf_and_get_observation():
    r"""Optimizes the acquisition function, and returns a new candidate and observation."""
    acqf = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_X,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([RAW_SAMPLES])),
        objective=risk_measure,
        prune_baseline=True,
    )

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
    )

    new_observations = evaluate_function(candidate)
    return candidate, new_observations

start_time = time()

for i in range(NUM_ITERATIONS):
    print(f"Starting iteration {i}, total time: {time() - start_time:.3f} seconds.")
    # optimize the acquisition function and get the observations
    candidate, observations = optimize_acqf_and_get_observation()

    # update the model with new observations
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, observations], dim=0)
    model = train_model(train_X, train_Y)

# update the input transform of the already trained model
new_intf = InputPerturbation(
    perturbation_set=draw_sobol_normal_samples(d=dim, n=RAW_SAMPLES, **tkwargs) * STD_DEV,
    bounds=bounds,
).eval()
model.input_transform = new_intf

risk_measure = VaR(alpha=ALPHA, n_w=RAW_SAMPLES)
expected_risk_measure = qSimpleRegret(model=model, objective=risk_measure)

with torch.no_grad():
    expected_rm_values = expected_risk_measure(train_X.unsqueeze(-2))
expected_final_rm, max_idx = expected_rm_values.max(dim=0)
final_candidate = train_X[max_idx]
print(f'final_candidate={final_candidate}')
