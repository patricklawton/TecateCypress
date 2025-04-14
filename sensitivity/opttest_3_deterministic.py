import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from botorch.models.model import Model
from botorch.acquisition.analytic import _log_ei_helper
import numpy as np
from scipy.optimize import minimize
import sys
import pickle
from matplotlib import pyplot as plt

verbose = False
NUM_TRAIN = 150
NUM_EPOCHS = 5_000
RAW_SAMPLES = 20
NUM_EPS_SAMPLES = 200
EPS = 2

class NN(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, 50)
        self.out = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        h = self.relu(self.hidden(X))
        return self.out(h)

    def posterior(self, X, num_samples=1):
        """
        Approximate posterior by Monte Carlo sampling.
        """
        outputs = torch.stack([self.forward(X) for _ in range(num_samples)])
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        return mean, std

    def posterior_robust(self, X_design, num_eps_samples=10, eps=0.1):
        """ Estimate robustness measure at each design point """
        robust_measures = torch.empty((X_design.shape[0], 1))
        for i, X_d in enumerate(X_design):
            X = torch.empty((num_eps_samples, 2))
            X[:,0] = X_d.repeat(num_eps_samples)
            X[:,1].uniform_(-eps, eps)
            outputs = self.forward(X)
            robust_measures[i] = torch.max(outputs, axis=0).values
        return robust_measures

# Define Noisy Objective Function
def expensive_function(x):
    """The 1st column of x gives the nominal inputs, 2nd column gives the noise"""
    noised_x = torch.sum(x, dim=1)
    noised_value = torch.where(noised_x >= 0, torch.sqrt(noised_x), -noised_x)
    #return noised_value 
    # Add some additional noise we want to capture
    #std = torch.where(x[:,0] >= 0, torch.sqrt(x[:,0]), 0.001)
    std = 0.1
    extra_noise = torch.normal(mean=torch.zeros(x.shape[0]), std=std)
    return noised_value + extra_noise

# Generate Training Data
train_x = torch.empty(NUM_TRAIN, 2)
train_x[:,0].uniform_(-2,2)
train_x[:,1].uniform_(-EPS, EPS)
train_y  = expensive_function(train_x)
torch.save(train_x, 'train_x_3')
torch.save(train_y, 'train_y_3')
#train_x = torch.load('train_x_3', weights_only=True)
#train_y = torch.load('train_y_3', weights_only=True)

# Train Neural Network
model = NN(input_dim=train_x.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output, _ = model.posterior(train_x)
    loss = loss_fn(output.squeeze(1), train_y)
    loss.backward()
    optimizer.step()
    if epoch % (int(NUM_EPOCHS/10)) == 0:
        print(f"Epoch {epoch}: ELBO Loss = {loss.item():.4f}")
with open('model_3.pkl', 'wb') as handle:
    pickle.dump(model, handle)
test_x = torch.empty(500, 2)
test_x[:,0].uniform_(-2, 2)
test_x[:,1].uniform_(-EPS, EPS);
mean, std = model.posterior(test_x)
mean = mean.squeeze(1).detach()
plt.scatter(test_x[:,0], mean)
plt.ylabel('f(x)')
plt.xlabel('x')
plt.savefig('trainedmodel3.png')

def optimize_nn(bounds, raw_samples = 10):
    dim = bounds.shape[1]

    # Generate raw samples using uniform sampling
    raw_candidates = torch.rand((raw_samples, dim)) * (bounds[1] - bounds[0]) + bounds[0]
    raw_values = model.posterior_robust(raw_candidates, eps=EPS).detach().numpy()
    if verbose: print(f'raw_candidates:\n{raw_candidates}')
    if verbose: print(f'raw_values:\n{raw_values}')

    # Select the best raw candidate as a starting point
    best_raw_idx = np.argmin(raw_values)
    best_x = raw_candidates[best_raw_idx].clone()
    if verbose: print(f"min approx f(x) = {np.min(raw_values)} at x = {best_x}")

    def objective(x):
        """Objective function for scipy.optimize (negate log-EI for maximization)."""
        x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        #return -acq_func(x_torch).item()  # Negative since scipy minimizes
        return model.posterior_robust(x_torch, num_eps_samples=NUM_EPS_SAMPLES, eps=EPS).item()
    if verbose: print(objective(best_x.numpy()))

    # Bounds for scipy optimizer
    scipy_bounds = [(bounds[0, i].item(), bounds[1, i].item()) for i in range(dim)]

    # Optimize using Powell 
    res = minimize(
        fun=objective,
        x0=best_x.numpy(),
        bounds=scipy_bounds,
        #method="L-BFGS-B",
        method="Powell",
    )
    if verbose: print(res)
    return torch.tensor(res.x, dtype=torch.float32)

bounds = torch.tensor([[-2.0], [2.0]])
num_candidates = 10
all_candidates = torch.empty(num_candidates)
for i, _ in enumerate(range(num_candidates)):
    bounds = torch.tensor([[-2.0], [2.0]])
    candidate = optimize_nn(bounds,  raw_samples=RAW_SAMPLES)
    print(f"candidate optimum: {candidate.item()}")
    all_candidates[i] = candidate
print(f'mean candidate: {torch.mean(all_candidates)}, std: {torch.std(all_candidates)}')
