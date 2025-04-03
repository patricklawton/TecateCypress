import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models.model import Model
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement
from botorch.acquisition.analytic import _log_ei_helper
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from scipy.optimize import minimize

verbose = False

# Define a Bayesian Neural Network (BNN) model
class BayesianNN(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, 50)
        self.out = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        h = self.relu(self.hidden(X))
        return self.out(h)

    def posterior(self, X, num_samples=50):
        """
        Approximate posterior by Monte Carlo sampling.
        """
        outputs = torch.stack([self.forward(X) for _ in range(num_samples)])
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        return mean, std

# Define noisy objective function
def expensive_function(x, eps=2):
    """Noisy function with uniform noise."""
    x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    noise = torch.empty_like(x).uniform_(-1,1) * eps
    noised_x = x + noise
    noised_value = torch.where(noised_x >= 0, torch.sqrt(noised_x), -noised_x)
    return noised_value

def robust_measure(x, num_samples=50):
    samples = torch.stack([expensive_function(x) for _ in range(num_samples)])
    maxima = torch.max(samples, axis=0).values
    return -maxima # Negate bc botorch maximizes by default, we're looking for min

# Generate initial training data
train_x = torch.empty(100, 1)
train_x.uniform_(-2,2)
train_y  = robust_measure(train_x)

# Define and train Bayesian Neural Network
bnn = BayesianNN(input_dim=1)
optimizer = optim.Adam(bnn.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    output, _ = bnn.posterior(train_x)
    loss = loss_fn(output, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Define acquisition function (Log Expected Improvement)
def acquisition(X):
    mean, std = bnn.posterior(X)
    if verbose: print(f'posterior mean:\n{mean}')
    if verbose: print(f'posterior std:\n{std}')
    std = std.clamp(1e-6)
    best_f = train_y.max()
    u = (mean - best_f) / (std)
    log_ei = _log_ei_helper(u)
    if verbose: print(f'log_ei:\n{log_ei}')
    return log_ei + std.log()
print('test acq:\n', acquisition(torch.tensor([[[0.45]]])),'\n')
print('test acq:\n', acquisition(torch.tensor([[[-1.]]])),'\n')
#import sys; sys.exit()

def optimize_acqf_custom(acq_func, bounds, num_restarts=10, raw_samples=100):
    dim = bounds.shape[1]

    # Generate raw samples using uniform sampling
    raw_candidates = torch.rand((raw_samples, dim)) * (bounds[1] - bounds[0]) + bounds[0]
    raw_values = acq_func(raw_candidates).detach().numpy()

    # Select the best raw candidate as a starting point
    best_raw_idx = np.argmax(raw_values)
    best_x = raw_candidates[best_raw_idx].clone()

    def objective(x):
        """Objective function for scipy.optimize (negate log-EI for maximization)."""
        x_torch = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        return -acq_func(x_torch).item()  # Negative since scipy minimizes

    # Bounds for scipy optimizer
    scipy_bounds = [(bounds[0, i].item(), bounds[1, i].item()) for i in range(dim)]

    # Optimize using L-BFGS-B
    res = minimize(
        fun=objective,
        x0=best_x.numpy(),
        bounds=scipy_bounds,
        method="L-BFGS-B",
    )

    # Convert result back to tensor
    best_x_optimized = torch.tensor(res.x, dtype=torch.float32)

    return best_x_optimized

for _ in range(10):
    # Optimize acquisition function to propose next query point
    bounds = torch.tensor([[-2.0], [2.0]])
    candidate = optimize_acqf_custom(acquisition, bounds, num_restarts=10, raw_samples=100)
    print(f"Next query point: {candidate.item()}")
