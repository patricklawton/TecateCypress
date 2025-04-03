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

# Define a Bayesian Neural Network (BNN) model
class BayesianNN(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, 50)
        self.out = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.log_noise = nn.Parameter(torch.tensor([0.1]))

    def forward(self, X):
        h = self.relu(self.hidden(X))
        return self.out(h)

    def posterior(self, X, observation_noise=False):
        mean = self.forward(X)
        noise = self.log_noise.exp()
        return mean, noise

# Define noisy objective function
def expensive_function(x, eps=2):
    """Noisy function with uniform noise."""
    x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    noise = torch.empty_like(x).uniform_(-1,1) * eps
    noised_x = x + noise
    noised_value = torch.where(noised_x >= 0, torch.sqrt(noised_x), -noised_x)
    return noised_value

def objective(x, num_samples=100):
    samples = torch.stack([expensive_function(x) for _ in range(num_samples)])
    maxima = torch.max(samples, axis=0).values
    return -maxima # Negate bc botorch maximizes by default, we're looking for min

# Generate initial training data
train_x = torch.empty(20, 1)
train_x.uniform_(-2,2)
train_y  = objective(train_x)

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

# Define acquisition function (Expected Improvement)
def acquisition(X):
    print(f'input to acqf:\n{X}')
    mean, noise = bnn.posterior(X)
    print(f'posterior mean:\n{mean}')
    print(f'posterior noise:\n{noise}')
    best_f = train_y.max()
    print(f'best_f:\n{best_f}')
    improvement = (mean - best_f).clamp(min=0)
    print(f'improvement:\n{improvement}')
    return improvement / (noise + 1e-6)
def acquisition(X):
    mean, noise = bnn.posterior(X)
    best_f = train_y.max()
    u = (mean - best_f) / (noise + 1e-6)
    log_ei = _log_ei_helper(u)
    print(f'log_ei:\n{log_ei}')
    return log_ei + noise.log()
print('test acq:\n', acquisition(torch.tensor([[[0.45]]])),'\n')
print('test acq:\n', acquisition(torch.tensor([[[-1.]]])),'\n')
#import sys; sys.exit()

def optimize_acqf_custom(acq_func, bounds, num_restarts=10, raw_samples=100):
    """
    Custom implementation of acquisition function optimization.

    Args:
        acq_func: The acquisition function to optimize (returns log-EI values).
        bounds: A tensor of shape (2, d) specifying the search space bounds.
        num_restarts: Number of optimization restarts.
        raw_samples: Number of initial raw samples.

    Returns:
        best_x: The best candidate found.
    """

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

for _ in range(1):
    # Optimize acquisition function to propose next query point
    bounds = torch.tensor([[-2.0], [2.0]])
    #candidate, _ = optimize_acqf(acquisition, bounds=bounds, q=1, num_restarts=1, raw_samples=20)
    #print(f"Next query point: {candidate.item()}")

    candidate = optimize_acqf_custom(acquisition, bounds, num_restarts=10, raw_samples=100)
    print(f"Next query point: {candidate.item()}")
