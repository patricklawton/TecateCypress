import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models.model import Model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import _log_ei_helper
import torch.distributions as dist
import numpy as np
from scipy.optimize import minimize

verbose = True

# Bayesian Linear Layer (Replaces nn.Linear)
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Bayesian Linear Layer with weight uncertainty.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Mean and standard deviation parameters for weights
        self.W_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_logvar = nn.Parameter(torch.zeros(out_features, in_features))  # log(σ²)

        # Mean and standard deviation for bias
        self.b_mu = nn.Parameter(torch.zeros(out_features))
        self.b_logvar = nn.Parameter(torch.zeros(out_features))

    def forward(self, X):
        """
        Perform a stochastic forward pass by sampling weights.
        """
        W_std = torch.exp(0.5 * self.W_logvar)  # Convert log variance to std
        b_std = torch.exp(0.5 * self.b_logvar)

        # Sample weights and bias from learned distributions
        W = self.W_mu + W_std * torch.randn_like(W_std)
        b = self.b_mu + b_std * torch.randn_like(b_std)

        return X @ W.T + b

# Bayesian Neural Network Model
class BayesianNN(Model):
    def __init__(self, input_dim):
        """
        Bayesian Neural Network with Variational Inference.
        """
        super().__init__()
        self.hidden = BayesianLinear(input_dim, 50)  # Bayesian hidden layer
        self.out = BayesianLinear(50, 1)  # Bayesian output layer
        self.relu = nn.ReLU()

    def forward(self, X):
        """
        Forward pass with stochastic Bayesian layers.
        """
        h = self.relu(self.hidden(X))  # Apply Bayesian layer + ReLU
        return self.out(h)  # Compute output

    def posterior(self, X, num_samples=50):
        """
        Approximate posterior by Monte Carlo sampling.
        """
        outputs = torch.stack([self.forward(X) for _ in range(num_samples)])
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        return mean, std

# Define Noisy Objective Function
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

# Generate Training Data
train_x = torch.empty(40, 1)
train_x.uniform_(-2,2)
train_y  = robust_measure(train_x)

# Define Variational Inference Loss (ELBO)
def elbo_loss(bnn, X, Y, num_samples=10):
    """
    Compute the Evidence Lower Bound (ELBO).
    """
    outputs = torch.stack([bnn(X) for _ in range(num_samples)])  # Monte Carlo samples
    likelihood = -((outputs - Y) ** 2).mean()  # Negative MSE (Gaussian likelihood)

    # KL divergence (prior regularization)
    kl = 0
    for layer in [bnn.hidden, bnn.out]:
        kl += torch.sum(0.5 * (layer.W_mu**2 + torch.exp(layer.W_logvar) - layer.W_logvar - 1))
        kl += torch.sum(0.5 * (layer.b_mu**2 + torch.exp(layer.b_logvar) - layer.b_logvar - 1))

    return -likelihood + 0.001 * kl  # Regularize with KL term

# Train Bayesian Neural Network
bnn = BayesianNN(input_dim=1)
optimizer = optim.Adam(bnn.parameters(), lr=0.01)

for epoch in range(2000):
    optimizer.zero_grad()
    loss = elbo_loss(bnn, train_x, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: ELBO Loss = {loss.item():.4f}")

# Define acquisition function (Log Expected Improvement)
def acquisition(X):
    if verbose: print(f'design point:\n{X}')
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
print('test acq:\n', acquisition(torch.tensor([[[-2.]]])),'\n')
import sys; sys.exit()

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

for _ in range(1):
    # Optimize Acquisition Function to Propose Next Query Point
    bounds = torch.tensor([[-2.0], [2.0]])
    #candidate, _ = optimize_acqf(acquisition, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

    #print(f"Next query point: {candidate.item()}")

    candidate = optimize_acqf_custom(acquisition, bounds, num_restarts=10, raw_samples=100)
    print(f"Next query point: {candidate.item()}")
