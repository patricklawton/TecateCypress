import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models.model import Model
import numpy as np
from scipy.optimize import minimize
import sys
import pickle
from matplotlib import pyplot as plt

verbose = False
NUM_TRAIN = 50
NUM_TRAIN_REPEATS = 3 
#NUM_X0 = 100  # Number of unique x0s
#NUM_SAMPLES_PER_X0 = 5  # Samples per x0
NUM_X0 = 20  # Number of unique x0s
NUM_SAMPLES_PER_X0 = 20  # Samples per x0
#NUM_EPOCHS = 5_000
WARMUP_EPOCHS = 100
#TOTAL_EPOCHS = 3000
TOTAL_EPOCHS = 7000
RAW_SAMPLES = 20
NUM_EPS_SAMPLES = 200
EPS = 2

class NN(Model):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.var_head = nn.Linear(hidden_dim, 1)  # Output will be transformed to ensure positivity

    def forward(self, x):
        h = self.net(x)
        mu = self.mean_head(h)
        raw_var = self.var_head(h)
        # Ensure variance is strictly positive
        #var = torch.exp(raw_var)  
        var = torch.nn.functional.softplus(raw_var) + 1e-6
        return mu, var

    def posterior(self, X, num_samples=1):
        """
        Approximate posterior by Monte Carlo sampling.
        """
        outputs = torch.stack([self.forward(X)[0] for _ in range(num_samples)])
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
            outputs = self.forward(X)[0]
            robust_measures[i] = torch.max(outputs, axis=0).values
        return robust_measures

# Define Noisy Objective Function
def expensive_function(x):
    """The 1st column of x gives the nominal inputs, 2nd column gives the noise"""
    noised_x = torch.sum(x, dim=1)
    noised_value = torch.where(noised_x >= 0, torch.sqrt(noised_x), -noised_x)
    # Add some additional noise we want to capture
    #std = torch.where(x[:,0] >= 0, torch.sqrt(x[:,0]), 0.001)
    #std = torch.sqrt(torch.abs(x[:,0]))
    std = 0.1
    extra_noise = torch.normal(mean=torch.zeros(x.shape[0]), std=std)
    #extra_noise = torch.where(extra_noise < 0, -extra_noise, extra_noise)
    #extra_noise = torch.empty_like(noised_value).uniform_(-0.1, 0.1)
    return noised_value + extra_noise

# Generate Training Data
train_x = torch.empty(NUM_TRAIN, 2)
train_x[:,0].uniform_(-2,2)
train_x[:,1].uniform_(-EPS, EPS)
train_x = train_x.repeat((NUM_TRAIN_REPEATS,1)) # Generate repeats of the initial points to help learn variance
#x0s = torch.linspace(-2, 2, NUM_X0).unsqueeze(1)  # shape [NUM_X0, 1]
#epsilons = torch.empty(NUM_X0 * NUM_SAMPLES_PER_X0, 1).uniform_(-EPS, EPS)
#train_x = x0s.repeat_interleave(NUM_SAMPLES_PER_X0, dim=0)  # repeat each x0
#train_x = torch.cat([train_x, epsilons], dim=1)  # shape [NUM_X0 * NUM_SAMPLES_PER_X0, 2]
#train_x = train_x.repeat((NUM_TRAIN_REPEATS,1)) # Generate repeats of the initial points to help learn variance
train_y  = expensive_function(train_x)
torch.save(train_x, 'train_x_3')
torch.save(train_y, 'train_y_3')
#train_x = torch.load('train_x_3', weights_only=True)
#train_y = torch.load('train_y_3', weights_only=True)

# Train Neural Network
model = NN(input_dim=train_x.shape[1], hidden_dim=30)
optimizer = optim.Adam(model.parameters(), lr=0.001)

## Stage 1: Train only on MSE to get a decent mean
#mse_loss = nn.MSELoss()
#for epoch in range(WARMUP_EPOCHS):
#    model.train()
#    optimizer.zero_grad()
#    mu_pred, _ = model(train_x)
#    loss = mse_loss(mu_pred.squeeze(), train_y)
#    loss.backward()
#    optimizer.step()

# Stage 2: Train with Gaussian NLL to learn variance
optimizer = optim.Adam(model.parameters(), lr=0.001)
gll = nn.GaussianNLLLoss(full=True, reduction='mean', eps=1e-6)
def tempered_nll(y, mean, var, alpha=0.7, eps=1e-6):
    #var = var + eps  # ensure numerical stability
    var = torch.clamp(var, min=eps)
    loss = ((y - mean)**2) / (2 * var**alpha) + (alpha / 2) * torch.log(var)
    return loss.mean()
def nll_with_min_variance_penalty(y, mean, var, penalty_weight=1e-3, eps=1e-6):
    #var = var + eps  # stability again
    var = torch.clamp(var, min=eps)
    nll = ((y - mean)**2) / (2 * var) + 0.5 * torch.log(var)
    penalty = penalty_weight * (1 / var).mean()
    return nll.mean() + penalty
for epoch in range(WARMUP_EPOCHS, TOTAL_EPOCHS):
    model.train()
    optimizer.zero_grad()
    mu_pred, var_pred = model(train_x)
    #loss = gll(mu_pred.squeeze(), train_y, var_pred.squeeze())
    #loss = tempered_nll(train_y, mu_pred.squeeze(), var_pred.squeeze(), alpha=0.95)
    loss = nll_with_min_variance_penalty(train_y, mu_pred.squeeze(), var_pred.squeeze(), penalty_weight=2.5e-3)
    loss.backward()
    optimizer.step()
    if epoch % (int(TOTAL_EPOCHS/10)) == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

with open('model_3.pkl', 'wb') as handle:
    pickle.dump(model, handle)
test_x = torch.empty(1000, 2)
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
