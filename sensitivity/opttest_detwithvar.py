import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
import sys
import pickle
from matplotlib import pyplot as plt

verbose = False
NUM_TRAIN = 65
NUM_TRAIN_REPEATS = 1
NUM_EPOCHS = 3_000
RAW_SAMPLES = 20
NUM_EPS_SAMPLES = 400
EPS = 2
MU_INIT = 0.0
LOGVAR_INIT = -5

# Define full model
class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.out_mean = nn.Linear(hidden_dim, 1)
        self.out_var = nn.Linear(hidden_dim, 1)

        # Init: Encourage small initial variance via softplus
        nn.init.constant_(self.out_var.bias, -5.0)  # softplus(-5) ≈ 0.0067

    def forward(self, x):
        x = self.relu(self.fc1(x))
        mean = self.out_mean(x)
        variance = F.softplus(self.out_var(x)) + 1e-4  # small floor to avoid 0
        return mean.squeeze(-1), variance.squeeze(-1)

    def posterior_robust(self, X_design, num_eps_samples=10, eps=0.1):
        """ Estimate robustness measure at each design point """
        robust_measures = torch.empty((X_design.shape[0], 1))
        for i, X_d in enumerate(X_design):
            X = torch.empty((num_eps_samples, 2))
            X[:,0] = X_d.repeat(num_eps_samples)
            X[:,1].uniform_(-eps, eps)
            means, _ = self.forward(X)
            robust_measures[i] = torch.max(means, axis=0).values
        return robust_measures

# -----------------------------
# Loss Function
# -----------------------------
def gaussian_nll_with_variance_penalty(mean, target, variance, penalty_weight=1e-3):
    base_nll = F.gaussian_nll_loss(mean, target, variance, full=True)
    penalty = penalty_weight * torch.mean(variance)
    return base_nll + penalty

# -----------------------------
# ELBO-style Wrapper
# -----------------------------
def sample_elbo(model, x, y, sample_nbr=3, complexity_cost_weight=1e-6):
    total_loss = 0.0
    for _ in range(sample_nbr):
        mean, var = model(x)
        loss = gaussian_nll_with_variance_penalty(mean, y, var)
        total_loss += loss
    return total_loss / sample_nbr

# Define Noisy Objective Function
def expensive_function(x):
    """The 1st column of x gives the nominal inputs, 2nd column gives the input noise"""
    noised_x = torch.sum(x, dim=1)
    noised_value = torch.where(noised_x >= 0, torch.sqrt(noised_x), -noised_x)
    # Add some additional noise we want to capture via Bayesian layers
    std = 0.1
    #std_slope = 0.2
    #std = torch.abs(x[:,0])*std_slope
    #mu = torch.ones(x.shape[0]) * -0.3
    #mu = torch.where(x[:,0] < 0.3, mu, -mu)
    mu = torch.zeros(x.shape[0])
    #mu_slope = 0.05
    #mu = (x[:,0]-2)*mu_slope
    extra_noise = torch.normal(mean=mu, std=std)
    #extra_noise = torch.empty_like(noised_value).log_normal_(-2, 0.5)
    return noised_value + extra_noise

# Generate Training Data
train_x = torch.empty(NUM_TRAIN, 2)
train_x[:,0].uniform_(-2,2)
train_x[:,1].uniform_(-EPS, EPS)
train_x = train_x.repeat((NUM_TRAIN_REPEATS,1)) # Generate repeats of the initial points to help learn variance
train_y  = expensive_function(train_x)
torch.save(train_x, 'train_x_3')
torch.save(train_y, 'train_y_3')
#train_x = torch.load('train_x_3', weights_only=True)
#train_y = torch.load('train_y_3', weights_only=True)

# Train Neural Network
model = NN(input_dim=train_x.shape[1], hidden_dim = 50)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = sample_elbo(model, train_x, train_y, sample_nbr=6, complexity_cost_weight=1e-6)
    loss.backward()
    optimizer.step()
    if epoch % (int(NUM_EPOCHS/10)) == 0:
        print(f"Epoch {epoch}: ELBO Loss = {loss.item():.4f}")
with open('model_3.pkl', 'wb') as handle:
    pickle.dump(model, handle)
test_x = torch.empty(500, 2)
test_x[:,0].uniform_(-2, 2)
test_x[:,1].uniform_(-EPS, EPS);
mean, var = model.forward(test_x)
mean = mean.detach()
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
    #print(f"candidate optimum: {candidate.item()}")
    all_candidates[i] = candidate
print(f'mean candidate: {torch.mean(all_candidates)}, std: {torch.std(all_candidates)}')
