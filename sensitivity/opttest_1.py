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
ROBUST_SAMPLES = 10
NUM_TRAIN = 15
NUM_EPOCHS = 2_000
RAW_SAMPLES = 10
#MU_INIT = 0.0
#LOGVAR_INIT = -20
MU_INIT = 0.01
LOGVAR_INIT = -7
#MU_INIT = 0.02
#LOGVAR_INIT = -5

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Mean and log variance for weights
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, MU_INIT))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).fill_(LOGVAR_INIT))

        # Mean and log variance for biases
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(0, MU_INIT))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).fill_(LOGVAR_INIT))

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # Sample weights using reparameterization trick
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_std * eps_w
        bias = self.bias_mu + bias_std * eps_b

        return F.linear(x, weight, bias)

    def kl_loss(self):
        # KL divergence between q(w|θ) ~ N(μ,σ²) and p(w) ~ N(0,1)
        # KL(N(μ,σ²) || N(0,1)) = log(1/σ) + (σ² + μ² - 1)/2

        def kl_term(mu, logvar):
            return 0.5 * torch.sum(-logvar + torch.exp(logvar) + mu**2 - 1)

        kl_w = kl_term(self.weight_mu, self.weight_logvar)
        kl_b = kl_term(self.bias_mu, self.bias_logvar)
        return kl_w + kl_b

# Bayesian Neural Network Model
class BayesianNN(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, 50)
        self.fc2_mean = BayesianLinear(50, 1)
        self.fc2_logvar = BayesianLinear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        mean = self.fc2_mean(hidden)
        logvar = self.fc2_logvar(hidden)  # Output log(σ²)
        return mean, logvar

    def posterior(model, x, num_samples=50):
        means = []
        variances = []

        for _ in range(num_samples):
            mean, logvar = model(x)
            means.append(mean)
            variances.append(torch.exp(logvar))  # aleatoric variance

        means = torch.stack(means)       # shape: [S, B, 1]
        variances = torch.stack(variances)

        # Total uncertainty = epistemic + aleatoric
        pred_mean = means.mean(0)
        epistemic_var = means.var(0)
        aleatoric_var = variances.mean(0)
        total_var = epistemic_var + aleatoric_var

        return pred_mean, total_var.sqrt()

# Define Noisy Objective Function
def expensive_function(x, eps=2):
    """Noisy function with uniform noise."""
    x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
    noise = torch.empty_like(x).uniform_(-1,1) * eps
    noised_x = x + noise
    noised_value = torch.where(noised_x >= 0, torch.sqrt(noised_x), -noised_x)
    return noised_value

def robust_measure(x, num_samples=100):
    samples = torch.stack([expensive_function(x) for _ in range(num_samples)])
    maxima = torch.max(samples, axis=0).values
    return -maxima # Negate bc botorch maximizes by default, we're looking for min

# Generate Training Data
#train_x = torch.empty(NUM_TRAIN, 1)
#train_x.uniform_(-2,2)
##train_x.uniform_(-3,3)
##train_x = torch.vstack([torch.empty(10,1).uniform_(-2,2), torch.empty(100,1).uniform_(0,1)])
#train_y  = robust_measure(train_x, num_samples=ROBUST_SAMPLES)
#torch.save(train_x, 'train_x_1')
#torch.save(train_y, 'train_y_1')
train_x = torch.load('train_x_1', weights_only=True)
train_y = torch.load('train_y_1', weights_only=True)

def gaussian_nll(y_pred_mean, y_pred_logvar, y_true):
    # Gaussian negative log-likelihood (per element)
    precision = torch.exp(-y_pred_logvar)
    return 0.5 * torch.sum(y_pred_logvar + precision * (y_true - y_pred_mean)**2)

# Define Variational Inference Loss (ELBO)
def sample_elbo(model, x, y, criterion, sample_nbr=3, complexity_cost_weight=1e-6):
    total_nll = 0.0
    total_kl = 0.0

    for _ in range(sample_nbr):
        mean, logvar = model(x)
        total_nll += gaussian_nll(mean, logvar, y)

    # Average over samples
    nll = total_nll / sample_nbr

    # Sum KL from all Bayesian layers
    for module in model.modules():
        if hasattr(module, 'kl_loss'):
            total_kl += module.kl_loss()

    loss = nll + complexity_cost_weight * total_kl
    return loss

# Train Bayesian Neural Network
bnn = BayesianNN(input_dim=1)
optimizer = optim.Adam(bnn.parameters(), lr=0.001)

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    loss = sample_elbo(bnn, train_x, train_y, criterion=nn.MSELoss(), 
                       sample_nbr=3, complexity_cost_weight=1e-6)
    loss.backward()
    optimizer.step()
    if epoch % (int(NUM_EPOCHS/10)) == 0:
        print(f"Epoch {epoch}: ELBO Loss = {loss.item():.4f}")
with open('model_1.pkl', 'wb') as handle:
    pickle.dump(bnn, handle)
#x = torch.empty(100, 1).uniform_(-2,2)
x = torch.linspace(-2,2,100).unsqueeze(1)
mean, std = bnn.posterior(x)
std = std.squeeze(1).detach()
mean = mean.squeeze(1).detach()
plt.errorbar(x.squeeze(1), mean, yerr=std, fmt='o', zorder=-1)
plt.scatter(train_x, train_y, c='r')
plt.ylabel('robust measure')
plt.xlabel('x')
plt.savefig('trainedmodel1.png')
#sys.exit()

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
print('test acq:\n', acquisition(torch.tensor([[[-2.]]])),'\n')
print('test acq:\n', acquisition(torch.tensor([[[-1.]]])),'\n')
print('test acq:\n', acquisition(torch.tensor([[[0.45]]])),'\n')
print('test acq:\n', acquisition(torch.tensor([[[-2.]]])),'\n')
#sys.exit()

def optimize_acqf_custom(acq_func, bounds, raw_samples=100):
    dim = bounds.shape[1]

    # Generate raw samples using uniform sampling
    raw_candidates = torch.rand((raw_samples, dim)) * (bounds[1] - bounds[0]) + bounds[0]
    raw_values = acq_func(raw_candidates).detach().numpy()
    #print(f'raw_candidates:\n{raw_candidates}')

    # Select the best raw candidate as a starting point
    best_raw_idx = np.argmax(raw_values)
    best_x = raw_candidates[best_raw_idx].clone()
    #print(f"max approx f(x) = {np.max(raw_values)} at x = {best_x}")

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
        #method="L-BFGS-B",
        method="Powell",
    )
    #print(res)

    # Convert result back to tensor
    best_x_optimized = torch.tensor(res.x, dtype=torch.float32)

    return best_x_optimized

num_candidates = 10
all_candidates = torch.empty(num_candidates)
for i, _ in enumerate(range(num_candidates)):
    # Optimize Acquisition Function to Propose Next Query Point
    bounds = torch.tensor([[-2.0], [2.0]])
    candidate = optimize_acqf_custom(acquisition, bounds,  raw_samples=RAW_SAMPLES)
    print(f"candidate optimum: {candidate.item()}")
    all_candidates[i] = candidate
print(f'mean candidate: {torch.mean(all_candidates)}')
