import torch
import torch.nn as nn
import torch.optim as optim
from botorch.models.model import Model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
import torch.distributions as dist

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
    #true_value = torch.sin(5 * x) * (1 - torch.tanh(x ** 2))
    #noise = torch.rand_like(true_value) * 0.2 - 0.1  # Uniform noise in [-0.1, 0.1]
    #return true_value + noise
    #true_value = torch.where(x >= 0, torch.sqrt(x), -x)
    noise = torch.empty_like(x).uniform_(-1,1) * eps
    noised_x = x + noise
    noised_value = torch.where(noised_x >= 0, torch.sqrt(noised_x), -noised_x)
    return noised_value

def objective(x, num_samples=100):
    samples = torch.stack([expensive_function(x) for _ in range(num_samples)])
    maxima = torch.max(samples, axis=0).values
    return -maxima # Negate bc botorch maximizes by default, we're looking for min

# Generate Training Data
#train_x = torch.linspace(-2, 2, 10).reshape(-1, 1)
#train_x = torch.linspace(-2, 2, 1000).reshape(-1, 1)
train_x = torch.empty(500, 1)
train_x.uniform_(-2,2)
#train_y = expensive_function(train_x)
train_y  = objective(train_x)

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
    #return -likelihood + 0.0005 * kl  # Regularize with KL term

# Train Bayesian Neural Network
bnn = BayesianNN(input_dim=1)
optimizer = optim.Adam(bnn.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    loss = elbo_loss(bnn, train_x, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: ELBO Loss = {loss.item():.4f}")

# Define Acquisition Function (Expected Improvement)
def acquisition(X):
    mean, std = bnn.posterior(X)
    best_f = train_y.max()
    #improvement = (mean - best_f).clamp(min=0)
    improvement = (best_f - mean).clamp(min=0)
    #print(f'X shape: {X.shape}')
    #print(X)
    #print(f'improvement shape: {improvement.shape}')
    #improvement = improvement.reshape(-1,1)
    #print(f'new improvement shape: {improvement.shape}')
    result = improvement / (std + 1e-6)
    result = result.reshape(-1,1)
    #return improvement / (std + 1e-6)
    print(result.shape)
    return result

for _ in range(10):
    # Optimize Acquisition Function to Propose Next Query Point
    bounds = torch.tensor([[-2.0], [2.0]])
    candidate, _ = optimize_acqf(acquisition, bounds=bounds, q=1, num_restarts=5, raw_samples=20)

    print(f"Next query point: {candidate.item()}")
