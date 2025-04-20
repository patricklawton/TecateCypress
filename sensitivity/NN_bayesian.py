import torch
import torch.nn as nn
import torch.nn.functional as F

MU_INIT = 0.0
LOGVAR_INIT = -5

# Define Bayesian Linear Layer
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters for weights
        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features).normal_(MU_INIT, 0.1))
        self.rho_weight = nn.Parameter(torch.Tensor(out_features, in_features).normal_(LOGVAR_INIT, 0.1))

        self.mu_bias = nn.Parameter(torch.Tensor(out_features).normal_(MU_INIT, 0.1))
        self.rho_bias = nn.Parameter(torch.Tensor(out_features).normal_(LOGVAR_INIT, 0.1))

    def forward(self, x):
        std_weight = F.softplus(self.rho_weight)
        std_bias = F.softplus(self.rho_bias)

        weight_eps = torch.randn_like(std_weight)
        bias_eps = torch.randn_like(std_bias)

        weight = self.mu_weight + std_weight * weight_eps
        bias = self.mu_bias + std_bias * bias_eps

        return F.linear(x, weight, bias)

    def kl_loss(self):
        std_weight = torch.log1p(torch.exp(self.rho_weight))
        std_bias = torch.log1p(torch.exp(self.rho_bias))

        kl_weight = 0.5 * (std_weight.pow(2) + self.mu_weight.pow(2) - 1 - torch.log(std_weight.pow(2) + 1e-8)).sum()
        kl_bias = 0.5 * (std_bias.pow(2) + self.mu_bias.pow(2) - 1 - torch.log(std_bias.pow(2) + 1e-8)).sum()

        return kl_weight + kl_bias

# Define full model
class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, hidden_dim)
        self.blinear2 = BayesianLinear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        return self.blinear2(x)

    def kl_loss(self):
        return self.blinear1.kl_loss() + self.blinear2.kl_loss()

    def posterior(self, X, num_samples=1):
        outputs = torch.stack([self.forward(X) for _ in range(num_samples)])
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        return mean, std

    def posterior_robust(self, x_design, robustness_thresh, num_eps_samples=10, eps=0.1):
        """ Estimate robustness measure at each design point """
        robust_measures = torch.empty((x_design.shape[0], 1))
        for i, x_d in enumerate(x_design):
            x = torch.empty((num_eps_samples, 2))
            x[:,0] = x_d.repeat(num_eps_samples)
            x[:,1].uniform_(-eps, eps)
            means = self.forward(x)
            robust_measure = torch.sum(means > robustness_thresh) / len(x)
            robust_measures[i] = robust_measure
        return robust_measures

# Loss function
def loss_fn(model, x, y, sample_nbr=3, complexity_cost_weight=1e-6):
    loss = nn.MSELoss()
    total_nll = 0.0
    total_kl = 0.0

    for _ in range(sample_nbr):
        preds = model(x).squeeze(1)
        total_nll += loss(preds, y)

    nll = total_nll / sample_nbr
    total_kl = model.kl_loss()
    loss = nll + complexity_cost_weight * total_kl
    return loss

