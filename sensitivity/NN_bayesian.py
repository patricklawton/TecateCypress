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
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(MU_INIT, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(LOGVAR_INIT, 0.1))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(MU_INIT, 0.1))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).normal_(LOGVAR_INIT, 0.1))

    def forward(self, x):
        batch_size = x.size(0)

        # Sample weight and bias epsilons per input in the batch
        eps_w = torch.randn(batch_size, self.out_features, self.in_features, device=x.device)
        eps_b = torch.randn(batch_size, self.out_features, device=x.device)

        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        # Broadcast sampled weights: shape [batch_size, out_features, in_features]
        weights = self.weight_mu.unsqueeze(0) + weight_std.unsqueeze(0) * eps_w
        biases = self.bias_mu.unsqueeze(0) + bias_std.unsqueeze(0) * eps_b

        # x: [batch_size, in_features] → [batch_size, 1, in_features]
        x_expanded = x.unsqueeze(1)  # [B, 1, I]

        # Batched matmul: [B, O, I] x [B, I, 1] → [B, O, 1] → squeeze → [B, O]
        out = torch.bmm(weights, x_expanded.transpose(1, 2)).squeeze(-1)

        # Add bias: [B, O]
        out = out + biases

        return out

    def kl_loss(self):
        std_weight = torch.log1p(torch.exp(self.weight_logvar))
        std_bias = torch.log1p(torch.exp(self.bias_logvar))

        kl_weight = 0.5 * (std_weight.pow(2) + self.weight_mu.pow(2) - 1 - torch.log(std_weight.pow(2) + 1e-8)).sum()
        kl_bias = 0.5 * (std_bias.pow(2) + self.bias_mu.pow(2) - 1 - torch.log(std_bias.pow(2) + 1e-8)).sum()

        return kl_weight + kl_bias

# Define full model
class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, hidden_dim)
        self.blinear2 = BayesianLinear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        preds = self.blinear2(x).squeeze(-1)
        return preds, torch.zeros_like(preds).squeeze(-1)

    def kl_loss(self):
        return self.blinear1.kl_loss() + self.blinear2.kl_loss()

    def posterior_robust(self, x_design, robustness_thresh, num_eps_samples=10, eps=0.1):
        """ Estimate robustness measure at each design point """
        robust_measures = torch.empty((x_design.shape[0], 1))
        for i, x_d in enumerate(x_design):
            x = torch.empty((num_eps_samples, 2))
            x[:,0] = x_d.repeat(num_eps_samples)
            x[:,1].uniform_(-eps, eps)
            means, _ = self.forward(x)
            robust_measure = torch.sum(means > robustness_thresh) / len(x)
            robust_measures[i] = robust_measure
        return robust_measures

# Loss function
def loss_fn(model, x, y, sample_nbr=3, complexity_cost_weight=1e-6, penalty_weight=None):
    loss = nn.MSELoss()
    total_nll = 0.0
    total_kl = 0.0

    for _ in range(sample_nbr):
        preds, _ = model(x)
        total_nll += loss(preds, y)

    nll = total_nll / sample_nbr
    total_kl = model.kl_loss()
    loss = nll + complexity_cost_weight * total_kl
    return loss
