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

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(MU_INIT, 0.1))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features).normal_(LOGVAR_INIT, 0.1))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(MU_INIT, 0.1))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features).normal_(LOGVAR_INIT, 0.1))

        self.efficient = False  # If True, share same weights across entire batch

    def forward(self, x):
        batch_size = x.size(0)

        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        if self.efficient:
            # Sample once per layer (shared across batch)
            eps_w = torch.randn_like(self.weight_mu)
            eps_b = torch.randn_like(self.bias_mu)

            weight = self.weight_mu + weight_std * eps_w
            bias = self.bias_mu + bias_std * eps_b

            return F.linear(x, weight, bias)
        else:
            # Sample per input
            eps_w = torch.randn(batch_size, self.out_features, self.in_features, device=x.device)
            eps_b = torch.randn(batch_size, self.out_features, device=x.device)

            weights = self.weight_mu.unsqueeze(0) + weight_std.unsqueeze(0) * eps_w
            biases = self.bias_mu.unsqueeze(0) + bias_std.unsqueeze(0) * eps_b

            x_exp = x.unsqueeze(1)  # [B, 1, I]
            out = torch.bmm(weights, x_exp.transpose(1, 2)).squeeze(-1)
            return out + biases

    def kl_loss(self):
        std_weight = torch.log1p(torch.exp(self.weight_logvar))
        std_bias = torch.log1p(torch.exp(self.bias_logvar))

        kl_weight = 0.5 * (std_weight.pow(2) + self.weight_mu.pow(2) - 1 - torch.log(std_weight.pow(2) + 1e-8)).sum()
        kl_bias = 0.5 * (std_bias.pow(2) + self.bias_mu.pow(2) - 1 - torch.log(std_bias.pow(2) + 1e-8)).sum()

        return kl_weight + kl_bias


# Full Bayesian Network
class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, hidden_dim)
        self.blinear2 = BayesianLinear(hidden_dim, 2)

    def forward(self, x):
        x = F.relu(self.blinear1(x))
        out = self.blinear2(x).squeeze(-1)
        mean = out[:, 0]
        log_var = out[:, 1]
        var = torch.exp(log_var).clamp(min=1e-6)
        return mean, var

    def kl_loss(self):
        return self.blinear1.kl_loss() + self.blinear2.kl_loss()

    def set_efficient(self, mode=True):
        """Toggle efficient sampling mode for all BayesianLinear layers."""
        for m in self.modules():
            if isinstance(m, BayesianLinear):
                m.efficient = mode

    def posterior_robust(self, x_design, robustness_thresh, num_eps_samples=10, eps=0.1, n_mc_samples=20):
        """Estimate robustness measure at each design point using MC + predictive variance."""
        self.set_efficient(True)  # Enable efficient sampling for speed

        robust_measures = torch.empty((x_design.shape[0], 1))
        for i, x_d in enumerate(x_design):
            # Sample noisy inputs
            x = torch.empty((num_eps_samples, 2))
            x[:, 0] = x_d.repeat(num_eps_samples)
            x[:, 1].uniform_(-eps, eps)

            # MC sampling from posterior predictive
            all_preds = []
            for _ in range(n_mc_samples):
                means, vars = self.forward(x)
                samples = torch.normal(means, torch.sqrt(vars))
                all_preds.append(samples)

            all_preds = torch.stack(all_preds)  # [n_mc_samples, num_eps_samples]
            robust_measure = torch.mean((all_preds > robustness_thresh).float()).item()
            robust_measures[i] = robust_measure

        self.set_efficient(False)  # Restore default for future training
        return robust_measures

# Loss function
def gaussian_nll_loss(y_pred_mean, y_pred_var, y_true):
    # Avoid exploding exponentials
    return torch.mean(0.5 * torch.log(y_pred_var) + 0.5 * ((y_true - y_pred_mean)**2 / y_pred_var))

def loss_fn(model, x, y, sample_nbr=3, complexity_cost_weight=1e-6, penalty_weight=None):
    total_nll = 0.0
    total_kl = 0.0

    for _ in range(sample_nbr):
        means, variances = model(x)
        loss = gaussian_nll_loss(means, variances, y)
        total_nll += loss

    nll = total_nll / sample_nbr
    total_kl = model.kl_loss()
    loss = nll + complexity_cost_weight * total_kl
    return loss
