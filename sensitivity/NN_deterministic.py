import torch
import torch.nn as nn
import torch.nn.functional as F

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
        nn.init.constant_(self.out_var.bias, LOGVAR_INIT)  # softplus(-5) ≈ 0.0067

    def forward(self, x):
        x = self.relu(self.fc1(x))
        mean = self.out_mean(x)
        variance = F.softplus(self.out_var(x)) + 1e-4  # small floor to avoid 0
        return mean.squeeze(-1), variance.squeeze(-1)

    def posterior_robust(self, x_design, robustness_thresh, num_eps_samples=10, eps=0.1, n_mc_samples=1):
        """ Estimate robustness measure at each design point """
        robust_measures = torch.empty((x_design.shape[0], 1))
        for i, x_d in enumerate(x_design):
            x = torch.empty((num_eps_samples, 2))
            x[:,0] = x_d.repeat(num_eps_samples)
            x[:,1].uniform_(-eps, eps)

            # MC sampling from posterior predictive
            all_preds = []
            for _ in range(n_mc_samples):
                means, _vars = self.forward(x)
                samples = torch.normal(means, torch.sqrt(_vars))
                all_preds.append(samples)

            all_preds = torch.stack(all_preds)  # [n_mc_samples, num_eps_samples]
            robust_measure = torch.mean((all_preds > robustness_thresh).float()).item()
            robust_measures[i] = robust_measure
        return robust_measures

# Loss Function
def gaussian_nll_with_variance_penalty(mean, target, variance, penalty_weight=1e-3):
    base_nll = F.gaussian_nll_loss(mean, target, variance, full=True)
    penalty = penalty_weight * torch.mean(variance)
    return base_nll + penalty

def loss_fn(model, x, y, sample_nbr=3, complexity_cost_weight=None, penalty_weight=1e-3):
    """ Sample ELBO """
    total_loss = 0.0
    for _ in range(sample_nbr):
        mean, var = model(x)
        loss = gaussian_nll_with_variance_penalty(mean, y, var, penalty_weight)
        total_loss += loss
    return total_loss / sample_nbr
