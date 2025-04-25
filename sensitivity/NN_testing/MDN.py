import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, MixtureSameFamily

class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_mixtures=3):
        super(NN, self).__init__()
        self.num_mixtures = num_mixtures

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.pi_layer = nn.Linear(hidden_dim, num_mixtures)
        self.mu_layer = nn.Linear(hidden_dim, num_mixtures)
        self.log_std_layer = nn.Linear(hidden_dim, num_mixtures)

    def forward(self, x):
        h = self.hidden(x)
        pi_logits = self.pi_layer(h)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        std = torch.exp(log_std)
        #var = torch.square(std)

        return pi_logits, mu, std

    def get_mixture_distribution(self, x):
        pi_logits, mu, std = self.forward(x)
        mix = Categorical(logits=pi_logits)
        comp = Normal(loc=mu, scale=std)
        gmm = MixtureSameFamily(mix, comp)
        return gmm

    def sample_predictions(self, x, n_samples=1):
        """Draw samples from the predictive distribution at input x.
        Returns tensor of shape (n_samples, x.shape[0])"""
        gmm = self.get_mixture_distribution(x)
        samples = gmm.sample((n_samples,))  # shape: (n_samples, batch_size)
        return samples  # shape: (n_samples, batch_size)

    def posterior_robust(self, x_design, robustness_thresh, num_eps_samples=10, eps=0.1, n_mc_samples=1):
        """ Estimate robustness measure at each design point """
        robust_measures = torch.empty((x_design.shape[0], 1))
        for i, x_d in enumerate(x_design):
            x = torch.empty((num_eps_samples, 2))
            x[:, 0] = x_d.repeat(num_eps_samples)
            x[:, 1].uniform_(-eps, eps)

            samples = self.sample_predictions(x, n_samples=n_mc_samples)  # shape: (n_mc_samples, num_eps_samples)
            prob_estimate = (samples > robustness_thresh).float().mean()
            robust_measures[i] = prob_estimate

        return robust_measures

    def predict_mean_and_variance(self, x):
        """ Return the mean and variance of the predictive distribution at each input x """
        pi_logits, mu, std = self.forward(x)
        pi = F.softmax(pi_logits, dim=-1)

        mean = torch.sum(pi * mu, dim=1)
        mean_sq = torch.sum(pi * (mu ** 2 + std ** 2), dim=1)
        var = mean_sq - mean ** 2
        return mean, var

def mdn_loss(pi_logits, mu, std, y):
    m = Categorical(logits=pi_logits)
    component_dist = Normal(mu, std)
    log_probs = component_dist.log_prob(y.unsqueeze(1))  # shape: [batch, num_mixtures]
    log_mix = torch.logsumexp(m.logits + log_probs, dim=1)
    return -torch.mean(log_mix)

def loss_fn(model, x, y, sample_nbr=None, complexity_cost_weight=None, penalty_weight=None):
    pi_logits, mu, std = model(x)
    loss = mdn_loss(pi_logits, mu, std, y)
    return loss
