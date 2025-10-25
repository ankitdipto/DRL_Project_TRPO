import torch
import torch.nn as nn
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        std = torch.exp(self.log_std)
        return mean, std
        
    def get_action(self, obs):
        mean, std = self(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
        
    def log_prob(self, obs, action):
        mean, std = self(obs)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)
        
    def kl_divergence(self, obs, other):
        # KL( self || other ) for Gaussian policies
        mean1, std1 = self(obs)
        mean2, std2 = other(obs)
        var1, var2 = std1.pow(2), std2.pow(2)
        kl = torch.log(std2 / std1) + (var1 + (mean1 - mean2).pow(2)) / (2 * var2) - 0.5
        return kl.sum(dim=-1)  # sum over action dims
            
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs) -> torch.Tensor:
        return self.net(obs).squeeze(-1)