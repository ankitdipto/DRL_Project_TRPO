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

"""
Policy Network for CPG-based Control (PMTG approach).

Instead of directly outputting joint positions, the policy outputs
parameters that modulate a Central Pattern Generator (CPG).
"""

class CPGModulatingPolicy(nn.Module):
    """
    Gaussian policy that outputs CPG parameters instead of raw joint commands.
    
    Action space (12-dimensional):
        - residual (12): Learned residual for joint commands [-0.5, 0.5]
    """
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        """
        Initialize CPG-modulating policy.
        
        Args:
            obs_dim: Observation dimension (34 for quadruped)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Shared feature extractor
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.residual_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small values for stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor):
        """
        Forward pass to compute mean and std of CPG parameters.
        
        Args:
            obs: (batch, obs_dim) observation tensor
            
        Returns:
            mean: (batch, 16) mean CPG parameters
            std: (batch, 16) standard deviation
        """
        # Shared features
        h = self.net(obs)
        
        # Frequency: map to [0.5, 3.0] Hz
        # Use sigmoid to get [0, 1], then scale to desired range
        # freq_raw = self.frequency_head(h)
        # frequency = 0.5 + 2.5 * torch.sigmoid(freq_raw)  # [0.5, 3.0]
        
        # Hip amplitude: map to [0.0, 0.3] radians
        # hip_amp_raw = self.hip_amp_head(h)
        # hip_amplitude = 0.3 * torch.sigmoid(hip_amp_raw)  # [0.0, 0.3]
        
        # Thigh amplitude: map to [0.0, 0.8] radians
        # thigh_amp_raw = self.thigh_amp_head(h)
        # thigh_amplitude = 0.8 * torch.sigmoid(thigh_amp_raw)  # [0.0, 0.8]
        
        # Calf amplitude: map to [0.0, 1.2] radians
        # calf_amp_raw = self.calf_amp_head(h)
        # calf_amplitude = 1.2 * torch.sigmoid(calf_amp_raw)  # [0.0, 1.2]
        
        # Stance offsets: map to [-0.5, 0.5] radians
        # stance_raw = self.stance_head(h)
        # stance_offset = 0.5 * torch.tanh(stance_raw)  # [-0.5, 0.5]
        
        mean = torch.tanh(self.residual_head(h))
        # Concatenate all parameters
        # mean = torch.cat([
        #     # frequency,
        #     # hip_amplitude,
        #     # thigh_amplitude,
        #     # calf_amplitude,
        #     # stance_offset
        #     residual
        # ], dim=-1)
        
        # Standard deviation (same across batch)
        std = torch.exp(self.log_std).expand_as(mean)
        
        return mean, std
    
    def get_action(self, obs: torch.Tensor):
        """
        Sample action from policy.
        
        Args:
            obs: (batch, obs_dim) observation
            
        Returns:
            action: (batch, 16) sampled CPG parameters
            log_prob: (batch,) log probability of action
        """
        mean, std = self(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor):
        """
        Compute log probability of given action.
        
        Args:
            obs: (batch, obs_dim) observation
            action: (batch, 16) action
            
        Returns:
            log_prob: (batch,) log probability
        """
        mean, std = self(obs)
        dist = Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)
    
    def kl_divergence(self, obs: torch.Tensor, other):
        """
        Compute KL divergence KL(self || other) for given observations.
        
        Args:
            obs: (batch, obs_dim) observations
            other: Another CPGModulatingPolicy
            
        Returns:
            kl: (batch,) KL divergence for each sample
        """
        mean1, std1 = self(obs)
        mean2, std2 = other(obs)
        
        var1 = std1.pow(2)
        var2 = std2.pow(2)
        
        # KL(N(μ1, σ1²) || N(μ2, σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        kl = (
            torch.log(std2 / std1) +
            (var1 + (mean1 - mean2).pow(2)) / (2 * var2) -
            0.5
        )
        return kl.sum(dim=-1)  # Sum over action dimensions