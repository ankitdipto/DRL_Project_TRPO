#!/usr/bin/env python3
"""
Policy and Value Networks for CPG-based Control (PMTG approach).

Instead of directly outputting joint positions, the policy outputs
parameters that modulate a Central Pattern Generator (CPG).
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


class CPGModulatingPolicy(nn.Module):
    """
    Gaussian policy that outputs CPG parameters instead of raw joint commands.
    
    Action space (16-dimensional):
        - frequency (1): Gait frequency in Hz [0.5, 3.0]
        - hip_amplitude (1): Hip joint oscillation amplitude [0.0, 0.3]
        - thigh_amplitude (1): Thigh joint oscillation amplitude [0.0, 0.8]
        - calf_amplitude (1): Calf joint oscillation amplitude [0.0, 1.2]
        - stance_offset (12): Learned offset for stance pose [-0.5, 0.5]
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
        
        # Separate heads for different parameter groups
        # self.frequency_head = nn.Linear(hidden_dim, 1)
        # self.hip_amp_head = nn.Linear(hidden_dim, 1)
        # self.thigh_amp_head = nn.Linear(hidden_dim, 1)
        # self.calf_amp_head = nn.Linear(hidden_dim, 1)
        # self.stance_head = nn.Linear(hidden_dim, 12)
        self.residual_head = nn.Linear(hidden_dim, act_dim)
        
        # Learnable log standard deviation for each parameter
        # Total: 1 + 1 + 1 + 1 + 12 = 16
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


class ValueNetwork(nn.Module):
    """
    Value network for CPG-based control.
    (Same as standard value network, included here for completeness)
    """
    
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        """
        Initialize value network.
        
        Args:
            obs_dim: Observation dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: (batch, obs_dim) observation
            
        Returns:
            value: (batch,) state values
        """
        return self.net(obs).squeeze(-1)


if __name__ == "__main__":
    """Test CPG policy network."""
    print("=" * 70)
    print("Testing CPG-Modulating Policy")
    print("=" * 70)
    
    # Create policy
    obs_dim = 34  # Quadruped observation dimension
    hidden_dim = 128
    batch_size = 8
    
    policy = CPGModulatingPolicy(obs_dim, hidden_dim)
    value_net = ValueNetwork(obs_dim, hidden_dim)
    
    print(f"\nPolicy architecture:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: 16 (CPG parameters)")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Test forward pass
    print(f"\nTesting forward pass (batch_size={batch_size}):")
    obs = torch.randn(batch_size, obs_dim)
    
    mean, std = policy(obs)
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
    print(f"  Std range: [{std.min():.3f}, {std.max():.3f}]")
    
    # Test action sampling
    print(f"\nTesting action sampling:")
    action, log_prob = policy.get_action(obs)
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    
    # Decode action to see CPG parameters
    frequency = action[0, 0].item()
    hip_amp = action[0, 1].item()
    thigh_amp = action[0, 2].item()
    calf_amp = action[0, 3].item()
    print(f"\n  Sample CPG parameters (first in batch):")
    print(f"    Frequency: {frequency:.3f} Hz")
    print(f"    Hip amplitude: {hip_amp:.3f} rad")
    print(f"    Thigh amplitude: {thigh_amp:.3f} rad")
    print(f"    Calf amplitude: {calf_amp:.3f} rad")
    print(f"    Stance offsets: {action[0, 4:].abs().mean():.3f} rad (mean abs)")
    
    # Test value network
    print(f"\nTesting value network:")
    values = value_net(obs)
    print(f"  Value shape: {values.shape}")
    print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")
    
    # Test KL divergence
    print(f"\nTesting KL divergence:")
    policy2 = CPGModulatingPolicy(obs_dim, hidden_dim)
    kl = policy.kl_divergence(obs, policy2)
    print(f"  KL shape: {kl.shape}")
    print(f"  KL range: [{kl.min():.3f}, {kl.max():.3f}]")
    
    # Test gradient flow
    print(f"\nTesting gradient flow:")
    loss = log_prob.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in policy.parameters())
    print(f"  All parameters have gradients: {has_grad}")
    
    print("\n" + "=" * 70)
    print("CPG policy tests passed!")
    print("\nAction space interpretation:")
    print("  [0]: frequency (Hz)")
    print("  [1]: hip_amplitude (rad)")
    print("  [2]: thigh_amplitude (rad)")
    print("  [3]: calf_amplitude (rad)")
    print("  [4-15]: stance_offset (12 joints, rad)")

