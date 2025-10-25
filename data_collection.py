from collections import namedtuple
import torch
import numpy as np
import copy

Trajectory = namedtuple("Trajectory", ["obs", "acts", "logps", "rews", "vals", "dones"])

# Device will be set from main module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RolloutBuffer:
    obs: torch.Tensor
    acts: torch.Tensor
    logps: torch.Tensor
    rews: torch.Tensor
    vals: torch.Tensor
    dones: torch.Tensor

    def __init__(self, n_steps, n_envs, n_obs, n_acts):
        s, e, o, a = n_steps, n_envs, n_obs, n_acts
        self.obs = torch.zeros((s, e, o), dtype=torch.float32).to(device)
        self.acts = torch.zeros((s, e, a), dtype=torch.float32).to(device)
        self.logps = torch.zeros((s, e), dtype=torch.float32).to(device)
        self.rews = torch.zeros((s, e), dtype=torch.float32).to(device)
        self.vals = torch.zeros((s, e), dtype=torch.float32).to(device)
        self.dones = torch.zeros((s, e), dtype=torch.bool).to(device)

    def add(self, t, obs, act, logp, rew, val, done):
        self.obs[t] = obs
        self.acts[t] = act
        self.logps[t] = logp
        self.rews[t] = torch.tensor(rew, dtype=torch.float32).to(device)
        self.vals[t] = val
        self.dones[t] = torch.tensor(done, dtype=torch.bool).to(device)

    # def get(self):
    #     return Trajectory(
    #         obs=torch.tensor(np.array(self.obs), dtype=torch.float32).to(device),
    #         acts=torch.tensor(np.array(self.acts), dtype=torch.float32).to(device),
    #         logps=torch.tensor(np.array(self.logps), dtype=torch.float32).to(device),
    #         rews=np.array(self.rews, dtype=np.float32),
    #         vals=torch.tensor(np.array(self.vals), dtype=torch.float32).to(device),
    #         dones=np.array(self.dones, dtype=np.bool_),
    #     )


    def compute_gae(self, last_values, gamma=0.99, lam=0.97):
        """
        Compute advantages (GAE) and returns-to-go for a trajectory.
        """
        T, E = self.rews.shape
        adv = torch.zeros((T, E), dtype=torch.float32).to(device)
        lastgaelam = torch.zeros(E, dtype=torch.float32).to(device)
        for t in reversed(range(T)):
            nonterminal = 1.0 - self.dones[t].float()
            nextval = self.vals[t + 1] if t + 1 < T else last_values
            delta = self.rews[t] + gamma * nextval * nonterminal - self.vals[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + self.vals
        return adv, returns

    def compute_monte_carlo_returns(self, last_values, gamma=0.99):
        """
        Compute returns-to-go for a trajectory using Monte Carlo estimation
        """
        T, E = self.rews.shape
        returns = self.rews.clone()
        non_terminal = 1.0 - self.dones[T-1]
        returns[T-1] = last_values * non_terminal
        for t in reversed(range(T - 1)):
            non_terminal = 1.0 - self.dones[t]
            returns[t] = returns[t] + returns[t+1] * non_terminal * gamma
        
        return returns 