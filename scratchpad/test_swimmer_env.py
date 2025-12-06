import gymnasium as gym
import numpy as np
from tqdm import tqdm

envs = gym.vector.SyncVectorEnv([lambda: gym.make("Swimmer-v5") for _ in range(4)])

obs, infos = envs.reset()

for t in tqdm(range(1100)):
    actions = envs.action_space.sample()
    obs, rews, terminateds, truncateds, infos = envs.step(actions)
    
    if np.any(terminateds) or np.any(truncateds):
        print(f"Terminated: {terminateds}, Truncated: {truncateds} at step {t}")
        break