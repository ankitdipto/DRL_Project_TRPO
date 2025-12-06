#!/usr/bin/env python3
"""
TRPO Training Script for Unitree Go1 Quadruped Locomotion.

This script adapts the TRPO implementation from main.py to work with the
quadruped environment using MuJoCo's rollout module for efficient parallel simulation.
"""

from collections import deque
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np
import copy
import os
from datetime import datetime
from tqdm import tqdm
import time
import statistics
import hydra
from omegaconf import DictConfig, OmegaConf

from actor_critic import GaussianPolicy, ValueNetwork
from data_collection import RolloutBuffer
from quadruped_env import QuadrupedEnv, make_quadruped_env

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def extract_env_dims(vec_env) -> tuple[int, int]:
    """
    Robustly extract observation and action dimensions from a vectorized environment.
    
    Handles both CustomVectorizedQuadrupedEnv and gym.vector.SyncVectorEnv (or any gym.vector.VectorEnv).
    
    Args:
        vec_env: Vectorized environment (CustomVectorizedQuadrupedEnv or gym.vector.VectorEnv)
    
    Returns:
        Tuple of (observation_dimension, action_dimension)
    
    Raises:
        ValueError: If dimensions cannot be extracted from the environment
    """
    if isinstance(vec_env, gym.vector.VectorEnv):
        # gym.vector wrappers expose single_observation_space and single_action_space
        single_obs_space = vec_env.single_observation_space
        single_act_space = vec_env.single_action_space
        if hasattr(single_obs_space, 'shape') and single_obs_space.shape is not None:
            obs_dim = int(single_obs_space.shape[0])
        else:
            raise ValueError(f"Unable to extract observation dimension from {type(single_obs_space)}")
        if hasattr(single_act_space, 'shape') and single_act_space.shape is not None:
            act_dim = int(single_act_space.shape[0])
        else:
            raise ValueError(f"Unable to extract action dimension from {type(single_act_space)}")
    else:
        # CustomVectorizedQuadrupedEnv or other custom envs
        obs_space = vec_env.observation_space
        act_space = vec_env.action_space
        if hasattr(obs_space, 'shape') and obs_space.shape is not None and len(obs_space.shape) > 0:
            obs_dim = int(obs_space.shape[0])
        else:
            raise ValueError(f"Unable to extract observation dimension from {type(obs_space)}")
        if hasattr(act_space, 'shape') and act_space.shape is not None and len(act_space.shape) > 0:
            act_dim = int(act_space.shape[0])
        else:
            raise ValueError(f"Unable to extract action dimension from {type(act_space)}")
    
    return obs_dim, act_dim


def flatten_params(params) -> torch.Tensor:
    """Flatten model parameters into a single vector."""
    return torch.cat([p.view(-1) for p in params])


def load_flat_params(model, flat_params):
    """Load flattened parameters back into model."""
    idx = 0
    for p in model.parameters():
        NP = p.numel()
        p.data.copy_(flat_params[idx : idx + NP].view(p.shape))
        idx += NP


def conjugate_gradient(Ax, b, cg_iters=10, residual_tol=1e-10, eps=1e-8):
    """
    Solve Ax = b using the conjugate gradient method.
    
    Args:
        Ax: A function that computes A @ x
        b: right-hand side vector
        cg_iters: Maximum number of iterations
        residual_tol: Convergence tolerance
        eps: Small constant for numerical stability
    """
    b = b.to(device)
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b).to(device)
    r_dot_r = torch.dot(r, r)

    for i in range(cg_iters):
        Ap = Ax(p)
        alpha = r_dot_r / (torch.dot(p, Ap) + eps)
        x += alpha * p
        r -= alpha * Ap
        new_r_dot_r = torch.dot(r, r)
        if new_r_dot_r < residual_tol:
            break
        beta = new_r_dot_r / r_dot_r
        p = r + beta * p
        r_dot_r = new_r_dot_r
    return x


def fisher_vector_product(policy: GaussianPolicy, old_policy: GaussianPolicy, obs, damping=1e-3):
    """
    Returns a function that computes F @ v, where F is the Fisher information matrix.
    Uses Pearlmutter's trick: Fv = ∇_θ [ v^T ∇_θ KL(π_old || π_θ) ]
    """
    def Fv(v):
        kl = old_policy.kl_divergence(obs, policy).mean()
        grad_kl = torch.autograd.grad(kl, list(policy.parameters()), create_graph=True)
        flat_grad_kl = flatten_params(grad_kl)
        v_dot_grad = (flat_grad_kl * v).sum()
        hvp = torch.autograd.grad(v_dot_grad, list(policy.parameters()))
        hvp_flat = flatten_params(hvp).detach()
        return hvp_flat + damping * v

    return Fv


def trpo_update(
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    value_func_optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    acts: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    rets: torch.Tensor,
    delta=0.01,
    cg_iters=10,
    max_backtrack=10,
    backtrack_coeff=0.8,
    damping=1e-3,
    val_func_epochs=10,
    writer=None,
    global_step: int = 0,
):
    """
    Perform one TRPO policy update and value function update.
    """
    # Move all tensors to device
    obs = obs.to(device)
    acts = acts.to(device)
    logp_old = logp_old.to(device)
    adv = adv.to(device)
    rets = rets.to(device)
    
    # Ensure advantage is normalized
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    # Freeze a copy of the current policy as "old" for KL computations
    old_policy = copy.deepcopy(policy)
    for p in old_policy.parameters():
        p.requires_grad_(False)

    # Step 1: Compute policy gradient g
    logp_new = policy.log_prob(obs, acts)
    ratio = torch.exp(logp_new - logp_old)
    surr_loss = -(ratio * adv).mean()
    surr_grad = torch.autograd.grad(surr_loss, list(policy.parameters()))
    g = flatten_params(surr_grad).detach()
    g = -g  # because we minimized loss; we want gradient of reward

    # Step 2: compute search direction (NPG direction) using CG
    Fv = fisher_vector_product(policy, old_policy, obs, damping=damping)
    x = conjugate_gradient(Fv, g, cg_iters=cg_iters)

    # Step 3: Compute optimal step size (shrink to satisfy KL ~ \delta)
    Fx = Fv(x)
    beta = torch.sqrt(2 * delta / (torch.dot(x, Fx) + 1e-8))
    theta_init = flatten_params(policy.parameters()).detach()

    # Step 4: Backtracking linesearch
    old_params = theta_init.clone()
    expected_improvement = float(torch.dot(g, x).item())
    success = False
    
    for i in range(max_backtrack):
        alpha = backtrack_coeff**i
        theta_new = theta_init + alpha * beta * x
        load_flat_params(policy, theta_new)

        # Evaluate new policy
        with torch.no_grad():
            logp_new = policy.log_prob(obs, acts)
            ratio = torch.exp(logp_new - logp_old)
            surr_loss_new = -(ratio * adv).mean()
            kl_div = old_policy.kl_divergence(obs, policy).mean()

        actual_improvement = float((surr_loss - surr_loss_new).item())
        if kl_div.item() <= delta and actual_improvement >= 0.1 * alpha * expected_improvement:
            # Accept update
            success = True
            break

    if not success:
        load_flat_params(policy, old_params)
        kl_div = torch.tensor(0.0)
        actual_improvement = 0.0

    # Step 5: Fit value function with regression for several epochs
    for _ in range(val_func_epochs):
        v_pred = value_net(obs)
        value_loss = F.mse_loss(v_pred, rets)
        value_func_optimizer.zero_grad()
        value_loss.backward()
        value_func_optimizer.step()

    # Log metrics to TensorBoard
    if writer is not None:
        writer.add_scalar("Loss/Surrogate", surr_loss.item(), global_step)
        writer.add_scalar("Loss/Value", value_loss.item(), global_step)
        writer.add_scalar("Policy/KL_Divergence", kl_div.item() if success else 0.0, global_step)
        writer.add_scalar("Policy/Entropy", -logp_new.mean().item(), global_step)
        writer.add_scalar("Policy/Actual_Improvement", actual_improvement if success else 0.0, global_step)

    return success


def evaluate_policy(env, policy, num_episodes=3):
    """
    Evaluate the policy by running episodes.
    
    Args:
        env: Single environment (not vectorized), can be wrapped with RecordVideo
        policy: Policy network
        num_episodes: Number of episodes to run
    
    Returns:
        Average episode return
    """
    episode_returns = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        step_count = 0
        
        with tqdm(total=200, desc="Evaluation", leave=False) as pbar:
            while not done and step_count < 200:  # Max 1000 steps per episode
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _ = policy.get_action(obs_tensor)
                    action = action.cpu().numpy()[0]
                    
                    # Clip actions if needed
                    if isinstance(env.action_space, gym.spaces.Box):
                        action = np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
                step_count += 1
                pbar.update(1)
        
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        print(f"  Episode {ep+1}/{num_episodes}: Return = {episode_return:.2f}, Length = {step_count}")
    
    avg_return = np.mean(episode_returns)
    avg_length = np.mean(episode_lengths)
    
    return avg_return, avg_length


def train_trpo_quadruped(
    vec_env: gym.vector.SyncVectorEnv,
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    epochs=5000,
    steps_per_epoch=200,
    gamma=0.99,
    lam=0.97,
    seed=42,
    log_dir="runs/trpo_quadruped",
    eval_env=None,
    eval_freq=100,
    save_freq=100
):
    """
    Train TRPO on quadruped locomotion task.
    
    Args:
        vec_env: Vectorized quadruped environment
        policy: Policy network
        value_net: Value network
        epochs: Number of training epochs
        steps_per_epoch: Number of environment steps per epoch (per environment)
        gamma: Discount factor
        lam: GAE lambda parameter
        seed: Random seed
        log_dir: Directory for TensorBoard logs
        eval_env: Optional separate environment for evaluation/video recording
        eval_freq: How often (in epochs) to evaluate and record videos
        save_freq: How often (in epochs) to save checkpoints
    """
    # Move networks to device
    policy = policy.to(device)
    value_net = value_net.to(device)
    
    value_func_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)

    run_name = f"go1_{datetime.now().strftime('%m_%d_%H_%M_%S')}"
    run_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir=run_dir)
    
    global_step = 0
    num_envs = vec_env.num_envs
    
    # Extract observation and action dimensions
    obs_dim, act_dim = extract_env_dims(vec_env)

    print(f"\n{'='*70}")
    print(f"Starting TRPO Training - Quadruped Locomotion")
    print(f"{'='*70}")
    print(f"Environments: {num_envs}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps per update: {steps_per_epoch * num_envs}")
    print(f"Log directory: {run_dir}")
    print(f"{'='*70}\n")

    # Reset environments
    obs, _ = vec_env.reset(seed=seed)

    for epoch in tqdm(range(epochs), desc="Training"):
        buffer = RolloutBuffer(steps_per_epoch, num_envs, obs_dim, act_dim)
        rewbuffer = deque(maxlen=num_envs)
        
        ep_rews = np.zeros(num_envs)
        ep_lens = np.zeros(num_envs)

        # ---------- Collect Rollout ----------
        for t in range(steps_per_epoch):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                actions, logp_t = policy.get_action(obs_tensor)
                acts = actions.cpu().numpy()
                vals = value_net(obs_tensor)

            # Step environment
            next_obs, rews, terminateds, truncateds, infos = vec_env.step(acts)
            dones = np.logical_or(terminateds, truncateds)
            new_ids = (dones > 0).nonzero()
            
            # print(f"new_ids: {new_ids}")

            ep_rews += rews
            ep_lens += 1

            rewbuffer.extend(ep_rews[new_ids])
            ep_rews[new_ids] = 0.0
            ep_lens[new_ids] = 0

            # Store transitions
            buffer.add(t, obs_tensor, actions, logp_t, rews, vals, dones)

            # Check for completed episodes
            # for i, info in enumerate(infos):
            #     if 'episode' in info:
            #         episode_info = info.get('episode', {})  # type: ignore[assignment]
            #         if episode_info:
            #             writer.add_scalar("Rollout/Episode_Return", episode_info.get('r', 0.0), global_step)  # type: ignore[arg-type]
            #             writer.add_scalar("Rollout/Episode_Length", episode_info.get('l', 0), global_step)  # type: ignore[arg-type]

            obs = next_obs
            global_step += num_envs



        # ---------- Compute advantages and returns ----------
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            last_vals = value_net(obs_tensor)
    
        adv, returns = buffer.compute_gae(last_vals, gamma, lam)

        # Flatten buffers from (T, E, ...) to (T*E, ...) for TRPO update
        obs_flat = buffer.obs.reshape(-1, obs_dim)
        acts_flat = buffer.acts.reshape(-1, act_dim)
        logps_flat = buffer.logps.reshape(-1)
        adv_flat = adv.reshape(-1)
        returns_flat = returns.reshape(-1)

        # ---------- TRPO Update ----------
        success = trpo_update(
            policy,
            value_net,
            value_func_optimizer,
            obs=obs_flat,
            acts=acts_flat,
            logp_old=logps_flat,
            adv=adv_flat,
            rets=returns_flat,
            writer=writer,
            global_step=global_step,
        )

        # ---------- Logging ----------
        try:
            avg_ep_rew = statistics.mean(rewbuffer)
        except Exception as e:
            avg_ep_rew = 0.0
        # avg_ep_rew = np.mean(ep_rews)

        writer.add_scalar("Rollout/Epoch_Reward", avg_ep_rew, epoch)
        
        # Log additional metrics
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Avg Epoch Reward: {avg_ep_rew:.2f}")
            print(f"  TRPO Update: {'Success' if success else 'Failed'}")
            print(f"  Global Step: {global_step}")

        # Periodic evaluation with video recording
        if eval_env is not None and (epoch + 1) % eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at Epoch {epoch+1}:")
            
            # Wrap environment with video recorder
            video_env = gym.wrappers.RecordVideo(
                eval_env,
                video_folder=run_dir,
                episode_trigger=lambda x: True,  # Record all episodes
                name_prefix=f"epoch_{epoch+1}"
            )
            
            avg_eval_return, avg_eval_length = evaluate_policy(video_env, policy, num_episodes=1)
            print(f"Average Evaluation Return: {avg_eval_return:.2f}")
            print(f"Average Evaluation Length: {avg_eval_length:.1f}")
            print(f"Videos saved to: {run_dir}/")
            print(f"{'='*60}\n")
            
            video_env.close()
            writer.add_scalar("Eval/Average_Return", avg_eval_return, epoch)
            writer.add_scalar("Eval/Average_Length", avg_eval_length, epoch)

        # Periodic checkpoint saving
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = os.path.join(run_dir, f"policy_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'policy_state_dict': policy.state_dict(),
                'value_state_dict': value_net.state_dict(),
                'optimizer_state_dict': value_func_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint = os.path.join(run_dir, "policy_final.pth")
    torch.save({
        'epoch': epochs,
        'policy_state_dict': policy.state_dict(),
        'value_state_dict': value_net.state_dict(),
        'optimizer_state_dict': value_func_optimizer.state_dict(),
    }, final_checkpoint)
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")
    
    writer.close()
    print(f"Training complete! Logs saved to {run_dir}")

    return run_dir


@hydra.main(version_base=None, config_path="configs", config_name="config_standard")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print("TRPO Training for Unitree Go1 Quadruped Locomotion")
    print("=" * 70)
    
    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    
    print(f"\nTraining Summary:")
    print(f"  Model: Unitree Go1")
    print(f"  Number of parallel environments: {cfg.train.num_envs}")
    print(f"  Hidden dimension: {cfg.train.hidden_dim}")
    print(f"  Epochs: {cfg.train.epochs}")
    print(f"  Steps per epoch: {cfg.train.steps_per_epoch}")
    print(f"  Total steps per update: {cfg.train.steps_per_epoch * cfg.train.num_envs}")
    print(f"  Evaluation frequency: every {cfg.train.eval_freq} epochs")
    print(f"  Save frequency: every {cfg.train.save_freq} epochs")
    print("=" * 70)
    
    # Create vectorized training environment
    print("\nCreating training environment...")
    reward_weights = dict(cfg.reward) if 'reward' in cfg else None
    
    vec_env = make_quadruped_env(
        num_envs=cfg.train.num_envs, 
        model_path=os.path.join(cfg.env.model_dir, cfg.env.model_file),  
        timestep=cfg.env.timestep,
        frame_skip=cfg.env.frame_skip, 
        max_episode_steps=cfg.env.max_episode_steps,
        reward_weights=reward_weights,
        damping_scale=cfg.env.damping_scale,
        stiffness_scale=cfg.env.stiffness_scale,
    )

    # Create separate environment for evaluation/video recording
    print("Creating evaluation environment...")
    camera_mode = cfg.env.get('camera_mode', 'follow')  # Default to 'follow' if not specified
    eval_env = QuadrupedEnv(
        model_path=os.path.join(cfg.env.model_dir, cfg.env.model_file),
        render_mode="rgb_array",
        max_episode_steps=cfg.env.max_episode_steps,
        frame_skip=cfg.env.frame_skip,
        timestep=cfg.env.timestep,
        reward_weights=reward_weights,
        damping_scale=cfg.env.damping_scale,
        stiffness_scale=cfg.env.stiffness_scale,
        camera_mode=camera_mode,  # Camera follows robot during video recording
    )
    
    # Get environment dimensions
    obs_dim, act_dim = extract_env_dims(vec_env)
    
    print(f"\nEnvironment specifications:")
    print(f"  Observation dimension: {obs_dim}")
    print(f"  Action dimension: {act_dim}")
    print("=" * 70)
    
    # Create policy and value networks
    print("\nInitializing networks...")
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=cfg.train.hidden_dim)
    value_net = ValueNetwork(obs_dim, hidden_dim=cfg.train.hidden_dim)
    
    print(f"  Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print(f"  Value network parameters: {sum(p.numel() for p in value_net.parameters())}")
    print("=" * 70)
    
    # Train
    print("\nStarting training...")
    run_dir = train_trpo_quadruped(
        vec_env=vec_env,
        policy=policy,
        value_net=value_net,
        epochs=cfg.train.epochs,
        steps_per_epoch=cfg.train.steps_per_epoch,
        gamma=cfg.train.gamma,
        lam=cfg.train.lam,
        log_dir=cfg.logging.log_dir,
        eval_env=eval_env,
        eval_freq=cfg.train.eval_freq,
        save_freq=cfg.train.save_freq
    )
    
    # Save the config as a YAML file in the run_dir
    config_save_path = os.path.join(run_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved configuration to {config_save_path}")
    
    # Final evaluation with video
    print("\n" + "=" * 70)
    print("Final Evaluation:")
    final_video_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder=run_dir,
        episode_trigger=lambda x: True,
        name_prefix="final"
    )
    final_return, final_length = evaluate_policy(final_video_env, policy, num_episodes=5)
    print(f"Final Average Return: {final_return:.2f}")
    print(f"Final Average Length: {final_length:.1f}")
    print(f"Final videos saved to: {run_dir}/")
    print("=" * 70)
    
    final_video_env.close()
    eval_env.close()
    vec_env.close()
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"View logs with: tensorboard --logdir={cfg.logging.log_dir}")
    print(f"View videos in: {run_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

