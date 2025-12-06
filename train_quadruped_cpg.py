#!/usr/bin/env python3
"""
TRPO Training Script for CPG-Based Quadruped Locomotion (PMTG Approach).

This script trains a policy that modulates a Central Pattern Generator (CPG)
instead of directly controlling joints. This should lead to faster learning
of natural walking gaits.
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
import statistics
import hydra
from omegaconf import DictConfig, OmegaConf

from actor_critic import CPGModulatingPolicy, ValueNetwork
from data_collection import RolloutBuffer
from quadruped_env_cpg import QuadrupedEnvCPG

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
    """Solve Ax = b using conjugate gradient method."""
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


def fisher_vector_product(policy, old_policy, obs, damping=1e-3):
    """Returns a function that computes F @ v (Fisher-vector product)."""
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
    policy,
    value_net,
    value_func_optimizer,
    obs,
    acts,
    logp_old,
    adv,
    rets,
    delta=0.01,
    cg_iters=10,
    max_backtrack=10,
    backtrack_coeff=0.8,
    damping=1e-3,
    val_func_epochs=10,
    writer=None,
    global_step=0,
):
    """Perform one TRPO policy update and value function update."""
    obs = obs.to(device)
    acts = acts.to(device)
    logp_old = logp_old.to(device)
    adv = adv.to(device)
    rets = rets.to(device)
    
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    old_policy = copy.deepcopy(policy)
    for p in old_policy.parameters():
        p.requires_grad_(False)

    # Compute policy gradient
    logp_new = policy.log_prob(obs, acts)
    ratio = torch.exp(logp_new - logp_old)
    surr_loss = -(ratio * adv).mean()
    surr_grad = torch.autograd.grad(surr_loss, list(policy.parameters()))
    g = flatten_params(surr_grad).detach()
    g = -g

    # Compute search direction using CG
    Fv = fisher_vector_product(policy, old_policy, obs, damping=damping)
    x = conjugate_gradient(Fv, g, cg_iters=cg_iters)

    # Compute step size
    Fx = Fv(x)
    beta = torch.sqrt(2 * delta / (torch.dot(x, Fx) + 1e-8))
    theta_init = flatten_params(policy.parameters()).detach()

    # Backtracking line search
    old_params = theta_init.clone()
    expected_improvement = float(torch.dot(g, x).item())
    success = False
    
    for i in range(max_backtrack):
        alpha = backtrack_coeff**i
        theta_new = theta_init + alpha * beta * x
        load_flat_params(policy, theta_new)

        with torch.no_grad():
            logp_new = policy.log_prob(obs, acts)
            ratio = torch.exp(logp_new - logp_old)
            surr_loss_new = -(ratio * adv).mean()
            kl_div = old_policy.kl_divergence(obs, policy).mean()

        actual_improvement = float((surr_loss - surr_loss_new).item())
        if kl_div.item() <= delta and actual_improvement >= 0.1 * alpha * expected_improvement:
            success = True
            break

    if not success:
        load_flat_params(policy, old_params)
        kl_div = torch.tensor(0.0)
        actual_improvement = 0.0

    # Fit value function
    for _ in range(val_func_epochs):
        v_pred = value_net(obs)
        value_loss = F.mse_loss(v_pred, rets)
        value_func_optimizer.zero_grad()
        value_loss.backward()
        value_func_optimizer.step()

    # Log metrics
    if writer is not None:
        writer.add_scalar("Loss/Surrogate", surr_loss.item(), global_step)
        writer.add_scalar("Loss/Value", value_loss.item(), global_step)
        writer.add_scalar("Policy/KL_Divergence", kl_div.item() if success else 0.0, global_step)
        writer.add_scalar("Policy/Entropy", -logp_new.mean().item(), global_step)
        writer.add_scalar("Policy/Actual_Improvement", actual_improvement if success else 0.0, global_step)

    return success


def evaluate_policy(env, policy, num_episodes=3):
    """Evaluate the policy by running episodes."""
    episode_returns = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        step_count = 0
        
        with tqdm(total=200, desc=f"Evaluation Ep {ep+1}", leave=False) as pbar:
            while not done and step_count < 200:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _ = policy.get_action(obs_tensor)
                    action = action.cpu().numpy()[0]
                    
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


def train_trpo_cpg(
    policy,
    value_net,
    model_path: str,
    epochs=2000,
    steps_per_epoch=200,
    num_envs=4,
    gamma=0.99,
    lam=0.97,
    seed=42,
    log_dir="runs/trpo_quadruped",
    eval_freq=50,
    save_freq=50,
    timestep=1 / 200,
    max_episode_steps=3000,
    frame_skip=5,
    reward_weights=None,
    gait_type='trot',
    run_prefix='',
    damping_scale=1.0,
    stiffness_scale=1.0,
    obs_dim=-1,
    act_dim=-1,
    camera_mode='follow',  # Camera mode for video recording: 'follow', 'fixed', 'side', 'top'
):
    """
    Train TRPO with CPG-based control.
    
    Note: We use separate single environments instead of vectorized
    for CPG because it's simpler and CPG already provides structure.
    """
    policy = policy.to(device)
    value_net = value_net.to(device)
    
    value_func_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)

    timestamp = datetime.now().strftime('%m_%d_%H_%M_%S')
    run_name = f"{timestamp}_{run_prefix}_fskip{frame_skip}_ts{timestep}_envs{num_envs}_hzn{steps_per_epoch}"
    run_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir=run_dir)
    
    # Create training environments
    train_envs = gym.vector.SyncVectorEnv([
        lambda: QuadrupedEnvCPG(
            model_path=model_path,
            gait_type=gait_type,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            timestep=timestep,
            reward_weights=reward_weights,
            damping_scale=damping_scale,
            stiffness_scale=stiffness_scale,
        )
        for _ in range(num_envs)
    ])
    
    # Create evaluation environment with camera following enabled
    eval_env = QuadrupedEnvCPG(
        model_path=model_path,
        gait_type=gait_type,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        frame_skip=frame_skip,  # Smoother video
        timestep=timestep,
        reward_weights=reward_weights,
        damping_scale=damping_scale,
        stiffness_scale=stiffness_scale,
        camera_mode=camera_mode,  # Camera follows robot during video recording
    )

    print(f"\n{'='*70}")
    print(f"Starting TRPO Training - CPG-Based Quadruped Locomotion")
    print(f"{'='*70}")
    print(f"Training environments: {num_envs}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim} (CPG parameters)")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Log directory: {run_dir}")
    print(f"{'='*70}\n")

    # Reset environments
    global_step = 0
    env_obs, infos = train_envs.reset(seed=seed)
    ep_rews = np.zeros(num_envs)
    ep_lens = np.zeros(num_envs)
    rewbuffer = deque(maxlen=10)

    for epoch in tqdm(range(epochs), desc="Training"):
        buffer = RolloutBuffer(steps_per_epoch, num_envs, obs_dim, act_dim)
        
        # Collect rollouts
        for t in range(steps_per_epoch):
            obs_tensor = torch.tensor(env_obs, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                actions, logp_t = policy.get_action(obs_tensor)
                acts = actions.cpu().numpy()
                vals = value_net(obs_tensor)

            # Step environments
            # for i, env in enumerate(train_envs):
            next_obs, rews, terminateds, truncateds, infos = train_envs.step(acts)
            dones = np.logical_or(terminateds, truncateds)
            
            ep_rews += rews
            ep_lens += 1
            
            new_ids = np.where(dones)[0]
            if len(new_ids) > 0:
                rewbuffer.extend(ep_rews[new_ids].tolist())
                ep_rews[new_ids] = 0.0
                ep_lens[new_ids] = 0
            
            # Store transitions for each environment
            buffer.add(t, obs_tensor, actions, logp_t, rews, vals, dones)
            
            env_obs = next_obs
            global_step += num_envs

        # ----------- Compute advantages and returns -----------
        with torch.no_grad():
            obs_tensor = torch.tensor(np.array(env_obs), dtype=torch.float32).to(device)
            last_vals = value_net(obs_tensor)
    
        adv, returns = buffer.compute_gae(last_vals, gamma, lam)

        # Flatten buffers
        obs_flat = buffer.obs.reshape(-1, obs_dim)
        acts_flat = buffer.acts.reshape(-1, act_dim)
        logps_flat = buffer.logps.reshape(-1)
        adv_flat = adv.reshape(-1)
        returns_flat = returns.reshape(-1)

        # TRPO update
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

        # Logging
        try:
            avg_ep_rew = statistics.mean(rewbuffer)
        except:
            avg_ep_rew = 0.0

        writer.add_scalar("Rollout/Epoch_Reward", avg_ep_rew, epoch)
        writer.add_scalar("Rewards/reward_forward", infos['reward_forward'].mean(), epoch)
        writer.add_scalar("Rewards/reward_alive", infos['reward_alive'].mean(), epoch)
        writer.add_scalar("Rewards/reward_smoothness", infos['reward_smoothness'].mean(), epoch)
        writer.add_scalar("Rewards/reward_orientation", infos['reward_orientation'].mean(), epoch)
        writer.add_scalar("Rewards/reward_energy", infos['reward_energy'].mean(), epoch)
        writer.add_scalar("Rewards/reward_joint_limits", infos['reward_joint_limits'].mean(), epoch)
        writer.add_scalar("Rewards/reward_height", infos['reward_height'].mean(), epoch)
        writer.add_scalar("Rewards/reward_lateral", infos['reward_lateral'].mean(), epoch)
        writer.add_scalar("Rewards/reward_angular", infos['reward_angular'].mean(), epoch)
        writer.add_scalar("State/forward_velocity", infos['forward_velocity'].mean(), epoch)
        writer.add_scalar("State/base_height", infos['base_height'].mean(), epoch)
        writer.add_scalar("State/orientation_w", infos['orientation_w'].mean(), epoch)
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Avg Epoch Reward: {avg_ep_rew:.2f}")
            print(f"  TRPO Update: {'Success' if success else 'Failed'}")
            print(f"  Global Step: {global_step}")

        # Periodic evaluation
        if (epoch + 1) % eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at Epoch {epoch+1}:")
            
            video_env = gym.wrappers.RecordVideo(
                eval_env,
                video_folder=run_dir,
                episode_trigger=lambda x: True,
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
    
    # Cleanup
    
    train_envs.close()
    eval_env.close()
    writer.close()
    
    print(f"Training complete! Logs saved to {run_dir}")
    return run_dir


@hydra.main(version_base=None, config_path="configs", config_name="config_cpg")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print("TRPO Training for CPG-Based Quadruped Locomotion (PMTG)")
    print("=" * 70)
    
    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 70)
    
    print(f"\nTraining Summary:")
    print(f"  Model: Unitree Go1")
    print(f"  Control: CPG-based (PMTG)")
    print(f"  Training environments: {cfg.train.num_envs}")
    print(f"  Hidden dimension: {cfg.train.hidden_dim}")
    print(f"  Epochs: {cfg.train.epochs}")
    print(f"  Steps per epoch: {cfg.train.steps_per_epoch}")
    print(f"  Evaluation frequency: every {cfg.train.eval_freq} epochs")
    print("=" * 70)
    
    # Create policy and value networks
    print("\nInitializing networks...")
    obs_dim = 34  # Quadruped observation
    # act_dim = 16  # CPG parameters
    act_dim = 12  # Residual action
    
    policy = CPGModulatingPolicy(obs_dim, act_dim, hidden_dim=cfg.train.hidden_dim)
    value_net = ValueNetwork(obs_dim, hidden_dim=cfg.train.hidden_dim)
    
    print(f"  Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print(f"  Value network parameters: {sum(p.numel() for p in value_net.parameters())}")
    print(f"  Action space: 12D residual action")
    print("=" * 70)
    
    # Train
    print("\nStarting training...")
    camera_mode = cfg.env.get('camera_mode', 'follow')  # Default to 'follow' if not specified
    run_dir = train_trpo_cpg(
        policy=policy,
        value_net=value_net,
        model_path=os.path.join(cfg.env.model_dir, cfg.env.model_file),
        epochs=cfg.train.epochs,
        steps_per_epoch=cfg.train.steps_per_epoch,
        num_envs=cfg.train.num_envs,
        gamma=cfg.train.gamma,
        lam=cfg.train.lam,
        log_dir=cfg.logging.log_dir,
        eval_freq=cfg.train.eval_freq,
        save_freq=cfg.train.save_freq,
        timestep=cfg.env.timestep,
        max_episode_steps=cfg.env.max_episode_steps,
        frame_skip=cfg.env.frame_skip,
        reward_weights=dict(cfg.reward) if 'reward' in cfg else None,
        gait_type=cfg.env.gait_type,
        run_prefix=cfg.env.run_prefix,
        damping_scale=cfg.env.damping_scale,
        stiffness_scale=cfg.env.stiffness_scale,
        obs_dim=obs_dim,
        act_dim=act_dim,
        camera_mode=camera_mode,
    )

    # Save the config as a YAML file in the run_dir
    config_save_path = os.path.join(run_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Saved configuration to {config_save_path}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"View logs with: tensorboard --logdir={cfg.logging.log_dir}")
    print(f"View videos in: {run_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

