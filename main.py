import os
os.environ["MUJOCO_GL"] = "egl"

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
import numpy as np
import copy

from actor_critic import GaussianPolicy, ValueNetwork
from data_collection import RolloutBuffer

from tqdm import tqdm
from datetime import datetime
from collections import deque
import argparse

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def make_env(env_id):
    return gym.make(env_id)


def flatten_params(params) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in params])


def load_flat_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        NP = p.numel()
        p.data.copy_(flat_params[idx : idx + NP].view(p.shape))
        idx += NP


def conjugate_gradient(Ax, b, cg_iters=10, residual_tol=1e-10, eps=1e-8):
    """
    Solve Ax = b using the conjugate gradient method.
    Ax: A function that computes A @ x
    b: right-hand side vector
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
    - policy: current policy (will be updated in-place)
    - value_net: value function network (trained with standard MSE)
    - optimizer_value: optimizer for value network
    - obs, acts, old_logps, adv, returns: tensors (batch)
    - writer: TensorBoard writer for logging
    - global_step: current global step for logging
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
        print("Backtracking linesearch failed. No policy update applied")
    else:
        pass
        # print(
        #     f"TRPO update successful. KL={kl_div.item():.6f}, actual={actual_improvement:.6f}"
        # )

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
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0
        step_count = 0
        
        while not done and step_count < 1000:  # Max 1000 steps per episode
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
        
        episode_returns.append(episode_return)
        print(f"  Episode {ep+1}/{num_episodes}: Return = {episode_return:.2f}")
    
    avg_return = np.mean(episode_returns)
    return avg_return


def train_trpo(
    envs: gym.vector.SyncVectorEnv,
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    epochs=1000,
    steps_per_epoch=128,
    gamma=0.99,
    lam=0.97,
    seed=42,
    n_envs=32,
    n_obs=4,
    n_acts=2,
    log_dir="runs/trpo",
    eval_env=None,
    eval_freq=50,
    env_id=None,  # Environment ID for creating fresh eval environments
):
    """
    Train TRPO with vectorized environments.
    
    Args:
        env: Vectorized environment (gym.vector.AsyncVectorEnv or SyncVectorEnv)
        policy: Policy network
        value_net: Value network
        epochs: Number of training epochs
        steps_per_epoch: Number of environment steps per epoch (per environment)
        gamma: Discount factor
        lam: GAE lambda parameter
        log_dir: Directory for TensorBoard logs
        eval_env: Optional separate environment for evaluation/video recording
        eval_freq: How often (in epochs) to evaluate and record videos
        video_folder: Directory to save evaluation videos
    """
    # Move networks to device
    policy = policy.to(device)
    value_net = value_net.to(device)
    
    value_func_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)

    run_name = f"{datetime.now().strftime('%m_%d_%H_%M_%S')}_envs{n_envs}_hzn{steps_per_epoch}"
    run_dir = os.path.join(log_dir, run_name)
    writer = SummaryWriter(log_dir=run_dir)
    
    # Episode tracking
    # episode_rewards = [[] for _ in range(num_envs)]
    # episode_lengths = [0 for _ in range(num_envs)]
    # completed_episodes = []
    
    completed_episode_returns = deque(maxlen=n_envs)
    global_step = 0
    obs, _ = envs.reset(seed=seed)
    ep_rews = np.zeros(n_envs)

    for epoch in tqdm(range(epochs), desc="Training"):
        buffer = RolloutBuffer(steps_per_epoch, n_envs, n_obs, n_acts)

        # ---------- Collect Rollout ----------
        for t in range(steps_per_epoch):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                actions, logp_t = policy.get_action(obs_tensor)
                acts = actions.cpu().numpy()
                # Clip actions if the action space is bounded
                if isinstance(envs.single_action_space, gym.spaces.Box):
                    acts = np.clip(acts, envs.single_action_space.low, envs.single_action_space.high)
                vals = value_net(obs_tensor)

            # Step environment
            next_obs, rews, terminateds, truncateds, infos = envs.step(acts)
            dones = np.logical_or(terminateds, truncateds)

            ep_rews += rews
            # Find environments that just completed episodes
            new_ids = np.where(dones)[0]  # More explicit than nonzero() for 1D arrays
            if len(new_ids) > 0:
                # Save completed episode returns before resetting
                completed_episode_returns.extend(ep_rews[new_ids].tolist())
                # Reset rewards for completed episodes
                ep_rews[new_ids] = 0.0
            
            # Store transitions for each environment
            buffer.add(t, obs_tensor, actions, logp_t, rews, vals, dones)

            obs = next_obs
            global_step += n_envs

        # ---------- Compute advantages and returns ----------
        
        # Bootstrap value for last state (if not done)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            last_vals = value_net(obs_tensor)
    
        adv, returns = buffer.compute_gae(last_vals, gamma, lam)

        # Flatten buffers from (T, E, ...) to (T*E, ...) for TRPO update
        obs_flat = buffer.obs.reshape(-1, buffer.obs.shape[-1])
        acts_flat = buffer.acts.reshape(-1, buffer.acts.shape[-1])
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
        # Log completed episode returns (more meaningful than incomplete episodes)
        if len(completed_episode_returns) > 0:
            avg_completed_return = np.mean(completed_episode_returns)
        else:
            avg_completed_return = 0.0
        
        writer.add_scalar("Rollout/Episode_Return", avg_completed_return, epoch)
        # writer.add_scalar("Rollout/Num_Completed_Episodes", len(completed_episode_returns), epoch)
        
        # Also log current (incomplete) episode rewards for reference
        # avg_ep_rew = np.mean(ep_rews)
        # writer.add_scalar("Rollout/Epoch_Reward", avg_ep_rew, epoch)

        # Periodic evaluation with video recording
        if eval_env is not None and (epoch + 1) % eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluating at Epoch {epoch+1}:")
            
            # Create a fresh environment for evaluation to avoid renderer corruption
            # This is important for MuJoCo environments with EGL rendering
            # The renderer can get corrupted when RecordVideo wrapper is closed/reused
            if env_id is not None:
                fresh_eval_env = gym.make(env_id, render_mode="rgb_array")
            else:
                # Fallback: try to get env_id from eval_env spec
                try:
                    # Unwrap to get the base environment if wrapped
                    base_env = eval_env
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    env_id_from_spec = base_env.spec.id if hasattr(base_env, 'spec') and base_env.spec else None
                    if env_id_from_spec:
                        fresh_eval_env = gym.make(env_id_from_spec, render_mode="rgb_array")
                    else:
                        # Last resort: use the existing eval_env (may have issues)
                        fresh_eval_env = eval_env
                except Exception as e:
                    print(f"Warning: Could not create fresh eval env: {e}. Using existing env.")
                    fresh_eval_env = eval_env
            
            # Wrap environment with video recorder
            video_env = gym.wrappers.RecordVideo(
                fresh_eval_env,
                video_folder=run_dir,
                episode_trigger=lambda x: True,  # Record all episodes
                name_prefix=f"epoch_{epoch+1}"
            )
            
            avg_eval_return = evaluate_policy(video_env, policy, num_episodes=1)
            print(f"Average Evaluation Return: {avg_eval_return:.2f}")
            print(f"Videos saved to: {run_dir}/")
            print(f"{'='*60}\n")
            
            video_env.close()
            # Only close if we created a fresh environment
            if fresh_eval_env is not eval_env:
                fresh_eval_env.close()
            writer.add_scalar("Eval/Average_Return", avg_eval_return, epoch)
            
            # Save checkpoint
            checkpoint_path = os.path.join(run_dir, f"model_{epoch+1}.pth")
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")

        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch}, Global Step {global_step}, TRPO success: {success}, Avg Reward: {avg_ep_rew:.2f}")
    
    # Save final checkpoint
    final_checkpoint = os.path.join(run_dir, "model_final.pth")
    torch.save(policy.state_dict(), final_checkpoint)
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")
    
    writer.close()
    print(f"Training complete! Logs saved to {log_dir}")

    return run_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="BipedalWalker-v3")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_freq", type=int, default=50)

    args = parser.parse_args()

    print("=" * 60)
    print(f"Training TRPO on {args.env_id}")
    print("=" * 60)
    
    # Create vectorized environment for training
    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make(args.env_id) for _ in range(args.num_envs)
    ])
    
    # Create separate environment for evaluation/video recording
    # Use render_mode="rgb_array" for video recording (works on remote servers)
    eval_env = gym.make(args.env_id, render_mode="rgb_array")
    
    # Get environment dimensions
    if args.env_id == "BipedalWalker-v3":
        obs_dim = 24
        act_dim = 4
    elif args.env_id == "Hopper-v5":
        obs_dim = 11
        act_dim = 3
    elif args.env_id == "Swimmer-v5":
        obs_dim = 8
        act_dim = 2
    elif args.env_id == "Walker2d-v5":
        obs_dim = 17
        act_dim = 6
    elif args.env_id == "InvertedPendulum-v5":
        obs_dim = 4
        act_dim = 1
    else:
        raise ValueError(f"Environment {args.env_id} not supported")

    print(f"Environment: {args.env_id}")
    print(f"Number of parallel environments: {args.num_envs}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")
    print(f"Video recording: Enabled (every 50 epochs)")
    print("=" * 60)
    
    # Create policy and value networks
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=64)
    value_net = ValueNetwork(obs_dim, hidden_dim=64)
    
    LOG_DIR = f"runs/{args.env_id}"
    # Train
    run_dir = train_trpo(
        envs=envs,
        policy=policy,
        value_net=value_net,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        gamma=args.gamma,
        lam=args.lam,
        log_dir=LOG_DIR,
        n_envs=args.num_envs,
        n_obs=obs_dim,
        n_acts=act_dim,
        eval_env=eval_env,
        eval_freq=args.eval_freq,  # Evaluate and record video every 40 epochs
        seed=args.seed,
        env_id=args.env_id,  # Pass env_id for creating fresh eval environments
    )
    
    # Final evaluation with video
    print("\n" + "=" * 60)
    print("Final Evaluation:")
    final_video_env = gym.wrappers.RecordVideo(
        eval_env,
        video_folder=run_dir,
        episode_trigger=lambda x: True,
        name_prefix="final"
    )
    final_return = evaluate_policy(final_video_env, policy, num_episodes=5)
    print(f"Final Average Return: {final_return:.2f}")
    print(f"Final videos saved to: {run_dir}/")
    print("=" * 60)
    
    final_video_env.close()
    eval_env.close()
    envs.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"View logs with: tensorboard --logdir={LOG_DIR}")
    print(f"View videos in: {run_dir}/")
    print("=" * 60)
