#!/usr/bin/env python3
"""
Testing script for CPG-based Quadruped Policy.

This script loads a trained policy checkpoint and evaluates it with video recording.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering

import torch
import numpy as np
import argparse
import gymnasium as gym
from pathlib import Path
from omegaconf import OmegaConf

from actor_critic import CPGModulatingPolicy, ValueNetwork
from quadruped_env_cpg import QuadrupedEnvCPG

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = OmegaConf.load(f)
    return config


def test_policy(
    run_name: str,
    checkpoint: str,
    log_dir: str = "runs/trpo_quadruped",
    num_episodes: int = 5,
    camera_mode: str = "follow",
    output_dir: str | None = None,
):
    """
    Test a trained CPG policy.
    
    Args:
        run_name: Name of the training run (directory name)
        checkpoint: Checkpoint filename (e.g., "policy_epoch_100.pth" or "policy_final.pth")
        log_dir: Base directory containing runs
        num_episodes: Number of episodes to run
        camera_mode: Camera mode for video recording
        output_dir: Directory to save test videos (defaults to run_dir/test_videos)
    """
    # Construct paths
    run_dir = Path(log_dir) / run_name
    checkpoint_path = run_dir / checkpoint
    config_path = run_dir / "config.yaml"
    
    # Validate paths
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print("=" * 70)
    print("CPG Policy Testing")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print("=" * 70)
    
    # Load configuration
    print("\nLoading configuration...")
    cfg = load_config(config_path)
    
    # Extract environment parameters
    model_path = os.path.join(cfg.env.model_dir, cfg.env.model_file)
    timestep = cfg.env.timestep
    frame_skip = cfg.env.frame_skip
    max_episode_steps = cfg.env.max_episode_steps
    gait_type = cfg.env.gait_type
    damping_scale = cfg.env.damping_scale
    stiffness_scale = cfg.env.stiffness_scale
    reward_weights = dict(cfg.reward) if 'reward' in cfg else None
    
    # Use camera_mode from config if not explicitly provided (default is "follow")
    # Command line argument takes precedence
    if camera_mode == "follow" and 'camera_mode' in cfg.env:
        camera_mode = cfg.env.camera_mode
    
    # Extract network parameters
    obs_dim = 34  # Quadruped observation dimension
    act_dim = 12  # Residual action dimension
    hidden_dim = cfg.train.hidden_dim
    
    print(f"  Model path: {model_path}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Gait type: {gait_type}")
    print(f"  Camera mode: {camera_mode}")
    
    # Create environment
    print("\nCreating environment...")
    env = QuadrupedEnvCPG(
        model_path=model_path,
        gait_type=gait_type,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        frame_skip=frame_skip,
        timestep=timestep,
        reward_weights=reward_weights,
        damping_scale=damping_scale,
        stiffness_scale=stiffness_scale,
        camera_mode=camera_mode,
    )
    
    # Create policy network
    print("\nCreating policy network...")
    policy = CPGModulatingPolicy(obs_dim, act_dim, hidden_dim=hidden_dim)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint_data, dict) and 'policy_state_dict' in checkpoint_data:
        policy.load_state_dict(checkpoint_data['policy_state_dict'])
        epoch = checkpoint_data.get('epoch', 'unknown')
        print(f"  Loaded policy from epoch {epoch}")
    else:
        # Fallback: assume checkpoint is just the state dict
        policy.load_state_dict(checkpoint_data)
        print("  Loaded policy state dict (no epoch info)")
    
    policy = policy.to(device)
    policy.eval()  # Set to evaluation mode
    
    # Create output directory for test videos
    if output_dir is None:
        output_dir_path = run_dir / "test_videos"
    else:
        output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Wrap environment with video recorder
    checkpoint_stem = checkpoint_path.stem  # Get filename without extension
    video_env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(output_dir_path),
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix=f"test_{checkpoint_stem}"
    )
    
    # Run episodes
    print(f"\n{'='*70}")
    print(f"Running {num_episodes} test episodes...")
    print(f"{'='*70}")
    
    episode_returns = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = video_env.reset()
        done = False
        episode_return = 0.0
        step_count = 0
        
        print(f"\nEpisode {ep+1}/{num_episodes}:", end=" ", flush=True)
        
        while not done and step_count < max_episode_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action, _ = policy.get_action(obs_tensor)
                action = action.cpu().numpy()[0]
                
                # Clip actions to valid range
                if isinstance(env.action_space, gym.spaces.Box):
                    action = np.clip(action, env.action_space.low, env.action_space.high)
            
            obs, reward, terminated, truncated, info = video_env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            step_count += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        print(f"Return = {episode_return:.2f}, Length = {step_count}")
    
    # Close environment
    video_env.close()
    env.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    print(f"Average Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min Return: {np.min(episode_returns):.2f}")
    print(f"Max Return: {np.max(episode_returns):.2f}")
    print(f"\nVideos saved to: {output_dir_path}/")
    print(f"{'='*70}")
    
    return episode_returns, episode_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a trained CPG-based quadruped policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test final checkpoint
  python test_cpg_policy.py --run_name "12_03_01_48_19_Go1_CPG_fskip10_ts0.005_envs32_hzn200" --checkpoint "policy_final.pth"
  
  # Test specific epoch checkpoint
  python test_cpg_policy.py --run_name "12_03_01_48_19_Go1_CPG_fskip10_ts0.005_envs32_hzn200" --checkpoint "policy_epoch_100.pth" --num_episodes 10
  
  # Test with custom camera mode
  python test_cpg_policy.py --run_name "12_03_01_48_19_Go1_CPG_fskip10_ts0.005_envs32_hzn200" --checkpoint "policy_final.pth" --camera_mode "side"
        """
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the training run (directory name in runs/trpo_quadruped/)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint filename (e.g., 'policy_epoch_100.pth' or 'policy_final.pth')"
    )
    
    parser.add_argument(
        "--log_dir",
        type=str,
        default="runs/trpo_quadruped",
        help="Base directory containing runs (default: runs/trpo_quadruped)"
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="follow",
        choices=["follow", "fixed", "side", "top"],
        help="Camera mode for video recording (default: follow)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save test videos (default: run_dir/test_videos)"
    )
    
    args = parser.parse_args()
    
    # Run testing
    try:
        returns, lengths = test_policy(
            run_name=args.run_name,
            checkpoint=args.checkpoint,
            log_dir=args.log_dir,
            num_episodes=args.num_episodes,
            camera_mode=args.camera_mode,
            output_dir=args.output_dir,
        )
        print("\n✅ Testing completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        raise

