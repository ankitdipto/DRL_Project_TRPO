#!/usr/bin/env python3
"""
Quick integration test for TRPO + Quadruped environment.

This script runs a few training epochs to verify everything works correctly
before starting a full training run.
"""

import torch
import torch.optim as optim
import numpy as np
import os

from actor_critic import GaussianPolicy, ValueNetwork
from quadruped_env import VectorizedQuadrupedEnv, QuadrupedEnv
from data_collection import RolloutBuffer

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def test_environment():
    """Test that the environment works correctly."""
    print("\n" + "="*70)
    print("TEST 1: Environment Functionality")
    print("="*70)
    
    menagerie_path = "/home/hice1/asinha389/scratch/mujoco_menagerie"
    model_path = os.path.join(menagerie_path, "unitree_go1/scene.xml")
    
    # Test single environment
    print("\n1. Testing single environment...")
    env = QuadrupedEnv(model_path=model_path, max_episode_steps=100)
    obs, info = env.reset(seed=42)
    
    print(f"   ✓ Reset successful")
    print(f"   ✓ Observation shape: {obs.shape}")
    print(f"   ✓ Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {i+1}: reward={reward:.3f}, height={info['base_height']:.3f}m")
        
        if terminated or truncated:
            print(f"   Episode ended: {info.get('termination_reason', 'unknown')}")
            break
    
    env.close()
    print("   ✓ Single environment test passed")
    
    # Test vectorized environment
    print("\n2. Testing vectorized environment...")
    vec_env = VectorizedQuadrupedEnv(
        model_path=model_path,
        num_envs=4,
        max_episode_steps=100,
        num_threads=2
    )
    
    obs, infos = vec_env.reset(seed=42)
    print(f"   ✓ Reset successful")
    print(f"   ✓ Observations shape: {obs.shape}")
    
    # Run a few steps
    for i in range(5):
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        print(f"   Step {i+1}: avg_reward={rewards.mean():.3f}, "
              f"terminated={terminated.sum()}, truncated={truncated.sum()}")
    
    vec_env.close()
    print("   ✓ Vectorized environment test passed")
    
    return True


def test_networks():
    """Test that policy and value networks work correctly."""
    print("\n" + "="*70)
    print("TEST 2: Network Functionality")
    print("="*70)
    
    obs_dim = 34
    act_dim = 12
    hidden_dim = 256
    batch_size = 64
    
    print(f"\n1. Creating networks...")
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim: {act_dim}")
    print(f"   Hidden dim: {hidden_dim}")
    
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    value_net = ValueNetwork(obs_dim, hidden_dim=hidden_dim).to(device)
    
    print(f"   ✓ Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print(f"   ✓ Value network parameters: {sum(p.numel() for p in value_net.parameters())}")
    
    # Test forward pass
    print(f"\n2. Testing forward pass...")
    obs = torch.randn(batch_size, obs_dim).to(device)
    
    actions, logp = policy.get_action(obs)
    values = value_net(obs)
    
    print(f"   ✓ Actions shape: {actions.shape}")
    print(f"   ✓ Log probs shape: {logp.shape}")
    print(f"   ✓ Values shape: {values.shape}")
    print(f"   ✓ Action range: [{actions.min():.2f}, {actions.max():.2f}]")
    
    # Test backward pass
    print(f"\n3. Testing backward pass...")
    optimizer = optim.Adam(list(policy.parameters()) + list(value_net.parameters()), lr=3e-4)
    
    loss = -logp.mean() + F.mse_loss(values, torch.zeros_like(values))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Backward pass successful")
    print(f"   ✓ Loss: {loss.item():.4f}")
    
    return True


def test_data_collection():
    """Test that data collection and GAE computation work correctly."""
    print("\n" + "="*70)
    print("TEST 3: Data Collection & GAE")
    print("="*70)
    
    n_steps = 10
    n_envs = 4
    n_obs = 34
    n_acts = 12
    
    print(f"\n1. Creating rollout buffer...")
    print(f"   Steps: {n_steps}")
    print(f"   Environments: {n_envs}")
    
    buffer = RolloutBuffer(n_steps, n_envs, n_obs, n_acts)
    print(f"   ✓ Buffer created")
    
    # Fill buffer with dummy data
    print(f"\n2. Filling buffer with data...")
    for t in range(n_steps):
        obs = torch.randn(n_envs, n_obs).to(device)
        acts = torch.randn(n_envs, n_acts).to(device)
        logps = torch.randn(n_envs).to(device)
        rews = np.random.randn(n_envs)
        vals = torch.randn(n_envs).to(device)
        dones = np.random.rand(n_envs) < 0.1
        
        buffer.add(t, obs, acts, logps, rews, vals, dones)
    
    print(f"   ✓ Buffer filled")
    
    # Compute GAE
    print(f"\n3. Computing GAE...")
    last_vals = torch.randn(n_envs).to(device)
    adv, returns = buffer.compute_gae(last_vals, gamma=0.99, lam=0.97)
    
    print(f"   ✓ Advantages shape: {adv.shape}")
    print(f"   ✓ Returns shape: {returns.shape}")
    print(f"   ✓ Advantage range: [{adv.min():.2f}, {adv.max():.2f}]")
    print(f"   ✓ Returns range: [{returns.min():.2f}, {returns.max():.2f}]")
    
    return True


def test_training_loop():
    """Test a few training iterations."""
    print("\n" + "="*70)
    print("TEST 4: Training Loop Integration")
    print("="*70)
    
    # Setup
    menagerie_path = "/home/hice1/asinha389/scratch/mujoco_menagerie"
    model_path = os.path.join(menagerie_path, "unitree_go1/scene.xml")
    
    num_envs = 4
    obs_dim = 34
    act_dim = 12
    hidden_dim = 64  # Smaller for quick test
    steps_per_epoch = 20
    
    print(f"\n1. Creating environment and networks...")
    vec_env = VectorizedQuadrupedEnv(
        model_path=model_path,
        num_envs=num_envs,
        max_episode_steps=100,
        num_threads=2
    )
    
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=hidden_dim).to(device)
    value_net = ValueNetwork(obs_dim, hidden_dim=hidden_dim).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)
    
    print(f"   ✓ Environment and networks created")
    
    # Run a few epochs
    print(f"\n2. Running {3} training epochs...")
    obs, _ = vec_env.reset(seed=42)
    
    for epoch in range(3):
        buffer = RolloutBuffer(steps_per_epoch, num_envs, obs_dim, act_dim)
        
        # Collect rollout
        for t in range(steps_per_epoch):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                actions, logp_t = policy.get_action(obs_tensor)
                acts = actions.cpu().numpy()
                vals = value_net(obs_tensor)
            
            next_obs, rews, terminateds, truncateds, infos = vec_env.step(acts)
            dones = np.logical_or(terminateds, truncateds)
            
            buffer.add(t, obs_tensor, actions, logp_t, rews, vals, dones)
            obs = next_obs
        
        # Compute advantages
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            last_vals = value_net(obs_tensor)
        
        adv, returns = buffer.compute_gae(last_vals, gamma=0.99, lam=0.97)
        
        # Flatten for update
        obs_flat = buffer.obs.reshape(-1, obs_dim)
        acts_flat = buffer.acts.reshape(-1, act_dim)
        returns_flat = returns.reshape(-1)
        
        # Simple value function update (skip full TRPO for speed)
        v_pred = value_net(obs_flat)
        value_loss = F.mse_loss(v_pred, returns_flat)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        avg_reward = buffer.rews.mean().item()
        print(f"   Epoch {epoch+1}: avg_reward={avg_reward:.3f}, value_loss={value_loss.item():.4f}")
    
    vec_env.close()
    print(f"   ✓ Training loop test passed")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("TRPO + QUADRUPED INTEGRATION TESTS")
    print("="*70)
    
    import torch.nn.functional as F
    
    tests = [
        ("Environment", test_environment),
        ("Networks", test_networks),
        ("Data Collection", test_data_collection),
        ("Training Loop", test_training_loop),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with error:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(success for _, success in results)
    
    print("="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for training!")
        print("\nTo start training, run:")
        print("  python train_quadruped.py")
    else:
        print("✗ SOME TESTS FAILED - Please fix errors before training")
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    import torch.nn.functional as F
    success = main()
    exit(0 if success else 1)

