# TRPO Quadruped Training - Quick Start Guide

## 🚀 Quick Commands

```bash
# Activate environment
conda activate DRL_HW

# Run tests (recommended first time)
python test_training_integration.py

# Start training (default: 5000 epochs, 16 envs)
python train_quadruped.py

# Monitor training
tensorboard --logdir=runs/trpo_quadruped
# Then open: http://localhost:6006
```

---

## 📊 What to Expect

### Training Progress

| Epochs | Episode Return | Episode Length | Forward Velocity | Behavior |
|--------|---------------|----------------|------------------|----------|
| 0 | ~450 | 1-5 steps | 0 m/s | Falls immediately |
| 500 | ~700 | 50-200 steps | 0.2 m/s | Stands briefly |
| 2000 | ~1200 | 500-1000 steps | 0.7 m/s | Walking gait |
| 5000 | ~1800 | 1000 steps | 1.5 m/s | Robust locomotion |

### Training Time
- **16 parallel environments** (default)
- **~8-12 hours** on CPU
- **~4-6 hours** with GPU acceleration
- **16M total environment steps**

---

## 🎛️ Key Configuration

### In `train_quadruped.py`

```python
# Environment
num_envs = 16              # More = faster training
hidden_dim = 256           # Network capacity
epochs = 5000              # Training duration

# TRPO
gamma = 0.99               # Discount factor
lam = 0.97                 # GAE lambda
delta = 0.01               # KL constraint

# Evaluation
eval_freq = 100            # Evaluate every N epochs
save_freq = 100            # Save every N epochs
```

### Reward Weights (in `quadruped_env.py`)

```python
reward_weights = {
    'forward_velocity': 1.0,      # ↑ for more speed
    'alive_bonus': 1.0,           # ↓ to discourage standing
    'orientation_penalty': 0.5,   # ↑ for more stability
    'energy_cost': 0.001,         # ↑ for efficiency
    'joint_limit_penalty': 0.1,
    'height_penalty': 0.5,
}
```

---

## 📈 Monitoring Training

### TensorBoard Metrics

**Key Metrics to Watch**:
- `Rollout/Epoch_Reward` - Should increase over time
- `Rollout/Episode_Return` - Target: 1500-2000+
- `Rollout/Episode_Length` - Target: 1000 (full episode)
- `Policy/KL_Divergence` - Should stay near 0.01
- `Eval/Average_Return` - Evaluation performance

**Signs of Good Training**:
- ✅ Epoch reward steadily increasing
- ✅ Episode length increasing
- ✅ KL divergence stable around 0.01
- ✅ Value loss decreasing

**Signs of Problems**:
- ❌ Reward plateaus early (< 800)
- ❌ KL divergence = 0 (no updates)
- ❌ Episode length stays < 10 steps
- ❌ NaN values in any metric

---

## 🎥 Videos

Videos are automatically saved during training:

```
runs/trpo_quadruped/go1_MM_DD_HH_MM_SS/
├── epoch_100-episode-0.mp4
├── epoch_100-episode-1.mp4
├── epoch_100-episode-2.mp4
├── epoch_200-episode-0.mp4
...
└── final-episode-0.mp4
```

Check videos at evaluation epochs to see learning progress!

---

## 🔧 Common Adjustments

### Robot Falls Immediately
```python
# Increase stability rewards
reward_weights['orientation_penalty'] = 1.0
reward_weights['height_penalty'] = 1.0
```

### Robot Stands Still (Doesn't Walk)
```python
# Emphasize forward motion
reward_weights['forward_velocity'] = 2.0
reward_weights['alive_bonus'] = 0.5
```

### Training Too Slow
```python
# Increase parallelism
num_envs = 32
num_threads = 8
```

### Policy Diverges
```python
# Tighten KL constraint
delta = 0.005
damping = 0.01
```

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `quadruped_env.py` | Environment implementation |
| `train_quadruped.py` | Training script |
| `test_training_integration.py` | Integration tests |
| `actor_critic.py` | Policy/value networks |
| `data_collection.py` | Rollout buffer |
| `main.py` | Original TRPO (Pendulum) |

---

## 🐛 Troubleshooting

### Import Error: No module named 'mujoco'
```bash
conda activate DRL_HW
pip install mujoco
```

### CUDA Out of Memory
```python
# Use CPU or reduce batch size
device = torch.device("cpu")
num_envs = 8  # Reduce from 16
```

### Videos Not Saving
```bash
# Install video dependencies
pip install opencv-python moviepy imageio
```

### Training Hangs
```bash
# Reduce threads if CPU overloaded
num_threads = 2  # Instead of 4
```

---

## 📞 Quick Help

### Check Environment Works
```bash
python quadruped_env.py
```

### Run All Tests
```bash
python test_training_integration.py
```

### Test Single Episode
```python
from quadruped_env import QuadrupedEnv

env = QuadrupedEnv(model_path="...")
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    if done or truncated:
        break
```

---

## 🎯 Success Criteria

Training is successful when:
- ✅ Episode return > 1500
- ✅ Episode length = 1000 (full episode)
- ✅ Forward velocity > 1.0 m/s
- ✅ Robot maintains upright posture
- ✅ Smooth, stable gait in videos

---

## 📚 Documentation

- **Full Guide**: `QUADRUPED_ENV_GUIDE.md`
- **Environment Details**: `quadruped_env.py` (docstrings)
- **TRPO Algorithm**: `README.md`
- **Go1 Specs**: `GO1_SIMULATION_SUMMARY.md`

---

**Ready to train?** Run: `python train_quadruped.py`

Good luck! 🚀🤖

