# Quick Reference - Quadruped Locomotion Project

## 🚀 Quick Start Commands

```bash
# Activate environment
conda activate DRL_HW

# Verify models
cd /home/hice1/asinha389/scratch/DRL_Project_TRPO
python verify_models.py

# View documentation
cat QUADRUPED_MODELS_SUMMARY.md
cat TASK_1_1_COMPLETION_REPORT.md
```

## 📁 Important Paths

```bash
# Project directory
/home/hice1/asinha389/scratch/DRL_Project_TRPO

# MuJoCo Menagerie
/home/hice1/asinha389/scratch/mujoco_menagerie

# Recommended model (Unitree Go1)
/home/hice1/asinha389/scratch/mujoco_menagerie/unitree_go1/go1.xml
```

## 🤖 Available Quadruped Models

| Robot | Path | MJX | Complexity |
|-------|------|-----|------------|
| **Unitree Go1** ⭐ | `unitree_go1/go1.xml` | ❌ | Medium |
| Unitree A1 | `unitree_a1/a1.xml` | ❌ | Medium |
| Unitree Go2 | `unitree_go2/go2.xml` | ✅ | Medium-High |
| ANYmal B | `anybotics_anymal_b/anymal_b.xml` | ❌ | High |
| **ANYmal C** ⭐ | `anybotics_anymal_c/anymal_c.xml` | ✅ | High |
| Boston Dynamics Spot | `boston_dynamics_spot/spot.xml` | ❌ | Very High |
| Google Barkour v0 | `google_barkour_v0/barkour_v0.xml` | ✅ | High |
| Google Barkour vB | `google_barkour_vb/barkour_vb.xml` | ✅ | High |

⭐ = Recommended

## 🎯 Model Specifications (All Quadrupeds)

- **DOF**: 18 (6 floating base + 12 actuated joints)
- **Actuators**: 12 (3 per leg: abduction, hip, knee)
- **Bodies**: 14-16
- **Control**: Position-controlled (can be modified to torque)
- **Timestep**: 0.001-0.002s

## 📝 Typical Observation Space (~48 dims)

```python
obs = [
    joint_positions,      # 12 dims
    joint_velocities,     # 12 dims
    body_orientation,     # 3-4 dims (euler/quaternion)
    body_linear_velocity, # 3 dims
    body_angular_velocity,# 3 dims
    # Optional:
    previous_actions,     # 12 dims
    foot_contacts,        # 4 dims
]
```

## 🎮 Action Space (12 dims)

```python
# Position control (recommended to start)
actions = target_joint_positions  # 12 dims, within joint limits

# OR Torque control (advanced)
actions = joint_torques  # 12 dims, within torque limits
```

## 🏆 Typical Reward Function

```python
reward = (
    w1 * forward_velocity_reward      # Encourage forward motion
    - w2 * energy_cost                # Penalize high torques
    - w3 * orientation_penalty        # Keep body upright
    - w4 * joint_limit_penalty        # Stay within limits
    + w5 * alive_bonus                # Survive longer
)
```

## 📊 Expected Training Parameters

```python
# Environment
obs_dim = 48
act_dim = 12

# Network
hidden_dim = 256  # Larger than Pendulum

# Training
epochs = 10000
steps_per_epoch = 2048
n_envs = 8
gamma = 0.99
lam = 0.95

# TRPO
delta = 0.01
cg_iters = 10
damping = 0.1

# Evaluation
eval_freq = 100
max_episode_steps = 1000
```

## 🔍 Loading a Model (Python)

```python
import mujoco

# Load model
model_path = "/home/hice1/asinha389/scratch/mujoco_menagerie/unitree_go1/go1.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Basic info
print(f"DOF: {model.nv}")
print(f"Actuators: {model.nu}")
print(f"Timestep: {model.opt.timestep}")

# Step simulation
mujoco.mj_step(model, data)
```

## 📚 Documentation Files

1. **QUADRUPED_MODELS_SUMMARY.md** - Detailed analysis of all models
2. **TASK_1_1_COMPLETION_REPORT.md** - Sub-task 1.1 completion report
3. **QUICK_REFERENCE.md** - This file
4. **verify_models.py** - Model verification script

## 🔗 Useful Links

- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- MuJoCo Docs: https://mujoco.readthedocs.io/
- MJX Docs: https://mujoco.readthedocs.io/en/stable/mjx.html
- ANYmal C Training Example: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb

## ✅ Completed Tasks

- [x] Sub-task 1.1: Install MuJoCo Menagerie & identify models
- [ ] Sub-task 1.2: Create custom Gymnasium environment
- [ ] Sub-task 1.3: Implement reward function
- [ ] Sub-task 2.1: Adapt TRPO training script
- [ ] Sub-task 2.2: Run initial training experiments

## 🚀 Next: Sub-task 1.2

Create custom Gymnasium environment for quadruped locomotion:
1. Create `envs/quadruped_env.py`
2. Implement `QuadrupedEnv` class
3. Define observation/action spaces
4. Implement step(), reset(), render()
5. Test with random actions

---

**Last Updated**: November 21, 2024

