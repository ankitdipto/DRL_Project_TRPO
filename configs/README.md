# Hydra Configuration System

This directory contains Hydra configuration files for managing training parameters.

## Structure

```
configs/
├── config_standard.yaml   # Base config for standard (direct joint) control
├── config_cpg.yaml         # Base config for CPG-based control
├── train/                  # Training configuration variants
│   ├── quick_test.yaml     # Fast test (50 epochs, 4 envs)
│   └── full_training.yaml  # Full training (5000 epochs, 64 envs)
├── env/                    # Environment configuration variants
│   ├── fast_sim.yaml       # Fast simulation (frame_skip=10, dt=0.01)
│   └── accurate_sim.yaml   # Accurate simulation (frame_skip=5, dt=0.005)
└── reward/                 # Reward weight configurations
    ├── speed_focused.yaml      # Emphasize forward velocity
    ├── stable_focused.yaml     # Emphasize stability
    └── efficient_focused.yaml  # Emphasize energy efficiency
```

## Usage

### Run with default config
```bash
python train_quadruped_cpg.py
python train_quadruped.py
```

### Override config groups
```bash
# Use quick test training config
python train_quadruped_cpg.py train=quick_test

# Use fast simulation config
python train_quadruped_cpg.py env=fast_sim

# Combine multiple config groups
python train_quadruped_cpg.py train=quick_test env=fast_sim
```

### Override specific parameters
```bash
# Single parameter
python train_quadruped_cpg.py train.epochs=1000

# Multiple parameters
python train_quadruped_cpg.py train.epochs=1000 train.num_envs=16

# Nested parameters
python train_quadruped.py trpo.delta=0.02 trpo.cg_iters=15

# Reward weights
python train_quadruped_cpg.py reward.forward_velocity=2.0 reward.energy_cost=0.01
```

### Use reward configurations
```bash
# Speed-focused training
python train_quadruped_cpg.py reward=speed_focused

# Stability-focused training
python train_quadruped.py reward=stable_focused

# Energy-efficient training
python train_quadruped_cpg.py reward=efficient_focused
```

### View config without running
```bash
python train_quadruped_cpg.py --cfg job
```

## Creating New Configs

To create a new training variant:

1. Create `configs/train/my_config.yaml`
2. Add parameters to override:
```yaml
train:
  epochs: 2000
  num_envs: 48
```
3. Use it: `python train_quadruped_cpg.py train=my_config`

