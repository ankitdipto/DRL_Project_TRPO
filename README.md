# Trust Region Policy Optimization (TRPO) with Gait Priors

**Extending TRPO for Quadrupedal Locomotion through Central Pattern Generator-Inspired Gait Priors**

## 📋 Overview

This project extends TRPO to tackle challenging quadrupedal locomotion by incorporating domain knowledge through gait priors. 

**Key Finding**: Vanilla TRPO fails to learn natural, symmetric gaits on the Unitree Go1 quadruped. By incorporating a CPG-inspired trotting gait prior and training a residual policy, the agent achieves natural, rhythmic, symmetric locomotion.

### Core Implementation
- Full TRPO with natural gradients (conjugate gradient solver) in pytorch
- Gaussian policies for continuous control
- GAE for advantage estimation

## 🎬 Results

<div align="center">

![TRPO Trained Policy](assets/trpo_quadruped_gait_prior.gif)

*CPG-based policy achieving smooth, symmetric trotting gait on Unitree Go1 quadruped.*

</div>

### Experimental Findings

| Approach | Result | Gait Quality |
|----------|--------|--------------|
| **Vanilla TRPO** | Failed | Asymmetric, unnatural movements |
| **CPG + Residual Policy** | ✅ Success | Natural, rhythmic, symmetric trot |

## 🚀 Installation

```bash
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, PyTorch 2.0+, MuJoCo, Gymnasium

## 📊 Experiments

### 1. Standard MuJoCo Tasks (Baseline)

Train TRPO on standard continuous control benchmarks:

```bash
python main.py --env_id Hopper-v5 --num_envs 32 --epochs 1000
```

**Supported environments**: `BipedalWalker-v3`, `Hopper-v5`, `Walker2d-v5`, `Swimmer-v5`, `InvertedPendulum-v5`

**Monitor training**:
```bash
tensorboard --logdir runs/
```

### 2. Quadruped Locomotion - Vanilla TRPO (Baseline)

Direct joint control without gait priors:

```bash
python train_quadruped.py
```

**Configuration**: `configs/config_standard.yaml`
- Action space: 12D joint positions
- Observation: 34D (joint states, base pose/velocity)
- Environments: 4 parallel
- Training: 5000 epochs

**Expected outcome**: Agent struggles to learn coordinated gait patterns, exhibits asymmetric and unnatural movements.

### 3. Quadruped Locomotion - CPG-based (Main Result)

Residual policy trained on top of trotting gait prior:

```bash
python train_quadruped_cpg.py
```

**Configuration**: `configs/config_cpg.yaml`
- Action space: 12D residual actions (added to base trot)
- Base controller: 1 Hz trotting gait (diagonal leg pairs)
- Policy learns: Gait modulation for forward locomotion
- Environments: 32 parallel
- Training: 2000 epochs

**Expected outcome**: Natural, symmetric trotting gait with smooth forward locomotion.

## ⚙️ Configuration

Modify hyperparameters via YAML configs in `configs/`:

```yaml
train:
  epochs: 2000
  steps_per_epoch: 200
  num_envs: 32
  hidden_dim: 128
  
env:
  timestep: 0.005      # 200 Hz simulation
  frame_skip: 10       # 20 Hz control
  stiffness_scale: 0.33  # Reduced stiffness for compliance
  
reward:
  forward_velocity: 2.0
  alive_bonus: 0.5
```

## 📁 Project Structure

```
DRL_Project_TRPO/
├── main.py                    # TRPO for standard Gym/MuJoCo tasks
├── train_quadruped.py         # Vanilla TRPO for Go1 quadruped
├── train_quadruped_cpg.py     # CPG-based TRPO for Go1
├── actor_critic.py            # Policy and value networks
├── quadruped_env.py           # Standard quadruped environment
├── quadruped_env_cpg.py       # CPG-based environment with gait prior
├── data_collection.py         # Rollout buffer and GAE
├── requirements.txt           # Python dependencies
├── configs/                   # Training configurations
│   ├── config_standard.yaml   # Vanilla TRPO config
│   ├── config_cpg.yaml        # CPG-based config
│   └── README.md              # Config documentation
├── mujoco_menagerie/          # Unitree Go1 robot model
├── assets/                    # Demo videos and GIFs
├── docs/                      # Additional documentation
└── scratchpad/                # Testing and development scripts
```

## 🔬 Technical Details

### CPG-Inspired Gait Prior

The base trotting controller generates coordinated leg movements:

- **Diagonal pairs**: FR+RL (phase 0), FL+RR (phase π)
- **Frequency**: 1 Hz base rhythm
- **Amplitudes**: Hip (0.0), Thigh (0.3), Calf (0.3 rad)

Policy outputs 12D residual actions scaled by 0.2 and added to base gait.

### TRPO Algorithm

1. Collect rollouts using current policy
2. Compute advantages via GAE (γ=0.99, λ=0.97)
3. Compute policy gradient of surrogate objective
4. Solve for natural gradient using conjugate gradient
5. Backtracking line search with KL constraint (δ=0.01)
6. Update value function (10 epochs of regression)

## 📈 Monitoring

TensorBoard logs key metrics:

- **Rollout/Epoch_Reward**: Episode returns
- **Policy/KL_Divergence**: Trust region constraint
- **State/forward_velocity**: Locomotion speed
- **Rewards/**: Individual reward components

Videos recorded every 100 epochs to `runs/trpo_quadruped/`.

## 📚 References

1. [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)
2. [Generalized Advantage Estimation (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
3. [OpenAI Spinning Up - TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)

---

**Author**: Ankit Sinha  
