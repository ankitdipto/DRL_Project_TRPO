# Trust Region Policy Optimization (TRPO) Implementation

A clean, educational implementation of Trust Region Policy Optimization (TRPO) for continuous control tasks using PyTorch and Gymnasium.

## 📋 Overview

This project implements TRPO, a model-free reinforcement learning algorithm that ensures monotonic policy improvement through trust region constraints. The implementation uses:

- **Gaussian policies** for continuous action spaces
- **Generalized Advantage Estimation (GAE)** for variance reduction
- **Conjugate Gradient** for efficient natural gradient computation
- **Backtracking line search** for step size optimization
- **Vectorized environments** for parallel data collection

## 🎯 Features

- ✅ Full TRPO implementation with KL divergence constraints
- ✅ Vectorized environment support for efficient training
- ✅ TensorBoard logging for training visualization
- ✅ Automatic video recording of evaluation episodes
- ✅ Clean, modular code structure
- ✅ GPU acceleration support

## 📁 Project Structure

```
DRL_Project_TRPO/
├── main.py                 # Main training script with TRPO algorithm
├── actor_critic.py         # Policy and value network definitions
├── data_collection.py      # Rollout buffer and GAE computation
├── evaluate.py             # Script to evaluate trained policies
├── config.py               # Configuration parameters
├── requirements.txt        # Python dependencies
├── runs/                   # TensorBoard logs, videos, and checkpoints
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# Install all dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision
pip install gymnasium[classic-control]
pip install tensorboard
pip install numpy tqdm opencv-python
```

### Training

Run the default training configuration (Pendulum-v1):

```bash
python main.py
```

This will:
- Train TRPO for 1000 epochs
- Use 16 parallel environments
- Collect 200 steps per epoch per environment (3,200 transitions per update)
- Evaluate and record videos every 40 epochs
- Save logs, videos, and checkpoints to `runs/trpo_pendulum/pendulum_<timestamp>/`

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir=runs/trpo_pendulum
```

Then open your browser to `http://localhost:6006`

## 📊 Tracked Metrics

The implementation logs the following metrics to TensorBoard:

### Training Metrics
- **Loss/Surrogate**: Policy surrogate loss
- **Loss/Value**: Value function MSE loss
- **Policy/KL_Divergence**: KL divergence between old and new policy
- **Policy/Entropy**: Policy entropy (exploration measure)
- **Policy/Actual_Improvement**: Actual improvement in surrogate objective
- **Rollout/Epoch_Reward**: Average reward per epoch

### Evaluation Metrics
- **Eval/Average_Return**: Average return over evaluation episodes

## 🎮 Supported Environments

This implementation will _most probably_ work with any Gymnasium environment with continuous action spaces:

- **Pendulum-v1** (default)
- **LunarLanderContinuous-v2**
- **BipedalWalker-v3**
- **HalfCheetah-v4** (MuJoCo)
- **Hopper-v4** (MuJoCo)
- And more...

### Changing Environments

To train on a different environment, modify the `__main__` section in `main.py`:

```python
# Example: LunarLanderContinuous-v2
envs = gym.vector.SyncVectorEnv([
    lambda: gym.make("LunarLanderContinuous-v2") for _ in range(num_envs)
])
eval_env = gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")

obs_dim = 8  # Update based on environment
act_dim = 2  # Update based on environment
```

## ⚙️ Hyperparameters

Key hyperparameters and their default values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 1000 | Number of training epochs |
| `steps_per_epoch` | 200 | Steps collected per environment per epoch |
| `n_envs` | 16 | Number of parallel environments |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.97 | GAE lambda parameter |
| `delta` | 0.01 | KL divergence constraint |
| `cg_iters` | 10 | Conjugate gradient iterations |
| `val_func_epochs` | 10 | Value function training epochs per update |
| `eval_freq` | 40 | Evaluation frequency (in epochs) |

## 🎥 Video Recording

Videos are automatically recorded during training:

- **Periodic evaluation**: Every `eval_freq` epochs, 3 episodes are recorded
- **Final evaluation**: 5 episodes recorded at the end of training
- **Location**: `runs/trpo_pendulum/pendulum_<timestamp>/`
- **Format**: MP4 files

Videos are named:
- `epoch_<N>-episode-<M>.mp4` for periodic evaluations
- `final-episode-<M>.mp4` for final evaluation

## 🧮 Algorithm Details

### TRPO Update Steps

1. **Collect Rollouts**: Gather trajectories using current policy
2. **Compute Advantages**: Use GAE for advantage estimation
3. **Policy Gradient**: Compute surrogate loss gradient
4. **Natural Gradient**: Solve Fisher-vector product using conjugate gradient
5. **Step Size**: Compute optimal step size from KL constraint
6. **Line Search**: Backtracking line search for valid update
7. **Value Update**: Fit value function with MSE loss

### Key Equations

**Surrogate Objective:**
```
L(θ) = E[π_θ(a|s) / π_θ_old(a|s) * A(s,a)]
```

**KL Constraint:**
```
E[KL(π_θ_old || π_θ)] ≤ δ
```

**GAE:**
```
A_t = Σ(γλ)^l δ_{t+l}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

## 📈 Expected Results

### Pendulum-v1

- **Random Policy**: ~-1600 to -1200
- **Trained Policy**: -200 to -150 (after ~500-800 epochs)

The pendulum should learn to swing up and balance within the first few hundred epochs.

## 🔧 Code Structure

### `main.py`
- `conjugate_gradient()`: Solves Ax=b using CG method
- `fisher_vector_product()`: Computes Fisher-vector product
- `trpo_update()`: Main TRPO update logic
- `evaluate_policy()`: Policy evaluation with video recording
- `train_trpo()`: Main training loop

### `actor_critic.py`
- `GaussianPolicy`: Continuous policy with diagonal Gaussian distribution
- `ValueNetwork`: State-value function approximator

### `data_collection.py`
- `RolloutBuffer`: Stores trajectory data
- `compute_gae()`: Computes GAE advantages and returns

## 📚 References

1. **TRPO Paper**: [Trust Region Policy Optimization (Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)
2. **GAE Paper**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)
3. **Spinning Up in Deep RL**: [OpenAI's TRPO Documentation](https://spinningup.openai.com/en/latest/algorithms/trpo.html)

---

**Author**: Ankit Sinha  
