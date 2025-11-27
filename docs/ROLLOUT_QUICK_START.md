# Multi-Robot Rollout - Quick Start Guide

## 🚀 Quick Run

```bash
# Activate environment
conda activate DRL_HW

# Run the script
cd /home/hice1/asinha389/scratch/DRL_Project_TRPO
python multi_go1_rollout.py
```

## 📝 Configuration

Edit the `main()` function in `multi_go1_rollout.py`:

```python
# Configuration
nbatch = 4          # Number of robot instances
duration = 8.0      # Simulation duration (seconds)
nthread = 4         # Number of parallel threads
fps = 30            # Video frame rate
```

## 🎮 Control Modes

The script automatically assigns different control modes to robots:

| Mode | Description | Behavior |
|------|-------------|----------|
| `standing` | Static standing pose | Maintains upright position |
| `trot` | Diagonal pair gait | FR-RL, FL-RR move together |
| `walk` | Sequential gait | FR → FL → RR → RL |
| `pace` | Lateral pair gait | FR-FL, RR-RL move together |
| `zero` | No control | Robot falls |
| `random` | Random actions | Chaotic motion |

### Custom Control Modes

To specify custom control modes:

```python
# In main() function, modify:
control_modes = ["trot", "trot", "walk", "walk"]  # 4 robots
state_traj, sensor_traj, ctrl_sequences, control_modes = simulate_multi_robot(
    model, nbatch, duration, nthread, control_modes=control_modes
)
```

## 🎨 Rendering Options

### Single Robot View
Shows one robot in detail:
```python
render_multi_robot(model, state_traj, control_modes, output_path, fps)
```

### Composite Grid View (Recommended)
Shows all robots simultaneously in a grid:
```python
render_all_robots_composite(model, state_traj, control_modes, output_path, fps)
```

## 📊 Output Files

### Videos
- `outputs/multi_go1_Nrobots_single.mp4` - Single robot view
- `outputs/multi_go1_Nrobots_composite.mp4` - All robots grid view ⭐

### Logs
- Console output with performance metrics
- Trajectory analysis for each robot

## 🔧 Advanced Usage

### Modify Initial Positions

Edit `setup_initial_states()`:

```python
def setup_initial_states(model, data, nbatch, spacing=2.5):
    # ...
    
    # Custom positions
    positions = [
        [0, 0, 0.35],      # Robot 0 at origin
        [2, 0, 0.35],      # Robot 1 to the right
        [0, 2, 0.35],      # Robot 2 forward
        [2, 2, 0.35],      # Robot 3 diagonal
    ]
    
    for i in range(nbatch):
        initial_states[i, 0:3] = positions[i]
```

### Custom Gait Parameters

Edit `get_walking_gait()`:

```python
def get_walking_gait(t, frequency=1.0, gait_type="trot", phase_offset=0.0):
    # Modify amplitudes
    hip_amplitude = 0.5      # Increase for larger steps
    knee_amplitude = 0.8     # Increase for higher lift
    
    # Modify frequency
    frequency = 1.5          # Increase for faster gait
```

### Scale to More Robots

```python
# In main()
nbatch = 16         # Simulate 16 robots
nthread = 8         # Use 8 threads

# Expected performance: ~150-200x realtime
```

## 📈 Performance Tips

### Maximize Speed
1. **Increase threads**: Match your CPU core count
   ```python
   nthread = 16  # For 16-core CPU
   ```

2. **Reduce rendering**: Skip rendering for pure simulation
   ```python
   # Comment out rendering calls
   # render_multi_robot(...)
   # render_all_robots_composite(...)
   ```

3. **Increase batch size**: More robots = better parallelization
   ```python
   nbatch = 32  # More robots per rollout
   ```

### Maximize Quality
1. **Higher resolution**: Increase viewport size (within framebuffer limits)
   ```python
   viewport_width = 400
   viewport_height = 300
   ```

2. **Higher FPS**: Smoother videos
   ```python
   fps = 60
   ```

3. **Longer duration**: More detailed analysis
   ```python
   duration = 16.0  # 16 seconds
   ```

## 🐛 Troubleshooting

### Error: "Image width > framebuffer width"
**Solution**: Reduce rendering resolution
```python
renderer = mujoco.Renderer(model, height=480, width=640)
```

### Error: "Not enough threads"
**Solution**: Reduce nthread or increase system resources
```python
nthread = 2  # Use fewer threads
```

### Slow Performance
**Check**:
1. CPU core count: `nproc` or `lscpu`
2. Match nthread to available cores
3. Reduce nbatch if memory limited

### Videos not generating
**Check**:
1. `outputs/` directory exists
2. `imageio` is installed: `pip install imageio`
3. Sufficient disk space

## 💡 Integration with TRPO

### Step 1: Extract Observations

```python
def extract_observations(state_traj):
    """Extract observations from state trajectory."""
    nbatch, nstep, _ = state_traj.shape
    
    observations = []
    for i in range(nbatch):
        for t in range(nstep):
            # Extract relevant state components
            qpos = state_traj[i, t, 0:19]  # Position
            qvel = state_traj[i, t, 19:37]  # Velocity
            
            # Build observation (customize as needed)
            obs = np.concatenate([
                qpos[7:19],      # Joint positions (12)
                qvel[6:18],      # Joint velocities (12)
                qpos[3:7],       # Base orientation (4)
                qvel[0:3],       # Base linear velocity (3)
                qvel[3:6],       # Base angular velocity (3)
            ])
            observations.append(obs)
    
    return np.array(observations)
```

### Step 2: Compute Rewards

```python
def compute_rewards(state_traj):
    """Compute rewards from state trajectory."""
    nbatch, nstep, _ = state_traj.shape
    
    rewards = np.zeros((nbatch, nstep))
    
    for i in range(nbatch):
        positions = state_traj[i, :, 0:3]
        velocities = state_traj[i, :, 19:22]
        
        # Forward velocity reward
        forward_vel = velocities[:, 0]
        
        # Alive bonus (height > threshold)
        alive = (positions[:, 2] > 0.15).astype(float)
        
        # Energy cost (from controls)
        # energy = np.sum(controls**2, axis=1)
        
        rewards[i] = forward_vel + 0.5 * alive  # - 0.001 * energy
    
    return rewards
```

### Step 3: Create Rollout-based Data Collector

```python
def collect_rollout_trajectories(model, policy, num_envs, horizon):
    """Collect trajectories using rollout."""
    
    # Setup initial states
    data = mujoco.MjData(model)
    initial_states = setup_initial_states(model, data, num_envs)
    
    # Generate controls using policy
    ctrl_sequences = np.zeros((num_envs, horizon, model.nu))
    
    # For each environment, generate actions from policy
    for env_idx in range(num_envs):
        obs = extract_obs_from_state(initial_states[env_idx])
        
        for step in range(horizon):
            action = policy(obs)
            ctrl_sequences[env_idx, step] = action
            # Update obs based on predicted next state (or use rollout iteratively)
    
    # Run rollout
    datas = [copy.copy(data) for _ in range(4)]  # nthread
    state_traj, sensor_traj = rollout.rollout(
        model, datas, initial_states, ctrl_sequences, nstep=horizon
    )
    
    # Extract observations and rewards
    observations = extract_observations(state_traj)
    rewards = compute_rewards(state_traj)
    
    return observations, rewards, state_traj
```

## 📚 References

### MuJoCo Rollout Documentation
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [Rollout Module](https://mujoco.readthedocs.io/en/stable/python.html#rollout)

### Related Scripts
- `test_unitree_go1_scene.py` - Single robot simulation
- `stream_unitree_go1.py` - Web-based streaming interface
- `verify_models.py` - Model verification

### Documentation
- `MULTI_ROBOT_ROLLOUT_SUMMARY.md` - Detailed overview
- `QUADRUPED_MODELS_SUMMARY.md` - Available robot models
- `GO1_SIMULATION_SUMMARY.md` - Single robot simulation guide

## 🎯 Example Workflows

### Workflow 1: Test Different Gaits
```python
# Compare trot vs walk vs pace
control_modes = ["trot", "walk", "pace", "bound"]
nbatch = 4
# Run and compare videos
```

### Workflow 2: Robustness Testing
```python
# Same gait, different initial conditions
control_modes = ["trot"] * 8
nbatch = 8

# Vary initial heights in setup_initial_states()
heights = np.linspace(0.3, 0.4, nbatch)
for i in range(nbatch):
    initial_states[i, 2] = heights[i]
```

### Workflow 3: Parameter Sweep
```python
# Test different gait frequencies
frequencies = [0.5, 1.0, 1.5, 2.0]
for freq in frequencies:
    # Modify get_walking_gait() to use freq
    # Run simulation
    # Analyze performance
```

---

**Ready to simulate! 🚀**

For questions or issues, refer to:
- `MULTI_ROBOT_ROLLOUT_SUMMARY.md` for detailed documentation
- `multi_go1_rollout.py` source code with inline comments

