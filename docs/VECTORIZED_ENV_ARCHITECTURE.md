# Vectorized Environment Architecture - Detailed Explanation

## Your Question

> "Do we not need to specify the env index when resetting self.data and self.model? My question is we are not iterating over the model or dats, then how do we ensure the correct env is resetting?"

This is an **excellent question** that gets to the heart of how the vectorized environment works!

---

## TL;DR Answer

**No, we don't need separate `MjData` instances per environment** because we use **state-based simulation**. The key insight is:

- `self.data` is a **temporary workspace**, not environment-specific storage
- Each environment is represented by a **state vector** in `next_states[i]`
- We use `self.data` to *generate* state vectors, then store them independently
- MuJoCo's `rollout` module simulates all environments from their state vectors in parallel

---

## Architecture Overview

### Traditional Approach (NOT what we do)

```python
# ❌ This is what you might expect (but we DON'T do this)
class TraditionalVectorEnv:
    def __init__(self, num_envs):
        self.models = [create_model() for _ in range(num_envs)]
        self.datas = [create_data(model) for model in self.models]
    
    def step(self, actions):
        for i in range(self.num_envs):
            self.datas[i].ctrl[:] = actions[i]
            mujoco.mj_step(self.models[i], self.datas[i])
```

**Problems**: 
- Memory intensive (N full MjData instances)
- Slow (sequential stepping)
- Doesn't leverage MuJoCo's parallel capabilities

### Our Approach (State-Based with Rollout)

```python
# ✅ What we actually do
class VectorizedQuadrupedEnv:
    def __init__(self, num_envs):
        self.model = create_model()  # ONE model (shared)
        self.data = create_data()    # ONE data (temporary workspace)
        self.datas = [create_data() for _ in range(num_threads)]  # For rollout
        
        # The actual environment states (THIS is what matters!)
        self.current_states = np.zeros((num_envs, state_size))
    
    def step(self, actions):
        # Simulate ALL envs in parallel from their states
        state_traj, _ = rollout.rollout(
            self.model,
            self.datas,  # Thread-local workspaces
            self.current_states,  # ← Each env's state
            actions,
            nstep=1
        )
        
        # Extract next states
        next_states = state_traj[:, 0, :]  # Shape: (num_envs, state_size)
```

**Benefits**:
- Memory efficient (state vectors are small)
- Fast (parallel simulation via rollout)
- Leverages MuJoCo's optimized C++ code

---

## Deep Dive: How State-Based Simulation Works

### What is a "State"?

A MuJoCo state is a compact representation of the full physics state:

```python
state = [
    qpos,      # Generalized positions (19 dims for Go1)
    qvel,      # Generalized velocities (18 dims for Go1)
    act,       # Actuator activations
    # ... other physics variables
]
```

For Go1: `state_size ≈ 38 dimensions`

### Key Functions

#### 1. `_get_state()` - Extract state from MjData

```python
def _get_state(self, data: mujoco.MjData) -> np.ndarray:
    """Extract state vector from MjData."""
    full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
    state = np.zeros(mujoco.mj_stateSize(self.model, full_physics))
    mujoco.mj_getState(self.model, data, state, full_physics)
    return state  # Returns a COPY of the state
```

**Important**: This returns a **copy** of the state, not a reference to `data`.

#### 2. `_set_state()` - Load state into MjData

```python
def _set_state(self, data: mujoco.MjData, state: np.ndarray):
    """Load state vector into MjData."""
    full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mujoco.mj_setState(self.model, data, state, full_physics)
    mujoco.mj_forward(self.model, data)
```

**Important**: This overwrites `data` with the state vector.

---

## The Reset Loop Explained

Let's trace through what happens when environment `i=2` terminates:

```python
for i in range(self.num_envs):  # i = 0, 1, 2, 3, ...
    if terminated[i] or truncated[i]:
        # Iteration i=2 (env 2 terminated)
        
        # Step 1: Reset self.data to default state
        mujoco.mj_resetData(self.model, self.data)
        # self.data now contains: qpos=[0,0,0.3,...], qvel=[0,0,0,...]
        
        # Step 2: Configure standing pose
        self.data.qpos[7:19] = [0.0, 0.9, -1.8, ...]  # + random noise
        self.data.qvel[6:18] = [small random velocities]
        # self.data now contains: standing pose for env 2
        
        # Step 3: Forward kinematics
        mujoco.mj_forward(self.model, self.data)
        # self.data now has consistent derived quantities
        
        # Step 4: Extract and store state
        next_states[i] = self._get_state(self.data)
        # next_states[2] = [0, 0, 0.3, ..., 0.0, 0.9, -1.8, ..., 0, 0, ...]
        #                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                   This is a COPY, independent of self.data
```

### Why This Works

1. **`self.data` is temporary**: We use it to *construct* a state, then immediately extract it
2. **State vectors are independent**: `next_states[i]` is a separate array, not a reference
3. **Each iteration is isolated**: Overwriting `self.data` in iteration `i=2` doesn't affect `next_states[0]` or `next_states[1]`

### Analogy

Think of `self.data` as a **calculator** and state vectors as **numbers written on paper**:

```
Calculator (self.data):  [temporary workspace]
Paper (next_states):     [0.5, 1.2, 3.7, 2.1, ...]
                          ↑    ↑    ↑    ↑
                         env0 env1 env2 env3

Process:
1. Clear calculator
2. Enter numbers for env 2
3. Press "=" (forward kinematics)
4. Write result on paper at position 2
5. (Calculator can now be reused for env 3)
```

---

## The Rollout Module

### How `rollout.rollout()` Works

```python
state_traj, sensor_traj = rollout.rollout(
    model,              # Shared model
    datas,              # Thread-local MjData instances (for parallel work)
    initial_states,     # (num_envs, state_size) - starting states
    ctrl_sequences,     # (num_envs, nstep, nu) - control inputs
    nstep=1            # Number of steps to simulate
)
```

**Internally** (simplified):

```cpp
// Pseudo-code for rollout.rollout()
for each batch of envs (divided among threads):
    thread_id = get_thread_id()
    mjData* d = datas[thread_id]  // Thread-local workspace
    
    for env_idx in batch:
        // Load state into thread-local data
        mj_setState(model, d, initial_states[env_idx])
        
        // Apply control and step
        d->ctrl = ctrl_sequences[env_idx, 0]
        mj_step(model, d)
        
        // Extract resulting state
        mj_getState(model, d, state_traj[env_idx, 0])
```

**Key insight**: Each thread has its own `MjData` workspace (`datas[thread_id]`), avoiding conflicts.

---

## Why We Need `self.datas` (Plural)

```python
self.datas = [mujoco.MjData(self.model) for _ in range(self.num_threads)]
```

- **Purpose**: Thread-local workspaces for `rollout.rollout()`
- **Why multiple**: Each thread needs its own `MjData` to avoid race conditions
- **Not per-environment**: We have `num_threads` (e.g., 4), not `num_envs` (e.g., 16)

### Thread Assignment

With 16 environments and 4 threads:

```
Thread 0: envs [0, 1, 2, 3]     → uses datas[0]
Thread 1: envs [4, 5, 6, 7]     → uses datas[1]
Thread 2: envs [8, 9, 10, 11]   → uses datas[2]
Thread 3: envs [12, 13, 14, 15] → uses datas[3]
```

Each thread processes its batch sequentially using its own `MjData`.

---

## Why We Need `self.data` (Singular)

```python
self.data = mujoco.MjData(self.model)
```

- **Purpose**: Temporary workspace for operations outside of rollout
- **Used for**:
  1. Converting states to observations (`_state_to_obs`)
  2. Computing rewards (`_compute_rewards_batch`)
  3. Checking termination (`_check_termination_batch`)
  4. Generating reset states (in the reset loop)

**Not used during rollout** - that uses `self.datas`.

---

## Common Misconceptions

### ❌ Misconception 1: "We need one MjData per environment"

**Reality**: We use state vectors to represent environments. `MjData` instances are just temporary workspaces.

### ❌ Misconception 2: "Overwriting self.data in the loop breaks previous environments"

**Reality**: We extract and store state vectors immediately. Once stored in `next_states[i]`, they're independent.

### ❌ Misconception 3: "The rollout module creates separate simulations"

**Reality**: The rollout module uses state-based simulation with thread-local workspaces. It's more like "save state → step → load state" than separate simulations.

---

## Performance Implications

### Memory Usage

**Traditional approach** (16 envs):
```
16 × sizeof(MjData) ≈ 16 × 50 KB = 800 KB
```

**Our approach** (16 envs, 4 threads):
```
4 × sizeof(MjData) + 16 × sizeof(state_vector)
≈ 4 × 50 KB + 16 × 0.3 KB
≈ 200 KB + 5 KB = 205 KB
```

**Savings**: ~4x less memory

### Speed

**Traditional approach**: Sequential stepping
```
Time = num_envs × step_time
     = 16 × 1 ms = 16 ms
```

**Our approach**: Parallel rollout
```
Time = (num_envs / num_threads) × step_time
     = (16 / 4) × 1 ms = 4 ms
```

**Speedup**: ~4x faster (matches thread count)

Plus additional optimizations from MuJoCo's C++ implementation!

---

## Verification

You can verify this works correctly by checking that each environment maintains independent state:

```python
# Test script
vec_env = VectorizedQuadrupedEnv(num_envs=4)
obs, _ = vec_env.reset(seed=42)

# Step with different actions
actions = np.array([
    [0.0] * 12,   # Env 0: zero action
    [0.5] * 12,   # Env 1: positive action
    [-0.5] * 12,  # Env 2: negative action
    [0.0] * 12,   # Env 3: zero action
])

obs, rewards, _, _, _ = vec_env.step(actions)

# Check that environments have different states
print(f"Env 0 height: {obs[0, 24]:.3f}")  # Should be similar to env 3
print(f"Env 1 height: {obs[1, 24]:.3f}")  # Should be different
print(f"Env 2 height: {obs[2, 24]:.3f}")  # Should be different
print(f"Env 3 height: {obs[3, 24]:.3f}")  # Should be similar to env 0
```

If the environments were sharing state incorrectly, they would all have the same observations.

---

## Summary

**Your intuition was correct** to question whether we need per-environment indexing! The answer is:

✅ **We DO have per-environment state** - stored in `next_states[i]`  
✅ **We DON'T need per-environment MjData** - we use state vectors instead  
✅ **`self.data` is a temporary workspace** - used to generate/process states  
✅ **The loop correctly resets each environment** - by generating and storing independent state vectors

This architecture is what makes the vectorized environment so efficient - we get the benefits of parallel simulation without the memory overhead of maintaining separate `MjData` instances for each environment!

---

**Key Takeaway**: In state-based simulation, the **state vector** is the environment, not the `MjData` instance. `MjData` is just a tool for manipulating states.

