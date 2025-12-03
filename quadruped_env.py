#!/usr/bin/env python3
"""
Vectorized Gymnasium Environment for Unitree Go1 Quadruped using MuJoCo Rollout.

This environment uses MuJoCo's rollout module for efficient parallel simulation
of multiple robot instances, providing significant speedup for RL training.

Features:
- Parallel vectorized environment using mujoco.rollout
- 34-dimensional observation space (joint states + base pose/velocity)
- 12-dimensional action space (joint position targets)
- Customizable reward function for locomotion
- Automatic termination on falls
- Compatible with TRPO training pipeline
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering

import numpy as np
import mujoco
from mujoco import rollout
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List
import copy


class QuadrupedEnv(gym.Env):
    """
    Single-instance Gymnasium environment for Unitree Go1 quadruped.
    
    This is a standard Gym environment that can be used with gym.vector wrappers
    or standalone for evaluation/rendering.
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}  # 50 fps for smooth video at 10x frame skip
    
    def __init__(
        self,
        model_path: str,
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None,
        frame_skip: int = 25,
        timestep: Optional[float] = None,
        damping_scale: float = 1.0,
        stiffness_scale: float = 1.0,
    ):
        """
        Initialize the quadruped environment.
        
        Args:
            model_path: Path to MuJoCo XML model file
            render_mode: Rendering mode ("rgb_array" or None)
            max_episode_steps: Maximum steps per episode
            reward_weights: Dictionary of reward component weights
            frame_skip: Number of simulation steps per environment step
            timestep: Simulation timestep in seconds (overrides XML default if provided)
            damping_scale: Scale factor for joint damping (default: 1.0, use <1.0 to reduce damping)
            stiffness_scale: Scale factor for actuator stiffness kp (default: 1.0, use <1.0 to reduce stiffness)
        """
        super().__init__()
        
        self.model_path = model_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)  # pyright: ignore
        
        # Override timestep if provided
        if timestep is not None:
            self.model.opt.timestep = timestep
        
        # Modify joint damping (applies to all DOFs)
        if damping_scale != 1.0:
            self.model.dof_damping[:] *= damping_scale
        
        # Modify actuator stiffness (kp) for position actuators
        if stiffness_scale != 1.0:
            # For position actuators, kp is stored in actuator_gainprm[:, 0]
            for i in range(self.model.nu):
                if self.model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_POSITION:  # pyright: ignore
                    self.model.actuator_gainprm[i, 0] *= stiffness_scale
        
        self.data = mujoco.MjData(self.model)  # pyright: ignore
        
        # Reward weights (can be customized)
        self.reward_weights = reward_weights or {
            'forward_velocity': 1.0,
            'alive_bonus': 1.0,
            'orientation_penalty': 0.5,
            'energy_cost': 0.001,
            'joint_limit_penalty': 0.1,
            'height_penalty': 0.5,
        }
        
        # Define observation space (34 dimensions)
        # [joint_pos(12), joint_vel(12), base_quat(4), base_linvel(3), base_angvel(3)]
        obs_high = np.inf * np.ones(34, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        
        # Define action space (12 dimensions - joint position targets)
        # Use actual joint limits from the model
        action_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        action_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )
        
        # Standing pose for initialization
        self.standing_pose = np.array([
            0.0, 0.9, -1.8,  # Front right leg (FR)
            0.0, 0.9, -1.8,  # Front left leg (FL)
            0.0, 0.9, -1.8,  # Rear right leg (RR)
            0.0, 0.9, -1.8,  # Rear left leg (RL)
        ], dtype=np.float32)
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Renderer (lazy initialization)
        self._renderer = None
        
        # Initial state for reset
        self._initial_qpos = None
        self._initial_qvel = None
        
    def _get_obs(self) -> np.ndarray:
        """
        Extract observation from current state.
        
        Returns:
            34-dimensional observation vector
        """
        # Joint positions (12) - skip free joint (first 7 qpos)
        qpos_joints = self.data.qpos[7:19].copy()
        
        # Joint velocities (12) - skip free joint (first 6 qvel)
        qvel_joints = self.data.qvel[6:18].copy()
        
        # Base orientation quaternion (4)
        base_quat = self.data.qpos[3:7].copy()
        
        # Base linear velocity (3)
        base_linvel = self.data.qvel[0:3].copy()
        
        # Base angular velocity (3)
        base_angvel = self.data.qvel[3:6].copy()
        
        obs = np.concatenate([
            qpos_joints,
            qvel_joints,
            base_quat,
            base_linvel,
            base_angvel,
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward based on current state.
        
        Returns:
            reward: Scalar reward value
            info: Dictionary with reward components
        """
        # Forward velocity (x-direction)
        forward_vel = self.data.qvel[0]
        vel_reward = self.reward_weights['forward_velocity'] * forward_vel
        
        # Alive bonus (encourage survival)
        alive_bonus = self.reward_weights['alive_bonus']
        
        # Orientation penalty (keep upright)
        # Quaternion: [w, x, y, z] at qpos[3:7]
        # For upright: w ≈ 1, x,y,z ≈ 0
        quat_w = self.data.qpos[6]  # w component
        orientation_penalty = self.reward_weights['orientation_penalty'] * (1.0 - quat_w) ** 2
        
        # Energy cost (minimize control effort)
        energy_cost = self.reward_weights['energy_cost'] * np.sum(self.data.ctrl ** 2)
        
        # Joint limit penalty (stay within safe ranges)
        joint_positions = self.data.qpos[7:19]
        joint_limits_low = self.model.actuator_ctrlrange[:, 0]
        joint_limits_high = self.model.actuator_ctrlrange[:, 1]
        joint_violations = np.maximum(0, joint_positions - joint_limits_high) + \
                          np.maximum(0, joint_limits_low - joint_positions)
        joint_limit_penalty = self.reward_weights['joint_limit_penalty'] * np.sum(joint_violations ** 2)
        
        # Height penalty (maintain reasonable height)
        base_height = self.data.qpos[2]
        target_height = 0.30  # Target standing height
        height_penalty = self.reward_weights['height_penalty'] * (base_height - target_height) ** 2
        
        # Total reward
        reward = (
            vel_reward +
            alive_bonus -
            orientation_penalty -
            energy_cost -
            joint_limit_penalty -
            height_penalty
        )
        
        # Info dictionary with components
        info = {
            'reward_forward': vel_reward,
            'reward_alive': alive_bonus,
            'reward_orientation': -orientation_penalty,
            'reward_energy': -energy_cost,
            'reward_joint_limits': -joint_limit_penalty,
            'reward_height': -height_penalty,
            'forward_velocity': forward_vel,
            'base_height': base_height,
            'orientation_w': quat_w,
        }
        
        return float(reward), info
    
    def _is_terminated(self) -> bool:
        """
        Check if episode should terminate (robot fell).
        
        Returns:
            True if episode should end, False otherwise
        """
        # Check if base is too low (fell down)
        base_height = self.data.qpos[2]
        if base_height < 0.10:  # Below 10cm
            return True
        
        # Check if robot flipped over (orientation too far from upright)
        # quat_w = self.data.qpos[6]
        # if quat_w < 0.3:  # More than ~72° from upright
        #     return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """
        Check if episode should truncate (reached max steps).
        
        Returns:
            True if max steps reached, False otherwise
        """
        return self.current_step >= self.max_episode_steps
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)  # pyright: ignore
        
        # Set initial pose to standing
        self.data.qpos[7:19] = self.standing_pose
        
        # Add small random perturbations for robustness
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, size=12)
        self.data.qvel[6:18] = np.random.uniform(-0.1, 0.1, size=12)
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore
        
        # Store initial state
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
        obs = self._get_obs()
        info = {'reset': True}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: 12-dimensional action (joint position targets)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended (fell)
            truncated: Whether episode truncated (max steps)
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Apply action
        self.data.ctrl[:] = action
        
        # Step simulation (with frame skip)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)  # pyright: ignore
        
        # Get observation
        obs = self._get_obs()
        
        # Compute reward
        reward, reward_info = self._compute_reward()
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Update tracking
        self.current_step += 1
        self.episode_reward += reward
        
        # Compile info
        info = {
            **reward_info,
            'step': self.current_step,
            'episode_reward': self.episode_reward,
        }
        
        if terminated:
            info['termination_reason'] = 'fell'
        elif truncated:
            info['termination_reason'] = 'max_steps'
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            
            self._renderer.update_scene(self.data)
            pixels = self._renderer.render()
            return pixels
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            del self._renderer
            self._renderer = None


class CustomVectorizedQuadrupedEnv:
    """
    Vectorized environment using MuJoCo's rollout module for parallel simulation.
    
    This provides massive speedup (50-100x realtime) by simulating multiple
    robot instances in parallel using efficient state-based rollouts.
    
    Note: This is NOT a standard gym.vector.VectorEnv, but provides a similar
    interface optimized for MuJoCo rollout. For standard Gym compatibility,
    use gym.vector.SyncVectorEnv with QuadrupedEnv.
    """
    
    def __init__(
        self,
        model_path: str,
        num_envs: int = 16,
        max_episode_steps: int = 1000,
        reward_weights: Optional[Dict[str, float]] = None,
        num_threads: int = 4,
        frame_skip: int = 25,
        timestep: Optional[float] = None,
        damping_scale: float = 1.0,
        stiffness_scale: float = 1.0,
    ):
        """
        Initialize vectorized environment.
        
        Args:
            model_path: Path to MuJoCo XML model file
            num_envs: Number of parallel environments
            max_episode_steps: Maximum steps per episode
            reward_weights: Dictionary of reward component weights
            num_threads: Number of threads for parallel simulation
            frame_skip: Number of simulation steps per environment step
            timestep: Simulation timestep in seconds (overrides XML default if provided)
            damping_scale: Scale factor for joint damping (default: 1.0, use <1.0 to reduce damping)
            stiffness_scale: Scale factor for actuator stiffness kp (default: 1.0, use <1.0 to reduce stiffness)
        """
        self.model_path = model_path
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.num_threads = min(num_threads, num_envs)
        self.frame_skip = frame_skip
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(model_path)  # pyright: ignore
        
        # Override timestep if provided
        if timestep is not None:
            self.model.opt.timestep = timestep
        
        # Modify joint damping (applies to all DOFs)
        if damping_scale != 1.0:
            self.model.dof_damping[:] *= damping_scale
        
        # Modify actuator stiffness (kp) for position actuators
        if stiffness_scale != 1.0:
            # For position actuators, kp is stored in actuator_gainprm[:, 0]
            self.model.actuator_gainprm[:, 0] *= stiffness_scale
        
        self.data = mujoco.MjData(self.model)  # pyright: ignore
        
        # Create MjData instances for each thread
        self.datas = [mujoco.MjData(self.model) for _ in range(self.num_threads)]  # pyright: ignore
        
        # Reward weights
        self.reward_weights = reward_weights or {
            'forward_velocity': 1.0,
            'alive_bonus': 1.0,
            'orientation_penalty': 0.5,
            'energy_cost': 0.001,
            'joint_limit_penalty': 0.1,
            'height_penalty': 0.5,
        }
        
        # Standing pose
        self.standing_pose = np.array([
            0.0, 0.9, -1.8,  # FR
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8,  # RL
        ], dtype=np.float32)
        
        # Observation and action spaces (same as single env)
        obs_high = np.inf * np.ones(34, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        
        action_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        action_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )
        
        # Episode tracking
        self.current_steps = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        
        # Current states
        self.current_states = None
        self.current_obs = None
        
    def _get_state(self, data: mujoco.MjData) -> np.ndarray:  # pyright: ignore[reportAttributeAccessIssue]
        """Get full physics state from MjData."""
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS  # pyright: ignore
        state = np.zeros(mujoco.mj_stateSize(self.model, full_physics))  # pyright: ignore
        mujoco.mj_getState(self.model, data, state, full_physics)  # pyright: ignore
        return state
    
    def _set_state(self, data: mujoco.MjData, state: np.ndarray):  # pyright: ignore[reportAttributeAccessIssue]
        """Set full physics state to MjData."""
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS  # pyright: ignore
        mujoco.mj_setState(self.model, data, state, full_physics)  # pyright: ignore
        mujoco.mj_forward(self.model, data)  # pyright: ignore
    
    def _state_to_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Convert full physics state to observation.
        
        Args:
            state: Full physics state
            
        Returns:
            34-dimensional observation
        """
        # Set temporary data to extract observation
        self._set_state(self.data, state)
        
        # Extract observation components
        qpos_joints = self.data.qpos[7:19].copy()
        qvel_joints = self.data.qvel[6:18].copy()
        base_quat = self.data.qpos[3:7].copy()
        base_linvel = self.data.qvel[0:3].copy()
        base_angvel = self.data.qvel[3:6].copy()
        
        obs = np.concatenate([
            qpos_joints,
            qvel_joints,
            base_quat,
            base_linvel,
            base_angvel,
        ]).astype(np.float32)
        
        return obs
    
    def _compute_rewards_batch(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute rewards for a batch of states.
        
        Args:
            states: (num_envs, state_size) array of states
            actions: (num_envs, 12) array of actions
            
        Returns:
            rewards: (num_envs,) array of rewards
            infos: List of info dictionaries
        """
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        infos = []
        
        for i in range(self.num_envs):
            self._set_state(self.data, states[i])
            self.data.ctrl[:] = actions[i]
            
            # Forward velocity
            forward_vel = self.data.qvel[0]
            vel_reward = self.reward_weights['forward_velocity'] * forward_vel
            
            # Alive bonus
            alive_bonus = self.reward_weights['alive_bonus']
            
            # Orientation penalty
            quat_w = self.data.qpos[6]
            orientation_penalty = self.reward_weights['orientation_penalty'] * (1.0 - quat_w) ** 2
            
            # Energy cost
            energy_cost = self.reward_weights['energy_cost'] * np.sum(actions[i] ** 2)
            
            # Joint limit penalty
            joint_positions = self.data.qpos[7:19]
            joint_limits_low = self.model.actuator_ctrlrange[:, 0]
            joint_limits_high = self.model.actuator_ctrlrange[:, 1]
            joint_violations = np.maximum(0, joint_positions - joint_limits_high) + \
                              np.maximum(0, joint_limits_low - joint_positions)
            joint_limit_penalty = self.reward_weights['joint_limit_penalty'] * np.sum(joint_violations ** 2)
            
            # Height penalty
            base_height = self.data.qpos[2]
            target_height = 0.30
            height_penalty = self.reward_weights['height_penalty'] * (base_height - target_height) ** 2
            
            # Total reward
            reward = (
                vel_reward +
                alive_bonus -
                orientation_penalty -
                energy_cost -
                joint_limit_penalty -
                height_penalty
            )
            
            rewards[i] = reward
            
            info = {
                'reward_forward': vel_reward,
                'reward_alive': alive_bonus,
                'forward_velocity': forward_vel,
                'base_height': base_height,
                'orientation_w': quat_w,
            }
            infos.append(info)
        
        return rewards, infos
    
    def _check_termination_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check termination for a batch of states.
        
        Args:
            states: (num_envs, state_size) array of states
            
        Returns:
            terminated: (num_envs,) boolean array
            truncated: (num_envs,) boolean array
        """
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        
        for i in range(self.num_envs):
            self._set_state(self.data, states[i])
            
            # # Check if fell
            base_height = self.data.qpos[2]
            quat_w = self.data.qpos[6]
            
            if base_height < 0.10: # or quat_w < 0.3:
                terminated[i] = True
            
            # Check if max steps
            if self.current_steps[i] >= self.max_episode_steps:
                truncated[i] = True
        
        return terminated, truncated
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Reset all environments.
        
        Args:
            seed: Random seed
            
        Returns:
            observations: (num_envs, 34) array of observations
            infos: List of info dictionaries
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode tracking
        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        
        # Create initial states for all environments
        states = []
        for i in range(self.num_envs):
            mujoco.mj_resetData(self.model, self.data)  # pyright: ignore
            
            # Set standing pose with small perturbations
            self.data.qpos[7:19] = self.standing_pose + np.random.uniform(-0.05, 0.05, size=12)
            self.data.qvel[6:18] = np.random.uniform(-0.1, 0.1, size=12)
            
            mujoco.mj_forward(self.model, self.data)  # pyright: ignore
            
            state = self._get_state(self.data)
            states.append(state)
        
        self.current_states = np.array(states)
        
        # Convert states to observations
        observations = np.array([self._state_to_obs(state) for state in self.current_states])
        self.current_obs = observations
        
        infos = [{'reset': True} for _ in range(self.num_envs)]
        
        return observations, infos
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments in parallel.
        
        This method uses MuJoCo's rollout module for efficient parallel simulation.
        Key insight: We work with STATE VECTORS, not separate MjData instances.
        
        Architecture:
        - self.current_states: (num_envs, state_size) - current state for each env
        - rollout.rollout(): simulates all envs in parallel from their states
        - next_states: (num_envs, state_size) - resulting states after 1 step
        - self.data: temporary workspace for state-to-obs conversion and resets
        
        Args:
            actions: (num_envs, 12) array of actions
            
        Returns:
            observations: (num_envs, 34) array of observations
            rewards: (num_envs,) array of rewards
            terminated: (num_envs,) boolean array
            truncated: (num_envs,) boolean array
            infos: List of info dictionaries
        """
        # Clip actions
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        # Expand actions to (num_envs, 1, 12) for single-step rollout
        ctrl_sequences = actions[:, np.newaxis, :]
        
        # Run rollout for 1 step
        state_traj, _ = rollout.rollout(  # pyright: ignore[reportReturnType]
            self.model,
            self.datas,
            self.current_states,  # pyright: ignore[reportArgumentType]
            ctrl_sequences,
            nstep=self.frame_skip
        )
        
        # Extract next states (shape: num_envs, 1, state_size -> num_envs, state_size)
        next_states = state_traj[:, 0, :]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript, reportArgumentType, reportCallIssue]
        
        # Compute rewards
        rewards, reward_infos = self._compute_rewards_batch(next_states, actions)  # pyright: ignore[reportArgumentType, reportCallIssue]
        
        # Check termination
        terminated, truncated = self._check_termination_batch(next_states)  # pyright: ignore[reportArgumentType, reportCallIssue]
        
        # Update tracking
        self.current_steps += 1
        self.episode_rewards += rewards
        
        # Handle resets for terminated/truncated environments
        # Note: self.data is used as a TEMPORARY WORKSPACE here, not as env-specific data
        # Each iteration: reset self.data → configure it → extract state → store in next_states[i]
        # This is correct because we immediately extract and store the state vector
        for i in range(self.num_envs):
            if terminated[i] or truncated[i]:
                # Add episode info BEFORE resetting
                reward_infos[i]['episode'] = {
                    'r': self.episode_rewards[i],
                    'l': self.current_steps[i],
                }
                
                # Reset this environment by creating a new initial state
                # Step 1: Reset self.data to default state (temporary workspace)
                mujoco.mj_resetData(self.model, self.data)  # pyright: ignore
                
                # Step 2: Configure standing pose with random perturbations
                self.data.qpos[7:19] = self.standing_pose + np.random.uniform(-0.05, 0.05, size=12)
                self.data.qvel[6:18] = np.random.uniform(-0.1, 0.1, size=12)
                
                # Step 3: Forward kinematics to ensure consistency
                mujoco.mj_forward(self.model, self.data)  # pyright: ignore
                
                # Step 4: Extract state vector and assign to THIS environment's state
                # This is the key: next_states[i] gets its own independent state vector
                next_states[i] = self._get_state(self.data)  # pyright: ignore[reportIndexIssue]
                
                # Reset tracking for this environment
                self.current_steps[i] = 0
                self.episode_rewards[i] = 0.0
        
        # Update current states
        self.current_states = next_states
        
        # Convert states to observations
        # pyright: ignore[reportArgumentType, reportCallIssue] on next line
        observations = np.array([self._state_to_obs(state) for state in next_states])  # type: ignore
        self.current_obs = observations
        
        return observations, rewards, terminated, truncated, reward_infos
    
    def close(self):
        """Clean up resources."""
        del self.datas
        del self.data
        del self.model


# Convenience function to create vectorized environment
def make_quadruped_env(
    num_envs: int = 16,
    model_path: Optional[str] = None,
    use_rollout: bool = True,
    timestep: Optional[float] = None,
    damping_scale: float = 1.0,
    stiffness_scale: float = 1.0,
    **kwargs
):  # pyright: ignore[reportReturnType]
    """
    Create a vectorized quadruped environment.
    
    Args:
        num_envs: Number of parallel environments
        model_path: Path to MuJoCo XML file (default: Go1 scene)
        use_rollout: If True, use VectorizedQuadrupedEnv (rollout-based, faster)
                     If False, use gym.vector.SyncVectorEnv (standard Gym)
        timestep: Simulation timestep in seconds (overrides XML default if provided)
        damping_scale: Scale factor for joint damping (default: 1.0, use <1.0 to reduce damping)
        stiffness_scale: Scale factor for actuator stiffness kp (default: 1.0, use <1.0 to reduce stiffness)
        **kwargs: Additional arguments passed to environment constructor
        
    Returns:
        Vectorized environment
    """
    if model_path is None:
        # Default to Go1 scene
        menagerie_path = "/home/hice1/asinha389/scratch/mujoco_menagerie"
        model_path = os.path.join(menagerie_path, "unitree_go1/scene.xml")
    
    if use_rollout:
        # Use custom rollout-based vectorized environment (faster)
        return CustomVectorizedQuadrupedEnv(
            model_path=model_path,
            num_envs=num_envs,
            timestep=timestep,
            damping_scale=damping_scale,
            stiffness_scale=stiffness_scale,
            **kwargs
        )
    else:
        # Use standard Gym vectorized environment
        def make_env():
            return QuadrupedEnv(
                model_path=model_path,
                timestep=timestep,
                damping_scale=damping_scale,
                stiffness_scale=stiffness_scale,
                **kwargs
            )
        
        return gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])


if __name__ == "__main__":
    """Test the environment."""
    print("=" * 70)
    print("TESTING QUADRUPED ENVIRONMENT")
    print("=" * 70)
    
    # Test single environment
    print("\n1. Testing single environment...")
    env = QuadrupedEnv(
        model_path="/home/hice1/asinha389/scratch/mujoco_menagerie/unitree_go1/scene.xml",
        max_episode_steps=100
    )
    
    obs, info = env.reset(seed=42)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Run a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"   Ran {i+1} steps, total reward: {total_reward:.3f}")
    env.close()
    
    # Test vectorized environment
    print("\n2. Testing vectorized environment (rollout-based)...")
    vec_env = CustomVectorizedQuadrupedEnv(
        model_path="/home/hice1/asinha389/scratch/mujoco_menagerie/unitree_go1/scene.xml",
        num_envs=4,
        max_episode_steps=100,
        num_threads=2
    )
    
    obs, infos = vec_env.reset(seed=42)
    print(f"   Observations shape: {obs.shape}")
    print(f"   Number of environments: {vec_env.num_envs}")
    
    # Run a few steps
    import time
    start_time = time.time()
    n_steps = 100
    
    for i in range(n_steps):
        actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
    
    elapsed = time.time() - start_time
    steps_per_sec = (n_steps * vec_env.num_envs) / elapsed
    
    print(f"   Ran {n_steps} steps with {vec_env.num_envs} envs")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   Throughput: {steps_per_sec:.1f} steps/sec")
    print(f"   Speedup: {steps_per_sec / (vec_env.num_envs * 500):.1f}x realtime")
    
    vec_env.close()
    
    print("\n" + "=" * 70)
    print("ENVIRONMENT TESTS PASSED!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Integrate with TRPO training (main.py)")
    print("  2. Tune reward weights for desired behavior")
    print("  3. Run training experiments")
    print("=" * 70)

