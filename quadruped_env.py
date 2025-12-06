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
        camera_mode: str = 'follow',  # 'follow', 'fixed', 'side', 'top'
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
            camera_mode: Camera tracking mode - 'follow' (tracks robot), 'fixed' (static), 'side' (side view), 'top' (top-down)
        """
        super().__init__()
        
        self.model_path = model_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.camera_mode = camera_mode
        
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
            self.model.actuator_gainprm[:, 0] *= stiffness_scale  # pyright: ignore
            self.model.actuator_biasprm[:, 0] = -35.0
        
        self.data = mujoco.MjData(self.model)  # pyright: ignore
        
        # Reward weights (can be customized)
        self.reward_weights = reward_weights or {
            'forward_velocity': 1.0,
            'alive_bonus': 0.5,
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
        
        # Renderer and camera (lazy initialization)
        self._renderer = None
        self._camera = None
        
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
        exp_vel = np.exp(-(forward_vel - 0.40) ** 2 / 0.30 ** 2)
        vel_reward = self.reward_weights['forward_velocity'] * exp_vel
        
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
    
    def _update_camera(self):
        """
        Update camera position based on camera_mode.
        Inspired by stream_unitree_go1.py camera following feature.
        """
        if self._camera is None:
            return
        
        # Get robot base position (center of the robot)
        robot_pos = self.data.qpos[0:3].copy()  # [x, y, z]
        
        if self.camera_mode == "follow":
            # Follow camera: tracks robot from behind and above
            lookat_offset = np.array([0.2, 0.0, 0.0])  # Look slightly ahead
            
            self._camera.lookat[:] = robot_pos + lookat_offset
            self._camera.distance = 1.5
            self._camera.azimuth = 90  # View from behind
            self._camera.elevation = -20  # Slight downward angle
            
        elif self.camera_mode == "side":
            # Side view: follows robot from the side
            self._camera.lookat[:] = robot_pos
            self._camera.distance = 2.5
            self._camera.azimuth = 0  # View from side
            self._camera.elevation = -15
            
        elif self.camera_mode == "top":
            # Top-down view: bird's eye view
            self._camera.lookat[:] = robot_pos
            self._camera.distance = 3.0
            self._camera.azimuth = 90
            self._camera.elevation = -89  # Almost straight down
            
        elif self.camera_mode == "fixed":
            # Fixed view: static camera at origin
            self._camera.lookat[:] = np.array([0.0, 0.0, 0.3])
            self._camera.distance = 3.0
            self._camera.azimuth = 90
            self._camera.elevation = -20
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            
            # Initialize camera if not already created
            if self._camera is None:
                self._camera = mujoco.MjvCamera()  # pyright: ignore
                mujoco.mjv_defaultFreeCamera(self.model, self._camera)  # pyright: ignore
            
            # Update camera position based on robot position
            self._update_camera()
            
            # Render with camera
            self._renderer.update_scene(self.data, camera=self._camera)
            pixels = self._renderer.render()
            return pixels
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            del self._renderer
            self._renderer = None
        if self._camera is not None:
            del self._camera
            self._camera = None

# Convenience function to create vectorized environment
def make_quadruped_env(
    num_envs: int = 16,
    model_path: Optional[str] = None,
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
