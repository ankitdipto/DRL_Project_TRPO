#!/usr/bin/env python3
"""
CPG-based Quadruped Environment (PMTG approach).

This environment wraps the standard quadruped environment and uses a
Central Pattern Generator (CPG) to convert high-level policy outputs
into low-level joint commands.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# from cpg_generators import TrottingGaitGenerator, get_cpg_generator


class QuadrupedEnvCPG(gym.Env):
    """
    Quadruped environment with CPG-based control.
    
    The policy outputs CPG parameters instead of direct joint commands:
    - Action space: 16D (frequency, amplitudes, stance offsets)
    - CPG generates 12D joint commands from these parameters
    - Observation space: same as standard environment (34D)
    """
    
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    
    def __init__(
        self,
        model_path: str,
        gait_type: str = 'trot',
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
        Initialize CPG-based quadruped environment.
        
        Args:
            model_path: Path to MuJoCo XML model
            gait_type: Type of CPG gait ('trot', 'bound', 'adaptive')
            render_mode: Rendering mode
            max_episode_steps: Max steps per episode
            reward_weights: Reward component weights
            frame_skip: Simulation steps per environment step
            timestep: MuJoCo timestep (overrides XML default)
            damping_scale: Scale factor for joint damping (default: 1.0, use <1.0 to reduce damping)
            stiffness_scale: Scale factor for actuator stiffness kp (default: 1.0, use <1.0 to reduce stiffness)
            camera_mode: Camera tracking mode - 'follow' (tracks robot), 'fixed' (static), 'side' (side view), 'top' (top-down)
        """
        super().__init__()
        
        self.model_path = model_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.gait_type = gait_type
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
            self.model.actuator_gainprm[:, 0] *= stiffness_scale
            self.model.actuator_biasprm[:, 1] = -35.0
        
        self.data = mujoco.MjData(self.model)  # pyright: ignore
        
        # Initialize CPG
        # self.cpg = get_cpg_generator(gait_type)
        
        # Reward weights (same as standard env, but added CPG-specific rewards)
        self.reward_weights = reward_weights or {
            'forward_velocity': 1.0,
            'alive_bonus': 0.1,
            'orientation_penalty': 0.0,
            'energy_cost': 0.0,  # Slightly lower since CPG generates smoother motions
            'joint_limit_penalty': 0.0,
            'height_penalty': 0.0,
            'lateral_stability': 0.0,
            'angular_stability': 0.0,
            'smoothness_bonus': 0.0,  # NEW: CPG naturally smooth
        }
        
        # Define observation space (34 dimensions - same as before)
        obs_high = np.inf * np.ones(34, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-obs_high, high=obs_high, dtype=np.float32
        )
        
        action_low = np.array([-1.0] * 12, dtype=np.float32)
        action_high = np.array([1.0] * 12, dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Renderer and camera
        self._renderer = None
        self._camera = None
        
        # Previous joint commands for smoothness reward
        self._prev_joint_commands = np.zeros(12, dtype=np.float32)
    
    def _get_trotting_base_controller(self, t: float, frequency: float = 1.0) -> np.ndarray:
        """Get trotting base controller."""
        standing_pose = np.array([
            0.0, 0.9, -1.8,  # FR
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8,  # RL    
        ])

        AMP_HIP = 0.0
        AMP_THIGH = 0.3
        AMP_CALF = 0.3
        
        phase = 2 * np.pi * frequency * t
        
        leg_phases = np.array([0.0, np.pi, np.pi, 0.0])
        
        base_controller = np.zeros(12)
        
        for leg_idx in range(4):
            leg_phase = phase + leg_phases[leg_idx]
            
            base_controller[leg_idx * 3 + 0] = standing_pose[leg_idx * 3 + 0] + \
                AMP_HIP * np.sin(leg_phase)
            
            base_controller[leg_idx * 3 + 1] = standing_pose[leg_idx * 3 + 1] + \
                AMP_THIGH * np.sin(leg_phase)
            
            base_controller[leg_idx * 3 + 2] = standing_pose[leg_idx * 3 + 2] - \
                AMP_CALF * np.maximum(0, np.sin(leg_phase))
            
        return base_controller

    
    def _get_obs(self) -> np.ndarray:
        """Get observation (same as standard environment)."""
        qpos_joints = self.data.qpos[7:19].copy()
        qvel_joints = self.data.qvel[6:18].copy()
        base_quat = self.data.qpos[3:7].copy()
        base_linvel = self.data.qvel[0:3].copy()
        base_angvel = self.data.qvel[3:6].copy()
        
        obs = np.concatenate([
            qpos_joints, qvel_joints, base_quat, base_linvel, base_angvel
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, joint_commands: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward based on current state.
        
        Args:
            joint_commands: (12,) joint commands generated by CPG
            
        Returns:
            reward: Scalar reward
            info: Dictionary with reward components
        """
        # Forward velocity (PRIMARY OBJECTIVE)
        forward_vel = self.data.qvel[0]
        exp_vel = np.exp(-(forward_vel - 0.40) ** 2 / 0.30 ** 2)
        vel_reward = self.reward_weights['forward_velocity'] * exp_vel
        
        # Alive bonus
        alive_bonus = self.reward_weights['alive_bonus']
        
        # Orientation penalty
        quat_w = self.data.qpos[6]
        orientation_penalty = self.reward_weights['orientation_penalty'] * (1.0 - quat_w) ** 2
        
        # Energy cost (on joint commands, not action)
        energy_cost = self.reward_weights['energy_cost'] * np.sum(joint_commands ** 2)
        
        # Joint limit penalty
        joint_positions = self.data.qpos[7:19]
        joint_limits_low = self.model.actuator_ctrlrange[:, 0]
        joint_limits_high = self.model.actuator_ctrlrange[:, 1]
        joint_violations = (
            np.maximum(0, joint_positions - joint_limits_high) +
            np.maximum(0, joint_limits_low - joint_positions)
        )
        joint_limit_penalty = self.reward_weights['joint_limit_penalty'] * np.sum(joint_violations ** 2)
        
        # Height penalty
        base_height = self.data.qpos[2]
        target_height = 0.30
        height_penalty = self.reward_weights['height_penalty'] * (base_height - target_height) ** 2
        
        # Lateral stability
        lateral_vel = self.data.qvel[1]
        lateral_penalty = self.reward_weights['lateral_stability'] * lateral_vel ** 2
        
        # Angular stability
        angular_vel_z = self.data.qvel[5]
        angular_penalty = self.reward_weights['angular_stability'] * angular_vel_z ** 2
        
        # Smoothness bonus (CPG naturally produces smooth motions)
        # Reward small changes in joint commands
        command_diff = np.sum((joint_commands - self._prev_joint_commands) ** 2)
        smoothness_bonus = self.reward_weights['smoothness_bonus'] * np.exp(-command_diff)
        
        # Total reward
        reward = (
            vel_reward +
            alive_bonus +
            smoothness_bonus -
            orientation_penalty -
            energy_cost -
            joint_limit_penalty -
            height_penalty -
            lateral_penalty -
            angular_penalty
        )
        
        info = {
            'reward_forward': vel_reward,
            'reward_alive': alive_bonus,
            'reward_smoothness': smoothness_bonus,
            'reward_orientation': -orientation_penalty,
            'reward_energy': -energy_cost,
            'reward_joint_limits': -joint_limit_penalty,
            'reward_height': -height_penalty,
            'reward_lateral': -lateral_penalty,
            'reward_angular': -angular_penalty,
            'forward_velocity': forward_vel,
            'base_height': base_height,
            'orientation_w': quat_w,
        }
        
        return float(reward), info
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        base_height = self.data.qpos[2]
        if base_height < 0.10:
            return True
        return False
    
    def _is_truncated(self) -> bool:
        """Check if episode should truncate."""
        return self.current_step >= self.max_episode_steps
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)  # pyright: ignore
        
        # Set initial pose (standing)
        standing_pose = np.array([
            0.0, 0.9, -1.8,  # FR
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8,  # RL
        ])
        self.data.qpos[7:19] = standing_pose
        
        # Add small perturbations
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[7:19] += np.random.uniform(-0.05, 0.05, size=12)
        self.data.qvel[6:18] = np.random.uniform(-0.1, 0.1, size=12)
        
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore
        
        # Reset tracking
        self.current_step = 0
        self.episode_reward = 0.0
        self._prev_joint_commands = standing_pose.copy()
        
        obs = self._get_obs()
        info = {'reset': True}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: (16,) CPG parameters
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Clip action to valid range
        #print("action shape: ", action.shape)
        res_action = np.clip(action, self.action_space.low, self.action_space.high)  # pyright: ignore
        
        res_action = 0.2 * res_action
        t = self.current_step * self.model.opt.timestep * self.frame_skip
        joint_commands = self._get_trotting_base_controller(t) + res_action

        # Clip joint commands to actuator limits
        joint_commands = np.clip(
            joint_commands,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1]
        )
        
        # Apply joint commands
        self.data.ctrl[:] = joint_commands
        
        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)  # pyright: ignore
        
        # Get observation and reward
        obs = self._get_obs()
        reward, reward_info = self._compute_reward(joint_commands)
        
        # Update previous commands
        self._prev_joint_commands = joint_commands.copy()
        
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
        """Render the environment."""
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

