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
        """
        super().__init__()
        
        self.model_path = model_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.gait_type = gait_type
        
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
        
        # Define action space (16 dimensions - CPG parameters)
        # [frequency(1), hip_amp(1), thigh_amp(1), calf_amp(1), stance_offsets(12)]
        # action_low = np.array([0.1, -0.86, -0.68, -2.82] + [-0.1] * 12, dtype=np.float32)
        # action_high = np.array([1.0, 0.86, 4.50, -0.89] + [0.1] * 12, dtype=np.float32)
        # self.action_space = spaces.Box(
        #     low=action_low, high=action_high, dtype=np.float32
        # )
        action_low = np.array([-1.0] * 12, dtype=np.float32)
        action_high = np.array([1.0] * 12, dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Renderer
        self._renderer = None
        
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
            
        # print(f"Current Step: {self.current_step}")
        # print(f"t: {t}")
        # print(f"leg_phase: {leg_phase}")
        # print(f"hip: {base_controller[leg_idx * 3 + 0]}")
        # print(f"thigh: {base_controller[leg_idx * 3 + 1]}")
        # print(f"calf: {base_controller[leg_idx * 3 + 2]}")
        return base_controller

    def _parse_cpg_params(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Parse action array into CPG parameter dictionary.
        
        Args:
            action: (16,) array of CPG parameters
            
        Returns:
            Dictionary with CPG parameters
        """

        raise NotImplementedError("Not implemented")
        params = {
            'frequency': float(action[0]),
            'hip_amplitude': float(action[1]),
            'thigh_amplitude': float(action[2]),
            'calf_amplitude': float(action[3]),
            'stance_offset': action[4:16].astype(np.float32),
        }
        return params
    
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
        vel_reward = self.reward_weights['forward_velocity'] * forward_vel
        
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
        
        # Reset CPG
        # self.cpg.reset()
        
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
        
        # Parse CPG parameters
        # cpg_params = self._parse_cpg_params(action)
        
        # Generate joint commands from CPG
        # dt = self.model.opt.timestep * self.frame_skip
        # joint_commands = self.cpg.generate(cpg_params, dt)
        res_action = 0.2 * res_action
        t = self.current_step * self.model.opt.timestep * self.frame_skip
        joint_commands = self._get_trotting_base_controller(t) + res_action
        # print(f"residual action: {res_action}")
        # print(f"base controller: {self._get_trotting_base_controller(self.current_step)}")
        # print(f"joint commands: {joint_commands}")

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
            # 'cpg_frequency': cpg_params['frequency'],
            # 'cpg_amp_mean': np.mean([
            #     cpg_params['hip_amplitude'],
            #     cpg_params['thigh_amplitude'],
            #     cpg_params['calf_amplitude']
            #]),
        }
        
        if terminated:
            info['termination_reason'] = 'fell'
        elif truncated:
            info['termination_reason'] = 'max_steps'
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
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


if __name__ == "__main__":
    """Test CPG-based environment."""
    print("=" * 70)
    print("Testing CPG-Based Quadruped Environment")
    print("=" * 70)
    
    # Load environment
    menagerie_path = "/home/asinha389/Documents/DRL_Project_TRPO/mujoco_menagerie"
    model_path = os.path.join(menagerie_path, "unitree_go1/scene.xml")
    
    print(f"\nCreating environment...")
    env = QuadrupedEnvCPG(
        model_path=model_path,
        gait_type='trot',
        max_episode_steps=200
    )
    
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Action space: CPG parameters (16D)")
    print(f"    [0]: frequency [0.5, 3.0] Hz")
    print(f"    [1]: hip amplitude [0.0, 0.3] rad")
    print(f"    [2]: thigh amplitude [0.0, 0.8] rad")
    print(f"    [3]: calf amplitude [0.0, 1.2] rad")
    print(f"    [4-15]: stance offsets [-0.5, 0.5] rad")
    
    # Test reset
    print(f"\nTesting reset...")
    obs, info = env.reset(seed=42)
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  Initial base height: {env.data.qpos[2]:.3f} m")
    
    # Test steps with fixed CPG parameters
    print(f"\nTesting steps with fixed CPG parameters...")
    
    # Set reasonable CPG parameters
    action = np.array([
        1.5,    # frequency: 1.5 Hz
        0.1,    # hip amplitude
        0.3,    # thigh amplitude
        0.4,    # calf amplitude
        0.0, 0.0, 0.0,  # FR offsets
        0.0, 0.0, 0.0,  # FL offsets
        0.0, 0.0, 0.0,  # RR offsets
        0.0, 0.0, 0.0,  # RL offsets
    ], dtype=np.float32)
    
    total_reward = 0.0
    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"  Step {step}: reward={reward:.3f}, height={info['base_height']:.3f}, "
                  f"vel={info['forward_velocity']:.3f}")
        
        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward / (step + 1):.3f}")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("CPG environment test complete!")
    print("\nKey differences from standard environment:")
    print("  - Action space: 16D (CPG params) vs 12D (joint positions)")
    print("  - Built-in periodicity via CPG")
    print("  - Smoother joint trajectories")
    print("  - Easier to learn structured gaits")

