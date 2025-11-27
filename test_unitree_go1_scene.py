#!/usr/bin/env python3
"""
Test script for Unitree Go1 quadruped robot with proper scene (includes ground).
This version loads the scene.xml which includes the robot on a ground plane.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering

import numpy as np
import mujoco
import imageio


# Path to the Unitree Go1 scene (includes robot + ground + lighting)
MENAGERIE_PATH = "/home/hice1/asinha389/scratch/mujoco_menagerie"
GO1_SCENE_PATH = os.path.join(MENAGERIE_PATH, "unitree_go1/scene.xml")


def print_model_info(model):
    """Print detailed information about the loaded model."""
    print("=" * 70)
    print("UNITREE GO1 MODEL INFORMATION")
    print("=" * 70)
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of DOF (nv): {model.nv}")
    print(f"Number of actuators: {model.nu}")
    print(f"Timestep: {model.opt.timestep} s")
    print(f"Gravity: {model.opt.gravity}")
    print()
    
    # Print actuator information
    print("ACTUATORS (Position Control):")
    print("-" * 70)
    for i in range(model.nu):
        actuator_id = i
        # Get actuator name
        name_start = model.name_actuatoradr[actuator_id]
        name_end = name_start
        while name_end < len(model.names) and model.names[name_end] != 0:
            name_end += 1
        actuator_name = model.names[name_start:name_end].decode('utf-8')
        
        # Get control range
        ctrl_range = model.actuator_ctrlrange[actuator_id]
        print(f"  {i:2d}. {actuator_name:20s} | Range: [{ctrl_range[0]:7.3f}, {ctrl_range[1]:7.3f}] rad")
    print("=" * 70)
    print()


def get_observation(model, data):
    """
    Extract observation from the current state.
    Observation includes:
    - Joint positions (12)
    - Joint velocities (12)
    - Body orientation (quaternion, 4)
    - Body linear velocity (3)
    - Body angular velocity (3)
    Total: 34 dimensions
    """
    # Joint positions (skip the free joint, take the 12 actuated joints)
    qpos_joints = data.qpos[7:19]  # Positions 7-18 are the 12 leg joints
    
    # Joint velocities
    qvel_joints = data.qvel[6:18]  # Velocities 6-17 are the 12 leg joints
    
    # Base orientation (quaternion)
    base_quat = data.qpos[3:7]  # Quaternion is at positions 3-6
    
    # Base linear velocity
    base_linvel = data.qvel[0:3]
    
    # Base angular velocity
    base_angvel = data.qvel[3:6]
    
    obs = np.concatenate([
        qpos_joints,    # 12
        qvel_joints,    # 12
        base_quat,      # 4
        base_linvel,    # 3
        base_angvel,    # 3
    ])
    
    return obs


def run_simulation(duration=5.0, fps=30, control_mode="zero", output_name=None):
    """
    Run a simulation of the Unitree Go1 robot with proper scene.
    
    Args:
        duration: Simulation duration in seconds
        fps: Frames per second for video output
        control_mode: "zero" (no control), "random" (random actions), or "standing" (hold standing pose)
        output_name: Custom output filename (without extension)
    """
    
    # Load model with scene (includes ground plane)
    print(f"Loading Unitree Go1 scene from: {GO1_SCENE_PATH}")
    model = mujoco.MjModel.from_xml_path(GO1_SCENE_PATH)  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    
    print_model_info(model)
    
    # Reset to initial state
    mujoco.mj_resetData(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Set initial joint positions to standing pose
    if control_mode == "standing":
        # Standing pose for Go1
        standing_pose = np.array([
            0.0, 0.9, -1.8,  # Front right leg (FR)
            0.0, 0.9, -1.8,  # Front left leg (FL)
            0.0, 0.9, -1.8,  # Rear right leg (RR)
            0.0, 0.9, -1.8,  # Rear left leg (RL)
        ])
        data.qpos[7:19] = standing_pose
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    
    print(f"Initial state:")
    print(f"  Base height: {data.qpos[2]:.3f} m")
    print(f"  Base position: [{data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}]")
    print()
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Simulation parameters
    frames = []
    frame_duration = 1.0 / fps
    n_frames = int(duration * fps)
    steps_per_frame = int(frame_duration / model.opt.timestep)
    
    print(f"Running simulation...")
    print(f"  Duration: {duration} s")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {n_frames}")
    print(f"  Steps per frame: {steps_per_frame}")
    print(f"  Control mode: {control_mode}")
    print()
    
    # Storage for trajectory data
    trajectory = {
        'observations': [],
        'actions': [],
        'base_height': [],
        'base_position': [],
    }
    
    # Run simulation
    for i in range(n_frames):
        t = i * frame_duration
        
        # Determine control action
        if control_mode == "zero":
            # No control - let the robot fall naturally
            action = np.zeros(model.nu)
        elif control_mode == "random":
            # Random actions within control limits
            action = np.random.uniform(
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1]
            )
        elif control_mode == "standing":
            # Try to maintain standing pose with position control
            standing_pose = np.array([
                0.0, 0.9, -1.8,  # FR
                0.0, 0.9, -1.8,  # FL
                0.0, 0.9, -1.8,  # RR
                0.0, 0.9, -1.8,  # RL
            ])
            action = standing_pose
        else:
            action = np.zeros(model.nu)
        
        # Apply control
        data.ctrl[:] = action
        
        # Step simulation
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Get observation
        obs = get_observation(model, data)
        
        # Store trajectory data
        trajectory['observations'].append(obs)
        trajectory['actions'].append(action)
        trajectory['base_height'].append(data.qpos[2])
        trajectory['base_position'].append(data.qpos[0:3].copy())
        
        # Render
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels.copy())
        
        # Print progress
        if (i + 1) % 30 == 0 or i == 0:
            base_height = data.qpos[2]
            base_pos = data.qpos[0:3]
            base_vel = data.qvel[0:3]
            print(f"  Frame {i+1:3d}/{n_frames} | t={t:5.2f}s | "
                  f"Height: {base_height:5.3f}m | "
                  f"Vel: [{base_vel[0]:6.3f}, {base_vel[1]:6.3f}, {base_vel[2]:6.3f}]")
    
    # Convert trajectory to numpy arrays
    trajectory['observations'] = np.array(trajectory['observations'])  # pyright: ignore[reportArgumentType]
    trajectory['actions'] = np.array(trajectory['actions'])  # pyright: ignore[reportArgumentType]
    trajectory['base_height'] = np.array(trajectory['base_height'])  # pyright: ignore[reportArgumentType]
    trajectory['base_position'] = np.array(trajectory['base_position'])  # pyright: ignore[reportArgumentType]
    
    # Save video
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    if output_name is None:
        output_name = f'go1_scene_{control_mode}'
    output_path = os.path.join(output_dir, f'{output_name}.mp4')
    
    print(f"\nSaving video to: {output_path}")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    
    # Print summary
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"Video: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"\nTrajectory Statistics:")
    print(f"  Total steps: {len(trajectory['observations'])}")
    print(f"  Observation shape: {trajectory['observations'].shape}")
    print(f"  Action shape: {trajectory['actions'].shape}")
    print(f"  Final base height: {trajectory['base_height'][-1]:.3f} m")
    print(f"  Min base height: {trajectory['base_height'].min():.3f} m")
    print(f"  Max base height: {trajectory['base_height'].max():.3f} m")
    print(f"  Avg base height: {trajectory['base_height'].mean():.3f} m")
    print(f"  Distance traveled (x): {trajectory['base_position'][-1, 0] - trajectory['base_position'][0, 0]:.3f} m")
    print(f"  Distance traveled (y): {trajectory['base_position'][-1, 1] - trajectory['base_position'][0, 1]:.3f} m")
    print(f"{'='*70}")
    
    return trajectory


if __name__ == "__main__":
    print("=" * 70)
    print("UNITREE GO1 QUADRUPED ROBOT - SCENE SIMULATION TEST")
    print("=" * 70)
    print()
    
    # Test 1: Zero control (robot falls and lands on ground)
    print("\n" + "=" * 70)
    print("TEST 1: Zero Control (Natural Dynamics with Ground)")
    print("=" * 70)
    traj_zero = run_simulation(
        duration=5.0,
        fps=30,
        control_mode="zero",
        output_name="go1_scene_zero_control"
    )
    
    # Test 2: Random control
    print("\n\n" + "=" * 70)
    print("TEST 2: Random Control (with Ground)")
    print("=" * 70)
    traj_random = run_simulation(
        duration=5.0,
        fps=30,
        control_mode="random",
        output_name="go1_scene_random_control"
    )
    
    # Test 3: Standing pose control
    print("\n\n" + "=" * 70)
    print("TEST 3: Standing Pose Control (with Ground)")
    print("=" * 70)
    traj_standing = run_simulation(
        duration=8.0,
        fps=30,
        control_mode="standing",
        output_name="go1_scene_standing_control"
    )
    
    print("\n\n" + "=" * 70)
    print("ALL TESTS COMPLETE!")
    print("=" * 70)
    print("\nGenerated videos (WITH GROUND PLANE):")
    print("  1. outputs/go1_scene_zero_control.mp4     - Robot with no control")
    print("  2. outputs/go1_scene_random_control.mp4   - Robot with random actions")
    print("  3. outputs/go1_scene_standing_control.mp4 - Robot trying to stand")
    print("\nKey Observations:")
    print("  - With ground plane, robot lands and interacts with surface")
    print("  - Standing control should maintain upright posture")
    print("  - Random control shows various dynamic behaviors")
    print("\nNext steps:")
    print("  - Review videos to understand robot-ground interactions")
    print("  - Design reward function for locomotion")
    print("  - Create Gymnasium environment wrapper")
    print("  - Integrate with TRPO training pipeline")
    print("=" * 70)

