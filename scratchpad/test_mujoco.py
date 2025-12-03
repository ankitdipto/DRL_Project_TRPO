#!/usr/bin/env python3
"""
Simple example showing how to create and run a basic MuJoCo simulation.
This script demonstrates key MuJoCo concepts:
- Creating a model programmatically
- Running forward dynamics
- Applying controls
- Recording video output
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering

import numpy as np
import mujoco
import imageio
from pathlib import Path


def create_simple_pendulum_xml():
    """Create a simple pendulum model XML."""
    xml = """
    <mujoco model="pendulum">
      <option timestep="0.01" gravity="0 0 -9.81"/>
      
      <visual>
        <global offwidth="640" offheight="480"/>
      </visual>
      
      <asset>
        <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"/>
        <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.2"/>
      </asset>
      
      <worldbody>
        <light diffuse="0.8 0.8 0.8" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1" material="grid"/>
        
        <!-- Fixed base -->
        <body name="base" pos="0 0 1">
          <geom name="base" type="cylinder" size="0.05 0.02" rgba="0.5 0.5 0.5 1"/>
          
          <!-- Pendulum link -->
          <body name="pendulum" pos="0 0 0">
            <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
            <geom name="link" type="capsule" fromto="0 0 0 0 0 -0.6" size="0.02" rgba="0.8 0.2 0.2 1" mass="0.5"/>
            <geom name="bob" type="sphere" pos="0 0 -0.6" size="0.08" rgba="0.2 0.2 0.8 1" mass="1.0"/>
          </body>
        </body>
        
        <camera name="side_view" pos="0 -2.5 1.0" xyaxes="1 0 0 0 0.342 0.940"/>
      </worldbody>
      
      <actuator>
        <motor joint="hinge" name="torque" gear="10" ctrllimited="true" ctrlrange="-1 1"/>
      </actuator>
    </mujoco>
    """
    return xml


def run_simulation(duration=5.0, fps=30, apply_control=True):
    """
    Run a pendulum simulation and save as video.
    
    Args:
        duration: Simulation duration in seconds
        fps: Frames per second for video output
        apply_control: If True, apply torque control to swing up the pendulum
    """
    
    # Create model from XML string
    xml_string = create_simple_pendulum_xml()
    model = mujoco.MjModel.from_xml_string(xml_string)  # pyright: ignore[reportAttributeAccessIssue]
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Set initial condition (pendulum hanging down)
    mujoco.mj_resetData(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[0] = np.pi  # Start at bottom
    data.qvel[0] = 0.0
    
    print(f"Model loaded successfully!")
    print(f"  - Number of bodies: {model.nbody}")
    print(f"  - Number of joints: {model.njnt}")
    print(f"  - Number of actuators: {model.nu}")
    print(f"  - Timestep: {model.opt.timestep} s")
    
    # Create renderer
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Simulation parameters
    frames = []
    frame_duration = 1.0 / fps
    n_frames = int(duration * fps)
    steps_per_frame = int(frame_duration / model.opt.timestep)
    
    print(f"\nRunning simulation...")
    print(f"  - Duration: {duration} s")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {n_frames}")
    print(f"  - Steps per frame: {steps_per_frame}")
    
    # Run simulation
    for i in range(n_frames):
        t = i * frame_duration
        
        if apply_control:
            # Simple swing-up controller
            # Apply torque based on angular velocity and position
            angle = data.qpos[0]
            velocity = data.qvel[0]
            
            # Swing up when hanging down, stabilize when up
            if angle > 0:  # Hanging down
                target_velocity = 3.0
                data.ctrl[0] = np.clip((target_velocity - velocity) * 0.5, -1, 1)
            else:  # Trying to swing up
                # PD control to stabilize upright
                angle_error = 0 - angle  # Target is 0 (upright)
                data.ctrl[0] = np.clip(angle_error * 2.0 - velocity * 0.5, -1, 1)
        else:
            # No control - just let it swing freely
            data.ctrl[0] = 0.0
        
        # Step simulation
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Render
        renderer.update_scene(data, camera="side_view")
        pixels = renderer.render()
        frames.append(pixels.copy())
        
        if (i + 1) % 30 == 0 or i == 0:
            angle_deg = np.rad2deg(data.qpos[0])
            print(f"  Frame {i+1}/{n_frames} | Angle: {angle_deg:6.1f}° | Velocity: {data.qvel[0]:6.2f} rad/s")
    
    # Save video
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    if apply_control:
        output_path = os.path.join(output_dir, 'pendulum_controlled.mp4')
    else:
        output_path = os.path.join(output_dir, 'pendulum_free.mp4')
    
    print(f"\nSaving video to: {output_path}")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264')
    
    print(f"\n{'='*60}")
    print("Simulation complete!")
    print(f"{'='*60}")
    print(f"Video: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Final state:")
    print(f"  - Angle: {np.rad2deg(data.qpos[0]):.1f}°")
    print(f"  - Velocity: {data.qvel[0]:.2f} rad/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("="*60)
    print("MuJoCo Pendulum Simulation Example")
    print("="*60)
    
    # Run with control
    print("\n1. Running controlled swing-up simulation...")
    run_simulation(duration=8.0, fps=30, apply_control=True)
    
    # Run without control (free swing)
    print("\n\n2. Running free-swing simulation...")
    run_simulation(duration=5.0, fps=30, apply_control=False)
    
    print("\n\nAll simulations complete! Check the outputs/ directory for videos.")

