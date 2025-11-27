#!/usr/bin/env python3
"""
Multi-robot simulation using MuJoCo's rollout module.

This script demonstrates efficient parallel simulation of multiple Go1 robots
using the rollout module. Each robot starts at a different position and can have
different control policies.

Key concept: Instead of creating multiple robots in one XML, we simulate
multiple instances of the same robot model with different initial states.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering

import numpy as np
import mujoco
from mujoco import rollout
import imageio
from pathlib import Path
import time
import copy


# Paths
MENAGERIE_PATH = "/home/hice1/asinha389/scratch/mujoco_menagerie"
GO1_SCENE_PATH = os.path.join(MENAGERIE_PATH, "unitree_go1/scene.xml")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_state(model, data, nbatch=1):
    """
    Get the full state of the system and replicate it for nbatch instances.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        nbatch: Number of batch instances
    
    Returns:
        Array of shape (nbatch, state_size) with replicated states
    """
    full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS  # pyright: ignore[reportAttributeAccessIssue]
    
    state = np.zeros((mujoco.mj_stateSize(model, full_physics),))  # pyright: ignore[reportAttributeAccessIssue]
    mujoco.mj_getState(model, data, state, full_physics)  # pyright: ignore[reportAttributeAccessIssue]
    
    return np.tile(state, (nbatch, 1))


def get_standing_pose():
    """Standing pose for Go1."""
    return np.array([
        0.0, 0.9, -1.8,  # Front right leg (FR)
        0.0, 0.9, -1.8,  # Front left leg (FL)
        0.0, 0.9, -1.8,  # Rear right leg (RR)
        0.0, 0.9, -1.8,  # Rear left leg (RL)
    ])


def get_walking_gait(t, frequency=1.0, gait_type="trot", phase_offset=0.0):
    """
    Generate walking gait using CPG (Central Pattern Generator).
    
    Args:
        t: Time in seconds
        frequency: Gait frequency in Hz
        gait_type: "trot", "walk", "pace", or "bound"
        phase_offset: Phase offset for this robot (to desynchronize)
    
    Returns:
        12D action vector (joint positions)
    """
    standing = get_standing_pose()
    
    # Gait phase relationships
    if gait_type == "trot":
        # Diagonal pairs move together: (FR, RL) and (FL, RR)
        phases = np.array([0, np.pi, np.pi, 0])
    elif gait_type == "walk":
        # Sequential: FR -> FL -> RR -> RL
        phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    elif gait_type == "pace":
        # Lateral pairs: (FR, FL) and (RR, RL)
        phases = np.array([0, 0, np.pi, np.pi])
    elif gait_type == "bound":
        # Front and rear pairs: (FR, FL) and (RR, RL)
        phases = np.array([0, 0, np.pi, np.pi])
    else:
        phases = np.array([0, np.pi, np.pi, 0])  # Default to trot
    
    # Add global phase offset
    phases = phases + phase_offset
    
    # Amplitude of motion
    hip_amplitude = 0.4
    knee_amplitude = 0.6
    
    action = standing.copy()
    
    for leg in range(4):
        leg_phase = 2 * np.pi * frequency * t + phases[leg]
        
        # Hip joint (thigh) - forward/backward motion
        action[leg * 3 + 1] = standing[leg * 3 + 1] + hip_amplitude * np.sin(leg_phase)
        
        # Knee joint (calf) - up/down motion with phase shift for swing
        swing_phase = np.sin(leg_phase)
        if swing_phase < 0:  # Swing phase
            action[leg * 3 + 2] = standing[leg * 3 + 2] + knee_amplitude * swing_phase
        else:  # Stance phase
            action[leg * 3 + 2] = standing[leg * 3 + 2] + 0.3 * knee_amplitude * swing_phase
    
    return action


def generate_control_sequence(model, nbatch, nstep, control_modes):
    """
    Generate control sequences for all robot instances.
    
    Args:
        model: MuJoCo model
        nbatch: Number of robot instances
        nstep: Number of simulation steps
        control_modes: List of control modes for each robot
    
    Returns:
        Array of shape (nbatch, nstep, nu) with control sequences
    """
    print(f"\nGenerating control sequences for {nbatch} robots...")
    
    ctrl_sequences = np.zeros((nbatch, nstep, model.nu))
    
    for batch_idx in range(nbatch):
        mode = control_modes[batch_idx]
        
        for step in range(nstep):
            t = step * model.opt.timestep
            
            if mode == "standing":
                action = get_standing_pose()
            elif mode == "zero":
                action = np.zeros(model.nu)
            elif mode == "random":
                action = np.random.uniform(
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1]
                )
            elif mode in ["trot", "walk", "pace", "bound"]:
                # Add phase offset based on batch index to desynchronize
                phase_offset = batch_idx * np.pi / 4
                action = get_walking_gait(t, frequency=1.0, gait_type=mode, phase_offset=phase_offset)
            else:
                action = get_standing_pose()
            
            ctrl_sequences[batch_idx, step] = action
    
    print(f"✓ Control sequences generated: shape {ctrl_sequences.shape}")
    return ctrl_sequences


def setup_initial_states(model, data, nbatch, spacing=2.5):
    """
    Setup initial states for multiple robot instances at different positions.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        nbatch: Number of robot instances
        spacing: Distance between robots (meters)
    
    Returns:
        Array of shape (nbatch, state_size) with initial states
    """
    print(f"\nSetting up initial states for {nbatch} robots...")
    
    # Reset to default state
    mujoco.mj_resetData(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Set standing pose
    standing = get_standing_pose()
    data.qpos[7:19] = standing  # Joint positions (skip 7 DOF floating base)
    
    # Forward kinematics
    mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Get base state and replicate
    initial_states = get_state(model, data, nbatch)
    
    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(nbatch)))
    rows = int(np.ceil(nbatch / cols))
    
    # Modify initial positions for each robot
    for i in range(nbatch):
        row = i // cols
        col = i % cols
        
        # Calculate position
        x = (col - (cols - 1) / 2) * spacing
        y = (row - (rows - 1) / 2) * spacing
        z = 0.35  # Starting height
        
        # Update position in state (first 3 elements of qpos)
        initial_states[i, 0] = x
        initial_states[i, 1] = y
        initial_states[i, 2] = z
        
        print(f"  Robot {i}: position [{x:.2f}, {y:.2f}, {z:.2f}]")
    
    print(f"✓ Initial states configured: shape {initial_states.shape}")
    return initial_states


def simulate_multi_robot(model, nbatch, duration=8.0, nthread=4, control_modes=None):
    """
    Simulate multiple robot instances using rollout.
    
    Args:
        model: MuJoCo model
        nbatch: Number of robot instances
        duration: Simulation duration in seconds
        nthread: Number of threads for parallel simulation
        control_modes: List of control modes for each robot
    
    Returns:
        Tuple of (state_trajectory, sensor_trajectory, control_sequences)
    """
    print(f"\n{'='*70}")
    print("MULTI-ROBOT ROLLOUT SIMULATION")
    print(f"{'='*70}")
    
    # Create data
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Calculate number of steps
    nstep = int(duration / model.opt.timestep)
    
    print(f"\nSimulation parameters:")
    print(f"  Number of robots: {nbatch}")
    print(f"  Duration: {duration}s")
    print(f"  Timestep: {model.opt.timestep}s")
    print(f"  Steps: {nstep}")
    print(f"  Threads: {nthread}")
    
    # Setup control modes
    if control_modes is None:
        # Default: assign different modes to different robots
        modes = ["standing", "trot", "walk", "pace"]
        control_modes = [modes[i % len(modes)] for i in range(nbatch)]
    
    print(f"\nControl modes:")
    for i, mode in enumerate(control_modes):
        print(f"  Robot {i}: {mode}")
    
    # Setup initial states
    initial_states = setup_initial_states(model, data, nbatch, spacing=2.5)
    
    # Generate control sequences
    ctrl_sequences = generate_control_sequence(model, nbatch, nstep, control_modes)
    
    # Create MjData instances for each thread
    print(f"\nCreating {nthread} MjData instances for parallel simulation...")
    datas = [copy.copy(data) for _ in range(nthread)]
    print(f"✓ MjData instances created")
    
    # Run rollout
    print(f"\n{'='*70}")
    print("RUNNING ROLLOUT...")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # rollout.rollout returns (state, sensordata)
    # state shape: (nbatch, nstep, state_size)
    state_traj, sensor_traj = rollout.rollout(  # pyright: ignore[reportAttributeAccessIssue]
        model, 
        datas, 
        initial_states,
        ctrl_sequences,
        nstep=nstep
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Rollout completed!")
    print(f"  Wall time: {elapsed_time:.2f}s")
    print(f"  Simulation time: {duration}s × {nbatch} robots = {duration * nbatch}s")
    print(f"  Speed: {(duration * nbatch) / elapsed_time:.1f}x realtime")
    print(f"  State shape: {state_traj.shape}")
    print(f"  Sensor shape: {sensor_traj.shape}")
    
    return state_traj, sensor_traj, ctrl_sequences, control_modes


def render_multi_robot(model, state_traj, control_modes, output_path, fps=30):
    """
    Render multiple robot trajectories into a single video showing all robots.
    
    Args:
        model: MuJoCo model
        state_traj: State trajectory array (nbatch, nstep, state_size)
        control_modes: List of control modes for labeling
        output_path: Path to save video
        fps: Frames per second
    """
    print(f"\n{'='*70}")
    print("RENDERING VIDEO")
    print(f"{'='*70}")
    
    nbatch, nstep, _ = state_traj.shape
    
    # Create data for rendering
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Create renderer (use default framebuffer size)
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    # Create camera
    camera = mujoco.MjvCamera()  # pyright: ignore[reportAttributeAccessIssue]
    mujoco.mjv_defaultFreeCamera(model, camera)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Set camera to view all robots from above
    camera.lookat[:] = [0.0, 0.0, 0.3]
    camera.distance = 4.0
    camera.azimuth = 90
    camera.elevation = -30
    
    # Calculate steps per frame
    steps_per_frame = max(1, int((1.0 / fps) / model.opt.timestep))
    
    print(f"\nRendering parameters:")
    print(f"  Robots: {nbatch}")
    print(f"  Steps: {nstep}")
    print(f"  FPS: {fps}")
    print(f"  Steps per frame: {steps_per_frame}")
    print(f"  Expected frames: {nstep // steps_per_frame}")
    
    frames = []
    
    # We'll render each robot separately and composite them
    # For simplicity, we'll render them sequentially in the same scene
    # by updating the robot position for each batch
    
    print("\nRendering frames...")
    for step_idx in range(0, nstep, steps_per_frame):
        # For visualization, we'll render just one robot at a time
        # In a real application, you'd want to composite all robots into one frame
        # For now, let's create a grid view
        
        # Render the first robot as example
        # TODO: Composite all robots into single frame
        batch_idx = 0  # Start with first robot
        
        # Get state for this robot at this timestep
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_setState(model, data, state_traj[batch_idx, step_idx], full_physics)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Forward kinematics
        mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Render
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)
        
        if len(frames) % 30 == 0:
            print(f"  Rendered {len(frames)} frames...")
    
    # Save video
    print(f"\nSaving video to: {output_path}")
    imageio.mimsave(output_path, frames, fps=fps)
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"✓ Video saved: {len(frames)} frames, {file_size:.1f} KB")


def render_all_robots_composite(model, state_traj, control_modes, output_path, fps=30):
    """
    Render all robots in a composite view (grid layout).
    
    This creates a video with multiple viewports, one for each robot.
    """
    print(f"\n{'='*70}")
    print("RENDERING COMPOSITE VIDEO (ALL ROBOTS)")
    print(f"{'='*70}")
    
    nbatch, nstep, _ = state_traj.shape
    
    # Create data for rendering
    data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(nbatch)))
    rows = int(np.ceil(nbatch / cols))
    
    # Create renderer for each robot (smaller viewports)
    viewport_width = 300
    viewport_height = 225
    renderer = mujoco.Renderer(model, height=viewport_height, width=viewport_width)
    
    # Create camera
    camera = mujoco.MjvCamera()  # pyright: ignore[reportAttributeAccessIssue]
    mujoco.mjv_defaultFreeCamera(model, camera)  # pyright: ignore[reportAttributeAccessIssue]
    
    # Set camera
    camera.lookat[:] = [0.0, 0.0, 0.3]
    camera.distance = 3.0
    camera.azimuth = 90
    camera.elevation = -20
    
    # Calculate steps per frame
    steps_per_frame = max(1, int((1.0 / fps) / model.opt.timestep))
    
    print(f"\nRendering parameters:")
    print(f"  Robots: {nbatch}")
    print(f"  Grid: {rows}x{cols}")
    print(f"  Viewport size: {viewport_width}x{viewport_height}")
    print(f"  Total size: {cols*viewport_width}x{rows*viewport_height}")
    print(f"  Steps: {nstep}")
    print(f"  FPS: {fps}")
    
    frames = []
    full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS  # pyright: ignore[reportAttributeAccessIssue]
    
    print("\nRendering frames...")
    for step_idx in range(0, nstep, steps_per_frame):
        # Create composite frame
        composite = np.zeros((rows * viewport_height, cols * viewport_width, 3), dtype=np.uint8)
        
        # Render each robot
        for batch_idx in range(nbatch):
            row = batch_idx // cols
            col = batch_idx % cols
            
            # Get state for this robot at this timestep
            mujoco.mj_setState(model, data, state_traj[batch_idx, step_idx], full_physics)  # pyright: ignore[reportAttributeAccessIssue]
            mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
            
            # Render
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            
            # Place in composite
            y_start = row * viewport_height
            y_end = y_start + viewport_height
            x_start = col * viewport_width
            x_end = x_start + viewport_width
            
            composite[y_start:y_end, x_start:x_end] = pixels
        
        frames.append(composite)
        
        if len(frames) % 30 == 0:
            print(f"  Rendered {len(frames)} frames...")
    
    # Save video
    print(f"\nSaving video to: {output_path}")
    imageio.mimsave(output_path, frames, fps=fps)
    
    file_size = os.path.getsize(output_path) / 1024  # KB
    print(f"✓ Video saved: {len(frames)} frames, {file_size:.1f} KB")


def analyze_trajectories(model, state_traj, control_modes):
    """Analyze and print statistics about all robot trajectories."""
    print(f"\n{'='*70}")
    print("TRAJECTORY ANALYSIS")
    print(f"{'='*70}")
    
    nbatch, nstep, _ = state_traj.shape
    
    for batch_idx in range(nbatch):
        print(f"\nRobot {batch_idx} ({control_modes[batch_idx]} mode):")
        
        # Extract positions over time (first 3 elements of qpos)
        positions = state_traj[batch_idx, :, 0:3]
        
        initial_pos = positions[0]
        final_pos = positions[-1]
        
        # Calculate displacement
        horizontal_disp = np.linalg.norm(final_pos[:2] - initial_pos[:2])
        
        print(f"  Initial position: [{initial_pos[0]:6.3f}, {initial_pos[1]:6.3f}, {initial_pos[2]:6.3f}]")
        print(f"  Final position:   [{final_pos[0]:6.3f}, {final_pos[1]:6.3f}, {final_pos[2]:6.3f}]")
        print(f"  Horizontal displacement: {horizontal_disp:.3f} m")
        print(f"  Height range: {positions[:, 2].min():.3f} - {positions[:, 2].max():.3f} m")
        print(f"  Final height: {final_pos[2]:.3f} m")


def main():
    """Main function."""
    print("=" * 70)
    print("MULTI-ROBOT SIMULATION WITH MUJOCO ROLLOUT")
    print("=" * 70)
    
    # Configuration
    nbatch = 4  # Number of robot instances
    duration = 8.0  # seconds
    nthread = 4  # Number of parallel threads
    fps = 30
    
    print(f"\nConfiguration:")
    print(f"  Number of robots: {nbatch}")
    print(f"  Duration: {duration}s")
    print(f"  Threads: {nthread}")
    print(f"  FPS: {fps}")
    
    # Load model
    print(f"\n{'='*70}")
    print("LOADING MODEL")
    print(f"{'='*70}")
    print(f"Model path: {GO1_SCENE_PATH}")
    
    model = mujoco.MjModel.from_xml_path(GO1_SCENE_PATH)  # pyright: ignore[reportAttributeAccessIssue]
    
    print(f"✓ Model loaded")
    print(f"  Bodies: {model.nbody}")
    print(f"  Joints: {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  DOF: {model.nv}")
    print(f"  Timestep: {model.opt.timestep}s")
    
    # Run simulation
    state_traj, sensor_traj, ctrl_sequences, control_modes = simulate_multi_robot(
        model, nbatch, duration, nthread
    )
    
    # Analyze trajectories
    analyze_trajectories(model, state_traj, control_modes)
    
    # Render videos
    output_path_single = OUTPUT_DIR / f"multi_go1_{nbatch}robots_single.mp4"
    render_multi_robot(model, state_traj, control_modes, output_path_single, fps)
    
    output_path_composite = OUTPUT_DIR / f"multi_go1_{nbatch}robots_composite.mp4"
    render_all_robots_composite(model, state_traj, control_modes, output_path_composite, fps)
    
    # Summary
    print(f"\n{'='*70}")
    print("SIMULATION COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Simulated {nbatch} robots for {duration}s")
    print(f"✓ Videos saved:")
    print(f"    - Single view: {output_path_single}")
    print(f"    - Composite view: {output_path_composite}")
    print("=" * 70)


if __name__ == "__main__":
    main()

