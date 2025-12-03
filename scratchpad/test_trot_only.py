#!/usr/bin/env python3
"""
Quick test for in-place trotting gait only.
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_unitree_go1_scene import run_simulation

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING IN-PLACE TROTTING GAIT")
    print("=" * 70)
    
    # Run trotting simulation
    traj = run_simulation(
        duration=10.0,
        fps=30,
        control_mode="trot_in_place",
        output_name="go1_trot_test"
    )
    
    print("\n" + "=" * 70)
    print("TROTTING TEST COMPLETE!")
    print("=" * 70)
    print(f"\nVideo saved: outputs/go1_trot_test.mp4")
    print(f"\nGait Analysis:")
    print(f"  Duration: 10.0s at 1.5 Hz = ~15 step cycles")
    print(f"  Final height: {traj['base_height'][-1]:.3f} m")
    print(f"  Avg height: {traj['base_height'].mean():.3f} m")  # pyright: ignore[reportCallIssue]
    print(f"  Height std: {traj['base_height'].std():.3f} m")  # pyright: ignore[reportCallIssue]
    print(f"  Forward drift: {traj['base_position'][-1, 0]:.3f} m")  # pyright: ignore[reportCallIssue]
    print(f"  Lateral drift: {traj['base_position'][-1, 1]:.3f} m")  # pyright: ignore[reportArgumentType]
    print("\nExpected behavior:")
    print("  - Robot lifts legs in diagonal pairs")
    print("  - FR+RL move together, FL+RR move together")
    print("  - Minimal forward/lateral movement (in-place)")
    print("  - Maintains standing height (~0.28-0.32 m)")
    print("=" * 70)

