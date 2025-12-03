#!/usr/bin/env python3
"""
Central Pattern Generators (CPG) for Quadruped Locomotion.

This module provides trajectory generators that produce periodic joint motions
for quadruped robots. These can be modulated by learned policies (PMTG approach).
"""

import numpy as np
from typing import Dict, Tuple


class TrottingGaitGenerator:
    """
    Central Pattern Generator for trotting gait.
    
    Trotting is a symmetrical gait where diagonal leg pairs move together:
    - Phase 1: FR (front-right) + RL (rear-left) in stance
    - Phase 2: FL (front-left) + RR (rear-right) in stance
    
    Joint order: [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
                  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
    """
    
    def __init__(self):
        """Initialize the trotting gait generator."""
        self.phase = 0.0
        
        # Phase offsets for each leg (trotting pattern)
        # FR=0°, FL=180°, RR=180°, RL=0°
        self.leg_phase_offsets = np.array([0.0, np.pi, np.pi, 0.0])
        
        # Default stance pose (standing configuration)
        self.default_stance = np.array([
            0.0, 0.9, -1.8,  # FR
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8,  # RL
        ], dtype=np.float32)
        
    def reset(self):
        """Reset the generator phase."""
        self.phase = 0.0
    
    def generate(self, params: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """
        Generate joint position commands for one timestep.
        
        Args:
            params: Dictionary containing:
                - 'frequency': float, oscillation frequency in Hz (0.5-3.0)
                - 'hip_amplitude': float, hip joint amplitude in radians
                - 'thigh_amplitude': float, thigh joint amplitude in radians
                - 'calf_amplitude': float, calf joint amplitude in radians
                - 'stance_offset': (12,) array, stance pose offset for each joint
            dt: timestep in seconds
            
        Returns:
            joint_commands: (12,) array of joint position targets
        """
        # Extract parameters
        frequency = float(params['frequency'])
        amp_hip = float(params['hip_amplitude'])
        amp_thigh = float(params['thigh_amplitude'])
        amp_calf = float(params['calf_amplitude'])
        stance_offset = params['stance_offset']
        
        # Update phase
        self.phase += 2 * np.pi * frequency * dt
        self.phase = self.phase % (2 * np.pi)
        
        # Generate commands for each leg
        commands = np.zeros(12, dtype=np.float32)
        
        for leg_idx in range(4):
            # Compute phase for this leg
            leg_phase = self.phase + self.leg_phase_offsets[leg_idx]
            
            # Base stance pose
            base_hip = self.default_stance[leg_idx * 3 + 0]
            base_thigh = self.default_stance[leg_idx * 3 + 1]
            base_calf = self.default_stance[leg_idx * 3 + 2]
            
            # Add learned offsets
            base_hip += stance_offset[leg_idx * 3 + 0]
            base_thigh += stance_offset[leg_idx * 3 + 1]
            base_calf += stance_offset[leg_idx * 3 + 2]
            
            # Hip joint (abduction/adduction) - minimal motion
            commands[leg_idx * 3 + 0] = amp_hip * np.sin(leg_phase) + base_hip
            
            # Thigh joint (forward/backward swing)
            commands[leg_idx * 3 + 1] = amp_thigh * np.sin(leg_phase) + base_thigh
            
            # Calf joint (knee flexion/extension)
            # Use different phase relationship for foot clearance
            # During swing phase (sin > 0), retract leg (more negative calf angle)
            calf_signal = np.maximum(0, np.sin(leg_phase))  # Only during swing
            commands[leg_idx * 3 + 2] = -amp_calf * calf_signal + base_calf
        
        return commands


class BoundingGaitGenerator:
    """
    Central Pattern Generator for bounding gait.
    
    Bounding: front legs move together, rear legs move together.
    - Phase 1: Front legs in swing, rear legs in stance
    - Phase 2: Front legs in stance, rear legs in swing
    """
    
    def __init__(self):
        """Initialize the bounding gait generator."""
        self.phase = 0.0
        
        # Phase offsets for bounding: front pair vs rear pair
        # FR=0°, FL=0°, RR=180°, RL=180°
        self.leg_phase_offsets = np.array([0.0, 0.0, np.pi, np.pi])
        
        self.default_stance = np.array([
            0.0, 0.9, -1.8,  # FR
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # RR
            0.0, 0.9, -1.8,  # RL
        ], dtype=np.float32)
    
    def reset(self):
        """Reset the generator phase."""
        self.phase = 0.0
    
    def generate(self, params: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """Generate joint commands (same interface as TrottingGaitGenerator)."""
        frequency = float(params['frequency'])
        amp_hip = float(params['hip_amplitude'])
        amp_thigh = float(params['thigh_amplitude'])
        amp_calf = float(params['calf_amplitude'])
        stance_offset = params['stance_offset']
        
        self.phase += 2 * np.pi * frequency * dt
        self.phase = self.phase % (2 * np.pi)
        
        commands = np.zeros(12, dtype=np.float32)
        
        for leg_idx in range(4):
            leg_phase = self.phase + self.leg_phase_offsets[leg_idx]
            
            base_hip = self.default_stance[leg_idx * 3 + 0] + stance_offset[leg_idx * 3 + 0]
            base_thigh = self.default_stance[leg_idx * 3 + 1] + stance_offset[leg_idx * 3 + 1]
            base_calf = self.default_stance[leg_idx * 3 + 2] + stance_offset[leg_idx * 3 + 2]
            
            commands[leg_idx * 3 + 0] = amp_hip * np.sin(leg_phase) + base_hip
            commands[leg_idx * 3 + 1] = amp_thigh * np.sin(leg_phase) + base_thigh
            calf_signal = np.maximum(0, np.sin(leg_phase))
            commands[leg_idx * 3 + 2] = -amp_calf * calf_signal + base_calf
        
        return commands


class AdaptiveCPG:
    """
    Adaptive CPG that can blend between multiple gait patterns.
    
    This allows the policy to learn when to use different gaits
    (e.g., trot at low speed, bound at high speed).
    """
    
    def __init__(self):
        """Initialize with multiple gait generators."""
        self.trot_gen = TrottingGaitGenerator()
        self.bound_gen = BoundingGaitGenerator()
        self.generators = [self.trot_gen, self.bound_gen]
        
    def reset(self):
        """Reset all generators."""
        for gen in self.generators:
            gen.reset()
    
    def generate(self, params: Dict[str, np.ndarray], dt: float) -> np.ndarray:
        """
        Generate joint commands by blending multiple gaits.
        
        Args:
            params: Must include 'gait_weights' in addition to standard params
                - 'gait_weights': (2,) array, blend weights for [trot, bound]
                - Other params as in TrottingGaitGenerator
        """
        gait_weights = params['gait_weights']
        gait_weights = np.abs(gait_weights) / (np.sum(np.abs(gait_weights)) + 1e-8)
        
        # Generate commands from each gait
        commands = np.zeros(12, dtype=np.float32)
        for weight, gen in zip(gait_weights, self.generators):
            gen_commands = gen.generate(params, dt)
            commands += weight * gen_commands
        
        return commands


def get_cpg_generator(gait_type: str = 'trot'):
    """
    Factory function to create CPG generators.
    
    Args:
        gait_type: 'trot', 'bound', or 'adaptive'
        
    Returns:
        CPG generator instance
    """
    if gait_type == 'trot':
        return TrottingGaitGenerator()
    elif gait_type == 'bound':
        return BoundingGaitGenerator()
    elif gait_type == 'adaptive':
        return AdaptiveCPG()
    else:
        raise ValueError(f"Unknown gait type: {gait_type}")


if __name__ == "__main__":
    """Test CPG generators."""
    print("Testing CPG Generators")
    print("=" * 70)
    
    # Test trotting generator
    print("\n1. Testing TrottingGaitGenerator:")
    trot_gen = TrottingGaitGenerator()
    
    params = {
        'frequency': 1.5,  # 1.5 Hz
        'hip_amplitude': 0.1,
        'thigh_amplitude': 0.3,
        'calf_amplitude': 0.4,
        'stance_offset': np.zeros(12),
    }
    
    dt = 0.01  # 10ms timestep
    print(f"  Frequency: {params['frequency']} Hz")
    print(f"  Timestep: {dt} s")
    
    # Generate a few steps
    for i in range(5):
        commands = trot_gen.generate(params, dt)
        print(f"  Step {i}: FR_thigh={commands[1]:.3f}, FL_thigh={commands[4]:.3f}")
    
    print("\n2. Testing phase relationships:")
    trot_gen.reset()
    
    # Generate one full cycle
    period = 1.0 / params['frequency']
    n_steps = int(period / dt)
    
    fr_thigh_trajectory = []
    fl_thigh_trajectory = []
    
    for _ in range(n_steps):
        commands = trot_gen.generate(params, dt)
        fr_thigh_trajectory.append(commands[1])
        fl_thigh_trajectory.append(commands[4])
    
    fr_thigh_trajectory = np.array(fr_thigh_trajectory)
    fl_thigh_trajectory = np.array(fl_thigh_trajectory)
    
    print(f"  FR thigh range: [{fr_thigh_trajectory.min():.3f}, {fr_thigh_trajectory.max():.3f}]")
    print(f"  FL thigh range: [{fl_thigh_trajectory.min():.3f}, {fl_thigh_trajectory.max():.3f}]")
    
    # Check phase difference (should be ~π for trotting)
    fr_max_idx = np.argmax(fr_thigh_trajectory)
    fl_max_idx = np.argmax(fl_thigh_trajectory)
    phase_diff = abs(fr_max_idx - fl_max_idx) * dt * 2 * np.pi * params['frequency']
    print(f"  Phase difference: {phase_diff:.3f} rad (expected: {np.pi:.3f} rad)")
    
    print("\n" + "=" * 70)
    print("CPG tests complete!")
    print("\nUsage:")
    print("  cpg = TrottingGaitGenerator()")
    print("  params = {'frequency': 1.5, 'hip_amplitude': 0.1, ...}")
    print("  commands = cpg.generate(params, dt=0.01)")

