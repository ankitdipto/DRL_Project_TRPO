#!/usr/bin/env python3
"""
Quick verification script to check that all quadruped models can be loaded.
This confirms that MuJoCo Menagerie is properly installed and accessible.
"""

import os
import mujoco

# Base path to mujoco_menagerie
MENAGERIE_PATH = "/home/hice1/asinha389/scratch/mujoco_menagerie"

# Quadruped models to verify
QUADRUPED_MODELS = {
    "Unitree Go1": "unitree_go1/go1.xml",
    "Unitree A1": "unitree_a1/a1.xml",
    "Unitree Go2": "unitree_go2/go2.xml",
    "ANYmal B": "anybotics_anymal_b/anymal_b.xml",
    "ANYmal C": "anybotics_anymal_c/anymal_c.xml",
    "Boston Dynamics Spot": "boston_dynamics_spot/spot.xml",
    "Google Barkour v0": "google_barkour_v0/barkour_v0.xml",
    "Google Barkour vB": "google_barkour_vb/barkour_vb.xml",
}


def verify_model(name, relative_path):
    """Load a model and print basic information."""
    full_path = os.path.join(MENAGERIE_PATH, relative_path)
    
    try:
        model = mujoco.MjModel.from_xml_path(full_path)  # pyright: ignore[reportAttributeAccessIssue]
        data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
        
        print(f"✅ {name}")
        print(f"   Path: {relative_path}")
        print(f"   Bodies: {model.nbody}")
        print(f"   Joints: {model.njnt}")
        print(f"   DOF: {model.nv}")
        print(f"   Actuators: {model.nu}")
        print(f"   Timestep: {model.opt.timestep}s")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ {name}")
        print(f"   Path: {relative_path}")
        print(f"   Error: {str(e)}")
        print()
        return False


def main():
    print("=" * 70)
    print("MuJoCo Menagerie Quadruped Models - Verification")
    print("=" * 70)
    print()
    
    # Check if menagerie path exists
    if not os.path.exists(MENAGERIE_PATH):
        print(f"❌ ERROR: MuJoCo Menagerie not found at {MENAGERIE_PATH}")
        print("Please run: git clone https://github.com/google-deepmind/mujoco_menagerie.git")
        return
    
    print(f"📁 Menagerie Path: {MENAGERIE_PATH}")
    print()
    
    # Verify each model
    results = {}
    for name, path in QUADRUPED_MODELS.items():
        results[name] = verify_model(name, path)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"Successfully loaded: {successful}/{total} models")
    print()
    
    if successful == total:
        print("🎉 All quadruped models verified successfully!")
        print()
        print("✅ Ready to proceed with Sub-task 1.2:")
        print("   - Create custom Gymnasium environment")
        print("   - Test loading and visualization")
        print()
        print("Recommended starting model: Unitree Go1")
    else:
        print("⚠️  Some models failed to load. Check errors above.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

