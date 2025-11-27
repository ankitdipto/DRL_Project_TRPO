# MuJoCo Menagerie Quadruped Models - Summary Report

**Date**: November 21, 2024  
**Task**: Sub-task 1.1 - Install MuJoCo Menagerie and identify target quadruped models  
**Status**: ✅ COMPLETED

---

## Repository Information

- **Repository**: [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- **Location**: `/home/hice1/asinha389/scratch/mujoco_menagerie`
- **Total Models**: 60+ robot models
- **Quadruped Models**: 8 available

---

## Available Quadruped Robots

### 1. **Unitree Go1** ⭐ RECOMMENDED FOR STARTING
- **Path**: `mujoco_menagerie/unitree_go1/`
- **Model File**: `go1.xml`
- **Scene File**: `scene.xml`
- **DOF**: 12 (3 per leg: abduction, hip, knee)
- **Actuators**: Position-controlled (kp=100)
- **Joint Ranges**:
  - Abduction: -0.863 to 0.863 rad
  - Hip: -0.686 to 4.501 rad
  - Knee: -2.818 to -0.888 rad
- **Force Limits**: -23.7 to 23.7 Nm (hip/abduction), -35.55 to 35.55 Nm (knee)
- **Features**: 
  - Simplified collision geometries
  - Softened foot contacts for rubber approximation
  - High impratio (100) to reduce slippage
  - Derived from official Unitree ROS description
- **Why Start Here**: 
  - Well-documented
  - Medium complexity
  - Good balance of realism and trainability
  - Popular in research community

### 2. **Unitree A1**
- **Path**: `mujoco_menagerie/unitree_a1/`
- **Model File**: `a1.xml`
- **Scene File**: `scene.xml`
- **DOF**: 12 (3 per leg)
- **Features**:
  - Similar to Go1 but slightly different specifications
  - Standing pose at joint position 0
  - Softened foot contacts
- **Comparison to Go1**: Slightly older model, similar capabilities

### 3. **Unitree Go2**
- **Path**: `mujoco_menagerie/unitree_go2/`
- **Model Files**: `go2.xml`, `go2_mjx.xml` (MJX-optimized version)
- **Scene Files**: `scene.xml`, `scene_mjx.xml`
- **DOF**: 12 (3 per leg)
- **Features**:
  - Newer generation Unitree robot
  - MJX-optimized version available for faster training
  - More advanced than Go1
- **Note**: Latest Unitree model, may have more complex dynamics

### 4. **ANYbotics ANYmal B**
- **Path**: `mujoco_menagerie/anybotics_anymal_b/`
- **Model File**: `anymal_b.xml`
- **Scene File**: `scene.xml`
- **DOF**: 12 (3 per leg)
- **Features**:
  - Research-grade quadruped
  - Used extensively in academia
  - More complex dynamics than Unitree models
- **Use Case**: Advanced experiments after mastering simpler models

### 5. **ANYbotics ANYmal C** ⭐ RECOMMENDED FOR ADVANCED WORK
- **Path**: `mujoco_menagerie/anybotics_anymal_c/`
- **Model Files**: `anymal_c.xml`, `anymal_c_mjx.xml`
- **Scene Files**: `scene.xml`, `scene_mjx.xml`
- **DOF**: 12 (3 per leg)
- **Features**:
  - Position-controlled actuators
  - Joint damping for D gains
  - Softened foot contacts
  - **MJX version available** for accelerated training
  - **Official Colab notebook** for policy gradient training
- **Special**: Has a dedicated training example using first-order policy gradients
- **Why Advanced**: More realistic dynamics, research-grade robot

### 6. **Boston Dynamics Spot**
- **Path**: `mujoco_menagerie/boston_dynamics_spot/`
- **Model Files**: `spot.xml`, `spot_arm.xml` (with arm attachment)
- **Scene Files**: `scene.xml`, `scene_arm.xml`
- **DOF**: 12 (base model)
- **Features**:
  - Industry-standard robot
  - Optional arm attachment
  - Complex dynamics
- **Use Case**: Industry-relevant experiments, most realistic

### 7. **Google Barkour v0**
- **Path**: `mujoco_menagerie/google_barkour_v0/`
- **Model Files**: `barkour_v0.xml`, `barkour_v0_mjx.xml`
- **Scene Files**: `scene.xml`, `scene_barkour.xml` (with obstacle course)
- **DOF**: 12 (3 per leg)
- **Features**:
  - Designed for agile locomotion and parkour
  - Includes obstacle course scene
  - MJX-optimized version
  - URDF also available
- **Special**: Optimized for dynamic movements (jumping, climbing)

### 8. **Google Barkour vB**
- **Path**: `mujoco_menagerie/google_barkour_vb/`
- **Model Files**: `barkour_vb.xml`, `barkour_vb_mjx.xml`
- **Scene Files**: `scene.xml`, `scene_hfield_mjx.xml` (with terrain)
- **DOF**: 12 (3 per leg)
- **Features**:
  - Updated version of Barkour v0
  - Height field terrain support
  - MJX-optimized
- **Special**: Includes terrain generation for robust locomotion

---

## Recommended Training Progression

### Phase 1: Learning the Basics
**Robot**: Unitree Go1  
**Rationale**: 
- Well-documented
- Medium complexity
- Good community support
- Proven to work well with RL

### Phase 2: Intermediate Challenges
**Robot**: Unitree Go2 or ANYmal B  
**Rationale**:
- More complex dynamics
- Test generalization of your approach
- MJX support for faster iteration (Go2)

### Phase 3: Advanced Research
**Robot**: ANYmal C or Barkour vB  
**Rationale**:
- Research-grade robots
- MJX support for large-scale experiments
- Terrain and obstacle support
- Published baselines available

### Phase 4: Industry Applications
**Robot**: Boston Dynamics Spot  
**Rationale**:
- Most realistic dynamics
- Industry-relevant
- Transfer learning potential

---

## Model File Structure

Each quadruped model directory contains:

```
robot_name/
├── robot_name.xml          # Main MuJoCo model file (MJCF)
├── robot_name_mjx.xml      # MJX-optimized version (if available)
├── scene.xml               # Complete scene with robot + environment
├── scene_mjx.xml           # MJX-optimized scene (if available)
├── assets/                 # Mesh files and textures
│   ├── *.obj              # 3D mesh files
│   └── *.png              # Texture files
├── README.md               # Model documentation
├── LICENSE                 # Licensing information
└── CHANGELOG.md            # Version history
```

---

## Key Model Characteristics

### Common Features Across All Models:
- **12 DOF**: 3 joints per leg (abduction/adduction, hip, knee)
- **Actuator Type**: Position-controlled or torque-controlled
- **Foot Contact**: Softened contacts to simulate rubber/compliance
- **Collision Geometries**: Simplified capsules/cylinders for efficiency
- **Free Joint**: Base has 6-DOF free joint for floating base dynamics

### Actuation Specifications:
- **Position Control**: Most models use position-controlled actuators with PD gains
- **Torque Limits**: Typically 20-40 Nm per joint
- **Control Frequency**: Depends on timestep (typically 0.001-0.002s)

### Physics Parameters:
- **Friction**: 0.6 (typical ground friction)
- **Contact Solver**: Elliptic or pyramidal friction cone
- **Impratio**: 100 (high to reduce foot slippage)
- **Damping**: 1-2 Nm/(rad/s) per joint

---

## MJX (MuJoCo XLA) Support

Models with MJX support for accelerated training:
- ✅ Unitree Go2
- ✅ ANYmal C (with training example!)
- ✅ Google Barkour v0
- ✅ Google Barkour vB

**Benefits of MJX**:
- 10-100x faster simulation on GPU/TPU
- Vectorized environments natively
- Ideal for large-scale RL training
- JIT compilation for efficiency

---

## Observation Space Design (Typical)

For locomotion tasks, typical observations include:

### Proprioceptive (48-60 dims):
- Joint positions (12)
- Joint velocities (12)
- Body orientation (quaternion: 4 or euler: 3)
- Body linear velocity (3)
- Body angular velocity (3)
- Previous actions (12) - optional
- Foot contact forces (4) - optional

### Exteroceptive (Optional):
- Height map around robot
- Terrain normals
- Goal direction/velocity

---

## Action Space Design

### Option 1: Position Control (Recommended for Starting)
- **Action Dim**: 12
- **Range**: Joint position limits
- **Advantages**: Stable, easier to learn
- **Implementation**: Actions are target joint positions

### Option 2: Torque Control (Advanced)
- **Action Dim**: 12
- **Range**: Torque limits (-30 to 30 Nm typical)
- **Advantages**: More flexible, energy-aware
- **Implementation**: Actions are joint torques

### Option 3: Hybrid (PD Control)
- **Action Dim**: 12 or 24
- **Range**: Position targets + optional stiffness
- **Advantages**: Balance of stability and flexibility

---

## Next Steps (Sub-task 1.2)

Now that we have identified the models, the next step is to:

1. **Create a test script** to load and visualize Unitree Go1
2. **Verify physics simulation** works correctly
3. **Test basic control** (apply random actions)
4. **Render and save video** to confirm everything works

**Command to proceed**:
```bash
cd /home/hice1/asinha389/scratch/DRL_Project_TRPO
python test_quadruped_load.py  # We'll create this next
```

---

## Useful Resources

### Official Documentation:
- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
- MuJoCo Docs: https://mujoco.readthedocs.io/
- MJX Documentation: https://mujoco.readthedocs.io/en/stable/mjx.html

### Training Examples:
- ANYmal C Colab: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb

### Research Papers:
- Unitree Robots: https://www.unitree.com/
- ANYmal: https://doi.org/10.1080/01691864.2017.1378591
- Barkour: Google Research publications

---

## Summary

✅ **Successfully cloned** mujoco_menagerie repository  
✅ **Identified 8 quadruped models** suitable for locomotion research  
✅ **Analyzed model specifications** and capabilities  
✅ **Recommended progression path**: Go1 → Go2/ANYmal B → ANYmal C/Barkour → Spot  

**Recommended Starting Point**: **Unitree Go1**
- Best balance of complexity and trainability
- Well-documented and widely used
- Good foundation for learning quadruped locomotion

**Ready for Sub-task 1.2**: Create test script to load and visualize the model!

