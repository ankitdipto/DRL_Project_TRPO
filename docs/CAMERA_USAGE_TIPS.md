# 📷 Camera Usage Tips & Best Practices

Quick reference for choosing the right camera mode for your use case.

---

## 🎯 Quick Decision Guide

### Choose **Follow Robot** when:
- ✅ Testing walking/locomotion
- ✅ Robot is moving around
- ✅ You want continuous tracking
- ✅ Doing long-distance movement
- ✅ Default/not sure which to use

### Choose **Fixed View** when:
- ✅ Robot is standing still
- ✅ Testing balance/stability
- ✅ Working near origin
- ✅ Comparing multiple runs from same perspective

### Choose **Side View** when:
- ✅ Analyzing gait patterns
- ✅ Studying leg motion
- ✅ Measuring stride length
- ✅ Checking ground contacts

### Choose **Top View** when:
- ✅ Visualizing path/trajectory
- ✅ Navigation experiments
- ✅ Multi-robot scenarios
- ✅ Checking orientation/heading

---

## 📊 Camera Mode Comparison

```
FOLLOW ROBOT (🎯)
     Camera → [Behind Robot]
     
        📷
         ↘
          → 🤖 →
     
Robot moves → Camera follows
Perfect for walking!


FIXED VIEW (📍)
     Camera → [Fixed Position]
     
        📷
         ↓
        🤖 → → → (robot may leave frame)
     
Camera stays put
Good for nearby behavior


SIDE VIEW (↔️)
     Camera → [Tracks from side]
     
    📷 ← tracking
    ↓
    🤖 →
     
Side profile, follows robot
Great for gait analysis


TOP VIEW (🔽)
     Camera → [Overhead, tracking]
     
         📷
         ↓
         🤖
         
Bird's eye view
Excellent for paths
```

---

## 🎮 Recommended Combinations

### For Walking Experiments
```
Control Mode: 🚶 Walk
Camera Mode:  🎯 Follow Robot
Why: Robot will move forward, camera keeps it in view
```

### For Standing Balance
```
Control Mode: 🧍 Standing
Camera Mode:  📍 Fixed View
Why: Robot stays in place, no need for tracking
```

### For Gait Analysis
```
Control Mode: 🚶 Walk
Camera Mode:  ↔️ Side View
Why: Best view to see leg motion and stride
```

### For Path Visualization
```
Control Mode: 🚶 Walk or 🎲 Random
Camera Mode:  🔽 Top View
Why: See overall movement pattern and direction
```

### For Chaos Testing
```
Control Mode: 🎲 Random
Camera Mode:  🎯 Follow Robot
Why: Random motion → robot goes everywhere, camera tracks it
```

---

## 🔄 When to Switch Cameras

### During a Single Simulation

You can switch cameras while simulation is running! Try this workflow:

1. **Start** with 🎯 **Follow Robot**
   - Get overall behavior view
   
2. **Switch** to ↔️ **Side View**
   - Study leg motion in detail
   
3. **Switch** to 🔽 **Top View**
   - Check path and heading
   
4. **Switch** back to 🎯 **Follow Robot**
   - Continue observing overall behavior

**No need to stop/restart!** Just click the camera button.

---

## 💡 Pro Tips

### Tip 1: Default to Follow Robot
When in doubt, use "Follow Robot" mode. It works well for everything and ensures you never lose sight of the robot.

### Tip 2: Use Side View for Debugging Gaits
If walking looks weird in follow mode, switch to side view to see exactly what the legs are doing.

### Tip 3: Top View for Distance Tracking
Want to see how far the robot traveled? Use top view and watch the path.

### Tip 4: Fixed View for Comparisons
Testing different parameters? Use fixed view so all runs have the same perspective.

### Tip 5: Combine with Control Modes
- **Standing** + **Fixed**: Perfect combo for balance testing
- **Walk** + **Follow**: Perfect combo for locomotion
- **Random** + **Top**: See chaos from above
- **Walk** + **Side**: Best for gait analysis

---

## 🎬 Example Workflows

### Workflow 1: Testing a New Walking Gait

```
1. Start simulation
2. Select "Walk" control mode
3. Select "Follow Robot" camera
4. Watch robot walk forward (camera tracks)
5. If gait looks wrong:
   → Switch to "Side View"
   → Analyze leg motion
   → Stop simulation
   → Adjust gait parameters
6. Restart and repeat
```

### Workflow 2: Measuring Stability

```
1. Start simulation
2. Select "Standing" control mode
3. Select "Fixed View" camera
4. Watch for 30 seconds
5. Check if robot stays in frame
   → If yes: Good stability!
   → If no: Robot is drifting
```

### Workflow 3: Visualizing Learning Progress

```
Training RL agent? Use this sequence:

Early Training (Random policy):
- Control: Random
- Camera: Follow Robot
- Goal: See what random behavior looks like

Mid Training (Learning to walk):
- Control: Trained policy
- Camera: Side View
- Goal: Analyze emerging gait patterns

Late Training (Good walking):
- Control: Trained policy
- Camera: Top View
- Goal: Visualize overall path and efficiency
```

---

## 🎨 Visual Quality Tips

### Best Lighting and Angles

**Follow Robot (azimuth=90, elevation=-20)**
- Good lighting on robot
- Clear view of body and legs
- Natural "gameplay" perspective

**Side View (azimuth=0, elevation=-15)**
- Profile view
- Best for leg visibility
- Good shadow contrast

**Top View (azimuth=90, elevation=-89)**
- No shadows (directly overhead)
- Clear position tracking
- May be harder to judge height

### If Robot is Too Small/Large

Edit `stream_unitree_go1.py`:

```python
# Make robot appear larger (closer camera)
renderer.camera.distance = 2.0  # Default: 2.5

# Make robot appear smaller (farther camera)
renderer.camera.distance = 3.5  # Default: 2.5
```

---

## 🐛 Troubleshooting

### Problem: Camera is shaking/jittery
**Cause**: Robot position updates every frame  
**Solution**: This is normal with 30 FPS. For smoother camera, could add interpolation (future feature)

### Problem: Can't see robot in Top View
**Cause**: Robot may be lying flat on ground  
**Solution**: Switch to Follow or Side view first, or use Standing mode

### Problem: Camera is too close/far
**Cause**: Default distance may not be ideal for your use case  
**Solution**: Edit distance parameter in `update_camera()` function

### Problem: Robot still goes off-screen in Fixed View
**Cause**: Robot walked too far from origin  
**Solution**: That's expected! Use Follow Robot mode instead

---

## 📊 Performance Notes

### Camera Update Cost
- **Computational Cost**: Nearly zero
- **Just sets camera parameters** (lookat, distance, angles)
- **No physics computation** involved
- **No rendering overhead**

### All Cameras Run at Same FPS
- Follow Robot: 30 FPS ✅
- Fixed View: 30 FPS ✅
- Side View: 30 FPS ✅
- Top View: 30 FPS ✅

No performance difference between camera modes!

---

## 🎓 Understanding Camera Coordinates

### MuJoCo Camera System

```python
# Camera position is defined by:
lookat:    Where camera looks at [x, y, z]
distance:  How far from lookat point
azimuth:   Horizontal rotation (0-360°)
elevation: Vertical angle (-89 to 89°)

# Example: Behind robot
lookat = [robot_x, robot_y, robot_z]
distance = 2.5
azimuth = 90    # 90° = behind, 270° = in front
elevation = -20  # Negative = looking down
```

### Coordinate Frame

```
     Y (Forward)
     ↑
     |
     |
     +----→ X (Right)
    /
   ↙
  Z (Up)
```

- **X-axis**: Left/Right (negative = left, positive = right)
- **Y-axis**: Forward/Back (negative = back, positive = forward)
- **Z-axis**: Up/Down (negative = down, positive = up)

---

## 📝 Quick Reference Card

| Scenario | Best Camera | Why |
|----------|-------------|-----|
| Walking forward | 🎯 Follow | Tracks movement |
| Standing balance | 📍 Fixed | Stationary target |
| Gait analysis | ↔️ Side | See legs clearly |
| Path planning | 🔽 Top | Overview of trajectory |
| Random motion | 🎯 Follow | Keeps robot in view |
| Comparing runs | 📍 Fixed | Same perspective |
| Leg debugging | ↔️ Side | Profile view |
| Navigation | 🔽 Top | Bird's eye view |

---

## 🚀 Advanced: Customizing Cameras

Want to create your own camera view? Edit `update_camera()` in `stream_unitree_go1.py`:

```python
def update_camera(renderer, data, mode="follow"):
    robot_pos = data.qpos[0:3].copy()
    
    if mode == "my_custom_view":
        # Custom configuration
        renderer.camera.lookat = robot_pos + np.array([1.0, 1.0, 0.5])
        renderer.camera.distance = 4.0
        renderer.camera.azimuth = 45   # Diagonal view
        renderer.camera.elevation = -30
```

Then add button to HTML template and route in Flask!

---

## ✅ Summary

**Default Choice**: 🎯 **Follow Robot** - Works for 90% of use cases

**For Analysis**: ↔️ **Side View** - When you need to study details

**For Overview**: 🔽 **Top View** - When you need the big picture

**For Stationary**: 📍 **Fixed View** - When robot doesn't move much

**Pro Move**: Switch cameras during simulation to see different perspectives!

---

Enjoy your camera-tracking simulation! 📹🤖

