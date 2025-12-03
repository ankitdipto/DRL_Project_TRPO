# Unitree Go1 - Realtime Web Streaming Guide

**Created**: November 21, 2024  
**Purpose**: View Go1 simulation live in your web browser

---

## 🚀 Quick Start

### 1. Install Flask (if not already installed)

```bash
pip install flask
```

### 2. Run the Streaming Server

```bash
cd /home/hice1/asinha389/scratch/DRL_Project_TRPO
python stream_unitree_go1.py
```

### 3. Open Your Browser

Navigate to:
```
http://localhost:5000
```

Or if accessing from another machine on the same network:
```
http://YOUR_SERVER_IP:5000
```

---

## 🎮 Controls

### Start/Stop
- **▶️ Start**: Begin the simulation
- **⏹️ Stop**: Pause the simulation

### Control Modes

#### 🧍 Standing Mode (Default)
- Robot maintains stable standing pose
- Height: ~0.265m
- Minimal drift
- **Best for**: Testing stability, understanding robot posture

#### 💤 Zero Control
- No actuation applied
- Robot falls and collapses onto ground
- **Best for**: Understanding passive dynamics

#### 🎲 Random Mode
- Random joint commands
- Chaotic motion with ground interaction
- **Best for**: Stress testing, understanding action space

#### 🚶 Walk Mode (Experimental)
- Simple CPG-based walking gait
- Sinusoidal joint patterns
- Trot gait: diagonal legs move together
- **Best for**: Initial locomotion experiments
- **Note**: Basic implementation, can be improved significantly

### 📷 Camera Views

#### 🎯 Follow Robot (Default, Recommended)
- Camera automatically tracks robot's position
- Views from behind and slightly above
- **Perfect for walking** - robot never leaves frame!
- Updates every frame based on robot's x,y,z position

#### 📍 Fixed View
- Static camera positioned at origin
- Good for nearby behavior (standing, small movements)
- Robot may leave frame if it walks too far

#### ↔️ Side View
- Follows robot from the side
- Excellent for analyzing gait patterns
- Shows leg motion clearly

#### 🔽 Top View
- Bird's eye view tracking robot
- Great for path visualization
- Shows overall motion pattern

**Pro Tip**: Use "Follow Robot" mode when testing walking gaits to keep the robot in view at all times!

---

## 🔧 Technical Details

### Architecture

```
Flask Server (Port 5000)
    ├── Main Thread: HTTP request handling
    └── Simulation Thread: MuJoCo physics loop
            ├── Load Go1 model
            ├── Generate actions based on mode
            ├── Step physics @ 500Hz
            └── Render frames @ 30 FPS
```

### Frame Generation
- **Simulation Rate**: 500 Hz (model.opt.timestep = 0.002s)
- **Rendering Rate**: 30 FPS
- **Resolution**: 640×480
- **Encoding**: JPEG streaming via multipart/x-mixed-replace

### Threading Model
- **Main Thread**: Flask HTTP server
- **Simulation Thread**: Physics simulation and rendering
- **Frame Lock**: Thread-safe frame sharing via mutex

---

## 📊 Performance Considerations

### Server Performance
- **CPU Usage**: Moderate (physics + rendering)
- **Memory**: ~500MB for model + renderer
- **Network**: Low bandwidth (JPEG compressed stream)

### Latency
- **Physics Step**: ~0.002s per step
- **Rendering**: ~0.033s per frame (30 FPS)
- **Network**: Depends on connection
- **Total**: ~50-100ms end-to-end latency

### Optimization Tips
1. **Reduce FPS** if server is slow:
   ```python
   fps = 15  # Line 265 in stream_unitree_go1.py
   ```

2. **Lower resolution** for faster rendering:
   ```python
   renderer = mujoco.Renderer(model, height=360, width=480)
   ```

3. **Increase JPEG quality** for better image (more bandwidth):
   ```python
   cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
   ```

---

## 🐛 Troubleshooting

### Issue: "Address already in use"
**Solution**: Kill existing process or use different port
```bash
# Find process using port 5000
lsof -ti:5000

# Kill process
kill -9 $(lsof -ti:5000)

# Or use different port
app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
```

### Issue: Black screen / No frames
**Solution**: Check MuJoCo rendering backend
```bash
# Verify EGL is working
echo $MUJOCO_GL

# Try alternative backend (osmesa)
export MUJOCO_GL=osmesa
python stream_unitree_go1.py
```

### Issue: Simulation runs too slow
**Solution**: Reduce physics steps per frame
```python
# Line 269 - reduce steps_per_frame
steps_per_frame = int(frame_duration / model.opt.timestep) // 2
```

### Issue: Can't access from remote machine
**Solution**: 
1. Check firewall allows port 5000
2. Use `0.0.0.0` as host (already set)
3. Find server IP: `hostname -I`

---

## 🎯 Use Cases

### 1. Development & Debugging
- View robot behavior in real-time
- Quickly iterate on control algorithms
- Debug physics issues visually

### 2. Demonstrations
- Show robot behavior to others
- Present results without video files
- Interactive demonstrations with mode switching

### 3. RL Training Monitoring
- Watch policy during training
- Verify reward function behavior
- Identify failure modes early

### 4. Remote Experiments
- Run simulation on server
- Monitor from laptop/desktop
- No need for X11 forwarding

---

## 🔮 Future Enhancements

### Possible Improvements:

1. **Multiple Camera Views**
   - Front, side, top views
   - Follow camera that tracks robot

2. **Real-time Metrics Display**
   - Base height, velocity, orientation
   - Joint positions and velocities
   - Reward (when integrated with RL)

3. **Interactive Controls**
   - Sliders for individual joints
   - Target velocity/direction input
   - Manual waypoint setting

4. **Recording from Browser**
   - Save video button
   - Screenshot capture
   - GIF generation

5. **Better Walking Gait**
   - Implement proper CPG
   - Add gait selection (trot, pace, gallop)
   - Speed control

6. **Multi-robot Support**
   - Switch between Go1, A1, Go2, etc.
   - Compare different models

---

## 📝 Code Structure

### Main Components

```python
# Global state
current_frame         # Shared frame buffer
frame_lock           # Thread synchronization
simulation_running   # Control flag
control_mode         # Current control mode

# Core functions
simulation_loop()    # Physics simulation (separate thread)
generate_frames()    # Frame generator for streaming
get_standing_pose()  # Standing configuration
get_walking_action() # Walking gait generator

# Flask routes
/                    # Main page (HTML interface)
/video_feed         # Video stream (MJPEG)
/start              # Start simulation (POST)
/stop               # Stop simulation (POST)
/set_mode/<mode>    # Change control mode (POST)
/status             # Get current status (GET)
```

---

## 🔗 Integration with Training

### How to Stream During RL Training

You can integrate this streaming system with your TRPO training loop:

```python
# In main.py, add streaming capability
from stream_unitree_go1 import start_streaming_server

# In train_trpo function
streaming_thread = threading.Thread(
    target=start_streaming_server, 
    args=(eval_env,), 
    daemon=True
)
streaming_thread.start()

# Then you can watch evaluation episodes live!
```

---

## 📚 Related Files

- **test_unitree_go1_scene.py** - Offline video recording version
- **GO1_SIMULATION_SUMMARY.md** - Simulation specifications
- **main.py** - TRPO training script (can be integrated)

---

## 🎓 Learning Resources

### Flask Streaming
- Flask Docs: https://flask.palletsprojects.com/
- MJPEG Streaming: https://en.wikipedia.org/wiki/Motion_JPEG

### MuJoCo Rendering
- MuJoCo Docs: https://mujoco.readthedocs.io/en/stable/python.html#rendering
- Renderer API: https://mujoco.readthedocs.io/en/stable/python.html#mujoco.Renderer

### CPG for Locomotion
- Central Pattern Generators: https://en.wikipedia.org/wiki/Central_pattern_generator
- Quadruped Gaits: https://en.wikipedia.org/wiki/Gait#Quadrupeds

---

## ✅ Summary

**Created**: `stream_unitree_go1.py` - Flask-based realtime streaming server  
**Features**:
- ✅ Live video streaming @ 30 FPS
- ✅ 4 control modes (Standing, Zero, Random, Walk)
- ✅ Web-based interface with controls
- ✅ Start/stop simulation on demand
- ✅ Real-time mode switching
- ✅ Status monitoring
- ✅ Thread-safe frame sharing
- ✅ No video files needed

**Access**: http://localhost:5000

---

**Enjoy watching your robot in action! 🤖**

