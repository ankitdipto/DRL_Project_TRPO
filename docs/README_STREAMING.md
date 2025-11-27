# 🎥 Realtime Web Streaming - Quick Start Guide

View your Unitree Go1 simulation live in your browser!

---

## 🚀 One-Line Start

```bash
./run_streaming.sh
```

Then open: **http://localhost:5000**

---

## 📋 Manual Start

```bash
# 1. Install Flask (if needed)
pip install flask

# 2. Run the server
python stream_unitree_go1.py

# 3. Open browser to http://localhost:5000
```

---

## 🎮 What You Can Do

### Control Modes

| Mode | Description | Best For |
|------|-------------|----------|
| 🧍 **Standing** | Stable upright pose at 0.265m | Testing stability |
| 💤 **Zero** | No control, robot falls | Passive dynamics |
| 🎲 **Random** | Random joint commands | Action space exploration |
| 🚶 **Walk** | CPG-based walking gait | Locomotion experiments |

### Camera Views

| View | Description | Best For |
|------|-------------|----------|
| 🎯 **Follow Robot** | Camera tracks robot position | Walking/locomotion (default) |
| 📍 **Fixed View** | Static camera at origin | Standing/nearby behavior |
| ↔️ **Side View** | Follows from the side | Analyzing gait patterns |
| 🔽 **Top View** | Bird's eye view | Path planning, position tracking |

### Interface Controls

- **▶️ Start** - Begin simulation
- **⏹️ Stop** - Pause simulation  
- **Robot Mode Buttons** - Switch control modes on-the-fly
- **Camera Buttons** - Switch camera views instantly
- **Auto Status Updates** - See current state in real-time

---

## 🌐 Access from Another Computer

```bash
# 1. Find your server's IP address
hostname -I

# 2. Open browser to:
http://YOUR_SERVER_IP:5000
```

---

## 📊 Features

✅ **Live Video Stream** @ 30 FPS  
✅ **Interactive Controls** - Start/stop, change modes  
✅ **Following Camera** - Tracks robot automatically (no more going off-screen!)  
✅ **Multiple Camera Views** - Follow, Fixed, Side, Top  
✅ **No Video Files** - Everything in real-time  
✅ **Thread-Safe** - Separate physics and rendering threads  
✅ **Low Latency** - ~50-100ms end-to-end  
✅ **Beautiful UI** - Clean, responsive web interface  

---

## 🎯 Comparison: Streaming vs Recording

| Feature | `test_unitree_go1_scene.py` | `stream_unitree_go1.py` |
|---------|---------------------------|------------------------|
| **View Mode** | Offline (after recording) | Real-time (live) |
| **Output** | MP4 video files | Web browser |
| **Interactivity** | None | Start/stop, mode switching |
| **Storage** | Saves videos to disk | No files generated |
| **Use Case** | Documentation, sharing | Development, debugging |
| **Feedback Loop** | Slow (record → watch) | Fast (immediate) |

---

## 💡 Use Cases

### 1. **Development & Debugging**
Watch robot behavior immediately as you test control algorithms

### 2. **Remote Monitoring**
Run simulation on server, monitor from your laptop anywhere

### 3. **Demonstrations**
Show robot to colleagues/advisors without sending video files

### 4. **RL Training Monitoring**
Integrate with training loop to watch policy evolution live

### 5. **Quick Experiments**
Test different control modes without generating video files

---

## 🔧 Troubleshooting

### Port Already in Use?
```bash
# Kill existing process
kill -9 $(lsof -ti:5000)

# Or use different port (edit line 506 in stream_unitree_go1.py)
app.run(host='0.0.0.0', port=5001, ...)
```

### Can't See Video?
```bash
# Check MuJoCo rendering backend
export MUJOCO_GL=egl
python stream_unitree_go1.py
```

### Simulation Too Slow?
Edit `stream_unitree_go1.py` line 262:
```python
fps = 15  # Reduce from 30
```

---

## 📁 Files Created

1. **stream_unitree_go1.py** - Main Flask server (506 lines)
2. **run_streaming.sh** - Quick start script
3. **STREAMING_GUIDE.md** - Detailed documentation
4. **README_STREAMING.md** - This quick start guide

---

## 🎓 Architecture Overview

```
┌─────────────────┐
│  Web Browser    │  ← You watch here (http://localhost:5000)
└────────┬────────┘
         │ HTTP/MJPEG
┌────────▼────────┐
│  Flask Server   │  ← Handles requests, serves HTML
└────────┬────────┘
         │ Threading
┌────────▼────────┐
│ Simulation Loop │  ← MuJoCo physics @ 500Hz
│  + Renderer     │  ← Renders frames @ 30 FPS
└─────────────────┘
```

---

## 🚀 Next Steps

Want to enhance the streaming system? Ideas:

1. **Add Metrics Display** - Show base height, velocity, orientation
2. ~~**Multiple Camera Views**~~ - ✅ DONE! Front, side, top views with robot tracking
3. **Record from Browser** - Add "Save Video" button
4. **Joint Control Sliders** - Manual joint position control
5. **Better Walking Gait** - Improve CPG implementation
6. **Free Camera Control** - Mouse drag to rotate/zoom

See `STREAMING_GUIDE.md` for implementation details!

---

## 📚 Learn More

- **STREAMING_GUIDE.md** - Comprehensive documentation
- **GO1_SIMULATION_SUMMARY.md** - Robot specifications
- **test_unitree_go1_scene.py** - Offline recording version

---

**Enjoy watching your robot live! 🤖**

Made with ❤️ for the DRL_Project_TRPO

