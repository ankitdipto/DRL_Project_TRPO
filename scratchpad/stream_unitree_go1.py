#!/usr/bin/env python3
"""
Realtime web streaming version of Unitree Go1 simulation using Flask.
View the simulation live in your browser at http://localhost:5000
"""

import os
os.environ['MUJOCO_GL'] = 'egl'  # Use EGL for headless rendering

import numpy as np
import mujoco
from flask import Flask, Response, render_template_string, request
import cv2
import threading
import time

# Path to the Unitree Go1 scene
MENAGERIE_PATH = "/home/asinha389/Documents/DRL_Project_TRPO/mujoco_menagerie"
GO1_SCENE_PATH = os.path.join(MENAGERIE_PATH, "unitree_go1/scene.xml")

# Global variables for frame sharing
current_frame = None
frame_lock = threading.Lock()
simulation_running = False
simulation_thread = None
control_mode = "standing"  # Default control mode
camera_mode = "follow"  # Default camera mode: "fixed", "follow", "side", "top"

# Flask app
app = Flask(__name__)

# HTML template with controls
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Unitree Go1 - Live Simulation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .video-container {
            text-align: center;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            border: 2px solid #333;
            border-radius: 5px;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            border: none;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .btn-start {
            background-color: #4CAF50;
            color: white;
        }
        .btn-start:hover {
            background-color: #45a049;
        }
        .btn-stop {
            background-color: #f44336;
            color: white;
        }
        .btn-stop:hover {
            background-color: #da190b;
        }
        .btn-mode {
            background-color: #2196F3;
            color: white;
        }
        .btn-mode:hover {
            background-color: #0b7dda;
        }
        .btn-mode.active {
            background-color: #0b7dda;
            font-weight: bold;
        }
        .btn-camera {
            background-color: #FF9800;
            color: white;
        }
        .btn-camera:hover {
            background-color: #F57C00;
        }
        .btn-camera.active {
            background-color: #F57C00;
            font-weight: bold;
        }
        .control-section {
            margin: 20px 0;
        }
        .control-section h4 {
            text-align: center;
            color: #555;
            margin: 10px 0;
        }
        .info {
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .info h3 {
            margin-top: 0;
            color: #1976D2;
        }
        .info ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .status {
            text-align: center;
            font-size: 18px;
            margin: 15px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .status.running {
            background-color: #c8e6c9;
            color: #2e7d32;
        }
        .status.stopped {
            background-color: #ffcdd2;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Unitree Go1 - Live Simulation Stream</h1>
        
        <div class="status {{ 'running' if status else 'stopped' }}" id="status">
            {% if status %}
                ✅ Simulation Running - {{ mode }} mode
            {% else %}
                ⏸️ Simulation Stopped
            {% endif %}
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Live Simulation Feed" id="videoFeed">
        </div>
        
        <div class="control-section">
            <h4>⚙️ Simulation Control</h4>
            <div class="controls">
                <button class="btn-start" onclick="startSimulation()">▶️ Start</button>
                <button class="btn-stop" onclick="stopSimulation()">⏹️ Stop</button>
            </div>
        </div>
        
        <div class="control-section">
            <h4>🤖 Robot Control Mode</h4>
            <div class="controls">
                <button class="btn-mode {{ 'active' if mode == 'standing' else '' }}" onclick="setMode('standing')">
                    🧍 Standing
                </button>
                <button class="btn-mode {{ 'active' if mode == 'zero' else '' }}" onclick="setMode('zero')">
                    💤 Zero Control
                </button>
                <button class="btn-mode {{ 'active' if mode == 'random' else '' }}" onclick="setMode('random')">
                    🎲 Random
                </button>
                <button class="btn-mode {{ 'active' if mode == 'walk' else '' }}" onclick="setMode('walk')">
                    🚶 Walk (Experimental)
                </button>
            </div>
        </div>
        
        <div class="control-section">
            <h4>📷 Camera View</h4>
            <div class="controls">
                <button class="btn-camera {{ 'active' if camera == 'follow' else '' }}" onclick="setCamera('follow')">
                    🎯 Follow Robot
                </button>
                <button class="btn-camera {{ 'active' if camera == 'fixed' else '' }}" onclick="setCamera('fixed')">
                    📍 Fixed View
                </button>
                <button class="btn-camera {{ 'active' if camera == 'side' else '' }}" onclick="setCamera('side')">
                    ↔️ Side View
                </button>
                <button class="btn-camera {{ 'active' if camera == 'top' else '' }}" onclick="setCamera('top')">
                    🔽 Top View
                </button>
            </div>
        </div>
        
        <div class="info">
            <h3>ℹ️ Information</h3>
            <p><strong>Robot Control Modes:</strong></p>
            <ul>
                <li><strong>Standing:</strong> Maintains upright standing pose at ~0.265m height</li>
                <li><strong>Zero Control:</strong> No actuation - robot falls and lies on ground</li>
                <li><strong>Random:</strong> Random joint commands - chaotic motion</li>
                <li><strong>Walk:</strong> Experimental walking gait pattern (CPG-based)</li>
            </ul>
            <p><strong>Camera Views:</strong></p>
            <ul>
                <li><strong>Follow Robot:</strong> Camera tracks robot position (recommended for walking)</li>
                <li><strong>Fixed View:</strong> Static camera at origin</li>
                <li><strong>Side View:</strong> Follows robot from the side</li>
                <li><strong>Top View:</strong> Bird's eye view tracking robot</li>
            </ul>
            <p><strong>Robot Specs:</strong> 12 actuators, 18 DOF, Position control @ 500Hz</p>
        </div>
    </div>
    
    <script>
        function startSimulation() {
            fetch('/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    location.reload();
                });
        }
        
        function stopSimulation() {
            fetch('/stop', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    location.reload();
                });
        }
        
        function setMode(mode) {
            fetch('/set_mode/' + mode, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    location.reload();
                });
        }
        
        function setCamera(camera) {
            fetch('/set_camera/' + camera, {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    location.reload();
                });
        }
        
        // Auto-refresh status every 2 seconds
        setInterval(() => {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('status');
                    if (data.running) {
                        statusDiv.className = 'status running';
                        statusDiv.innerHTML = '✅ Simulation Running - ' + data.mode + ' mode | 📷 ' + data.camera + ' camera';
                    } else {
                        statusDiv.className = 'status stopped';
                        statusDiv.innerHTML = '⏸️ Simulation Stopped';
                    }
                });
        }, 2000);
    </script>
</body>
</html>
"""


def update_camera(camera, data, mode="follow"):
    """
    Update camera position based on mode.
    
    Args:
        camera: MuJoCo camera object (mujoco.MjvCamera)
        data: MuJoCo data
        mode: Camera mode - "follow", "fixed", "side", "top"
    """
    # Get robot base position (center of the robot)
    robot_pos = data.qpos[0:3].copy()  # [x, y, z]
    
    if mode == "follow":
        # Follow camera: tracks robot from behind and above
        lookat_offset = np.array([0.5, 0.0, 0.0])  # Look slightly ahead
        
        camera.lookat[:] = robot_pos + lookat_offset
        camera.distance = 2.5
        camera.azimuth = 90  # View from behind
        camera.elevation = -20  # Slight downward angle
        
    elif mode == "side":
        # Side view: follows robot from the side
        camera.lookat[:] = robot_pos
        camera.distance = 2.5
        camera.azimuth = 0  # View from side
        camera.elevation = -15
        
    elif mode == "top":
        # Top-down view: bird's eye view
        camera.lookat[:] = robot_pos
        camera.distance = 3.0
        camera.azimuth = 90
        camera.elevation = -89  # Almost straight down
        
    elif mode == "fixed":
        # Fixed view: static camera at origin
        camera.lookat[:] = np.array([0.0, 0.0, 0.3])
        camera.distance = 3.0
        camera.azimuth = 90
        camera.elevation = -20


def get_standing_pose():
    """Standing pose for Go1."""
    return np.array([
        0.0, 0.9, -1.8,  # Front right leg (FR)
        0.0, 0.9, -1.8,  # Front left leg (FL)
        0.0, 0.9, -1.8,  # Rear right leg (RR)
        0.0, 0.9, -1.8,  # Rear left leg (RL)
    ])


def get_walking_action(t, model):
    """
    Generate a simple walking gait using CPG (Central Pattern Generator).
    This is a basic sinusoidal pattern - can be improved significantly.
    """
    frequency = 1.0  # Hz
    phase_offset = np.pi / 2  # 90 degrees between legs
    
    # Standing pose as baseline
    standing = get_standing_pose()
    
    # Amplitude of motion
    hip_amplitude = 0.3
    knee_amplitude = 0.4
    
    # Phase for each leg (trot gait: FR-RL together, FL-RR together)
    phases = np.array([
        0,              # FR
        np.pi,          # FL (opposite to FR)
        np.pi,          # RR (opposite to FR)
        0,              # RL (same as FR)
    ])
    
    action = standing.copy()
    
    for leg in range(4):
        leg_phase = 2 * np.pi * frequency * t + phases[leg]
        
        # Hip joint (thigh)
        action[leg * 3 + 1] = standing[leg * 3 + 1] + hip_amplitude * np.sin(leg_phase)
        
        # Knee joint (calf) - needs to flex more during swing
        action[leg * 3 + 2] = standing[leg * 3 + 2] + knee_amplitude * np.sin(leg_phase)
    
    # Clip to joint limits
    action = np.clip(action, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    
    return action


def simulation_loop():
    """Main simulation loop that runs in a separate thread."""
    global current_frame, simulation_running, control_mode
    
    try:
        # Load model
        print(f"Loading Unitree Go1 scene from: {GO1_SCENE_PATH}")
        model = mujoco.MjModel.from_xml_path(GO1_SCENE_PATH)  # pyright: ignore[reportAttributeAccessIssue]
        data = mujoco.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Reset simulation
        mujoco.mj_resetData(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Set initial standing pose if needed
        if control_mode == "standing" or control_mode == "walk":
            standing_pose = get_standing_pose()
            data.qpos[7:19] = standing_pose
        
        # Forward kinematics
        mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Create renderer
        renderer = mujoco.Renderer(model, height=480, width=640)
        
        # Create camera
        camera = mujoco.MjvCamera()  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mjv_defaultFreeCamera(model, camera)  # pyright: ignore[reportAttributeAccessIssue]
        
        # Simulation parameters
        fps = 30
        frame_duration = 1.0 / fps
        steps_per_frame = int(frame_duration / model.opt.timestep)
        
        print(f"Simulation started - Control mode: {control_mode}")
        print(f"FPS: {fps}, Steps per frame: {steps_per_frame}")
        
        sim_time = 0.0
        
        while simulation_running:
            # Determine control action based on mode
            if control_mode == "zero":
                action = np.zeros(model.nu)
            elif control_mode == "random":
                action = np.random.uniform(
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1]
                )
            elif control_mode == "standing":
                action = get_standing_pose()
            elif control_mode == "walk":
                action = get_walking_action(sim_time, model)
            else:
                action = np.zeros(model.nu)
            
            # Apply control
            data.ctrl[:] = action
            
            # Step simulation
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)  # pyright: ignore[reportAttributeAccessIssue]
            
            sim_time += frame_duration
            
            # Update camera position
            update_camera(camera, data, camera_mode)
            
            # Render frame
            renderer.update_scene(data, camera=camera)
            pixels = renderer.render()
            
            # Update global frame
            with frame_lock:
                current_frame = pixels.copy()
            
            # Maintain frame rate
            time.sleep(frame_duration)
        
        print("Simulation stopped")
        
    except Exception as e:
        print(f"Error in simulation loop: {e}")
        simulation_running = False


def generate_frames():
    """Generator function to yield frames for streaming."""
    global current_frame
    
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                # Create a blank frame if no simulation is running
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No simulation running", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ret:
            continue
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE, 
                                 status=simulation_running, 
                                 mode=control_mode,
                                 camera=camera_mode)


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start', methods=['POST'])
def start_simulation():
    """Start the simulation."""
    global simulation_running, simulation_thread
    
    if not simulation_running:
        simulation_running = True
        simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
        simulation_thread.start()
        return {'status': 'started', 'mode': control_mode}
    return {'status': 'already_running', 'mode': control_mode}


@app.route('/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation."""
    global simulation_running
    
    if simulation_running:
        simulation_running = False
        return {'status': 'stopped'}
    return {'status': 'already_stopped'}


@app.route('/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    """Change control mode."""
    global control_mode, simulation_running
    
    valid_modes = ['standing', 'zero', 'random', 'walk']
    if mode in valid_modes:
        was_running = simulation_running
        
        # Stop simulation if running
        if simulation_running:
            simulation_running = False
            time.sleep(0.5)  # Wait for thread to finish
        
        # Update mode
        control_mode = mode
        
        # Restart if it was running
        if was_running:
            start_simulation()
        
        return {'status': 'mode_changed', 'mode': mode}
    return {'status': 'invalid_mode', 'mode': control_mode}


@app.route('/set_camera/<mode>', methods=['POST'])
def set_camera(mode):
    """Change camera mode."""
    global camera_mode
    
    valid_modes = ['follow', 'fixed', 'side', 'top']
    if mode in valid_modes:
        camera_mode = mode
        return {'status': 'camera_changed', 'camera': mode}
    return {'status': 'invalid_camera', 'camera': camera_mode}


@app.route('/status')
def get_status():
    """Get current simulation status."""
    return {
        'running': simulation_running,
        'mode': control_mode,
        'camera': camera_mode
    }


if __name__ == "__main__":
    print("=" * 70)
    print("UNITREE GO1 - REALTIME WEB STREAMING")
    print("=" * 70)
    print()
    print(f"Model path: {GO1_SCENE_PATH}")
    print()
    print("Starting Flask server...")
    print()
    print("=" * 70)
    print("🌐 Open your browser and go to:")
    print()
    print("    http://localhost:5000")
    print()
    print("=" * 70)
    print()
    print("Controls:")
    print("  - Click 'Start' to begin simulation")
    print("  - Click 'Stop' to pause simulation")
    print("  - Select control mode: Standing, Zero, Random, or Walk")
    print("  - Select camera view: Follow, Fixed, Side, or Top")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

