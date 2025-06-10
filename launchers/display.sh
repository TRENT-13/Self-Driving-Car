#!/bin/bash
set -e
source /environment.sh
dt-launchfile-init

echo "Starting ROS Master..."

roscore &

echo "Starting VNC display for Duckietown..."

# Kill any existing processes
pkill -f Xvfb || true
pkill -f x11vnc || true
pkill -f websockify || true
pkill -f fluxbox || true

# Start virtual display
echo "Starting virtual display..."
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &
sleep 3
export DISPLAY=:99

# Test display
if ! xdpyinfo -display :99 > /dev/null 2>&1; then
    echo "Display failed to start"
    exit 1
fi
echo "Display started: $DISPLAY"

# Start window manager
echo "Starting window manager..."
fluxbox -display :99 > /dev/null 2>&1 &
sleep 2

# Start VNC server
echo "Starting VNC server..."
x11vnc -display :99 -nopw -listen 0.0.0.0 -forever -shared -rfbport 5900 > /dev/null 2>&1 &
sleep 2

# Verify VNC is running
if ! netstat -tlnp | grep -q ":5900"; then
    echo "VNC server failed to start"
    exit 1
fi
echo "VNC server started on port 5900"

# Install and start websockify
echo "Setting up web interface..."
pip3 install websockify || echo "websockify installation check complete"

# Start websockify
websockify --web /usr/share/novnc 0.0.0.0:6080 localhost:5900 > /dev/null 2>&1 &
sleep 3

echo "Web server started on port 6080"

# Set simulation environment
export DISPLAY=:99
export PYGLET_HEADLESS=0
export SDL_VIDEODRIVER=x11

echo ""
echo "Web VNC: http://localhost:6080/vnc.html"
echo "Direct VNC: localhost:5900"
echo ""

# Test with a simple GUI application first
echo "Testing display with xclock..."
xclock -display :99 &
XCLOCK_PID=$!
sleep 2

echo "You should see xclock in VNC. Press Enter to continue to simulation..."
read

# Kill xclock
kill $XCLOCK_PID 2>/dev/null || true

echo "Starting Duckietown simulator with keyboard controls..."

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up processes..."
    kill $SIMULATOR_PID 2>/dev/null || true
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Start keyboard controls in background
echo "Starting keyboard controls..."
DISPLAY=:99 dt-exec rosrun simulator manual_control.py &
SIMULATOR_PID=$!

# Wait for both processes
echo "Process started."
wait $SIMULATOR_PID

dt-launchfile-join
