#!/bin/bash
set -e
source /environment.sh
dt-launchfile-init

echo "🚗 Starting lane follower..."

dt-exec rosrun utils lane_follower.py
dt-exec rosrun utils wheel_controller.py

echo "🚗 Lane follower started!"

dt-launchfile-join
