#!/usr/bin/env python3
"""
This script runs the Duckietown simulation and publishes camera images to ROS
"""
import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import argparse
import sys
import os
import gym
import numpy as np
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from duckietown_msgs.msg import WheelsCmdStamped

class DuckietownROSBridge:
    def __init__(self, args):
        # Initialize ROS
        rospy.init_node('duckietown_simulator', anonymous=True)
        
        # Get vehicle name from ROS parameter or environment
        self.veh_name = os.environ['VEHICLE_NAME']
        
        self.env = DuckietownEnv(
             seed=args.seed,
             map_name=args.map_name,
             draw_curve=args.draw_curve,
             draw_bbox=args.draw_bbox,
             domain_rand=args.domain_rand,
             frame_skip=args.frame_skip,
             distortion=args.distortion,
             camera_rand=args.camera_rand,
             dynamics_rand=args.dynamics_rand,
        )
            
        self.env.reset()
        self.env.render()

        # Create publishers
        self.image_pub = rospy.Publisher(
            f'/123/camera_node/image/raw', 
            Image, 
            queue_size=1
        )
        self.compressed_pub = rospy.Publisher(
            f'/123/camera_node/image/compressed', 
            CompressedImage, 
            queue_size=1
        )
        
        wheels_cmd_topic = f"/123/wheels_driver_node/wheels_cmd"

        self.wheels_sub = rospy.Subscriber(
            wheels_cmd_topic,
            WheelsCmdStamped,
            self.wheels_cmd_callback,
            queue_size=1
        )

        self.current_action = np.array([0.0, 0.0])
        self.new_command_received = False
        self.first_image_sent = False

        self.bridge = CvBridge()
        
    def wheels_cmd_callback(self, msg: WheelsCmdStamped):
        rospy.loginfo(f"Received wheel command: vel_left={msg.vel_left:.3f}, vel_right={msg.vel_right:.3f}")

        effective_wheel_distance = 0.102
        linear_velocity = (msg.vel_left + msg.vel_right) / 2.0
        angular_velocity = (msg.vel_right - msg.vel_left) / effective_wheel_distance 

        self.current_action = np.array([linear_velocity, angular_velocity])
        self.new_command_received = True
        
        rospy.loginfo(f"Converted action: linear={self.current_action[0]:.3f}, angular={self.current_action[1]:.3f}")
        
    def publish_image(self, obs):
        """Publish camera observation to ROS topics"""
        try:
            # Convert numpy array to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(obs, encoding="rgb8")
            ros_image.header.stamp = rospy.Time.now()
            ros_image.header.frame_id = f"123/camera_optical_frame"
            
            # Publish raw image
            self.image_pub.publish(ros_image)
            
            # Create and publish compressed image
            compressed_msg = CompressedImage()
            compressed_msg.header = ros_image.header
            compressed_msg.format = "jpeg"
            
            # Convert to JPEG
            _, jpeg_data = cv2.imencode('.jpg', cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
            compressed_msg.data = jpeg_data.tobytes()
            
            self.compressed_pub.publish(compressed_msg)
            
        except Exception as e:
            rospy.logerr(f"Error publishing image: {e}")
    
    def update(self, dt):
        """Update function called at every frame"""
        if rospy.is_shutdown():
            self.env.close()
            sys.exit(0)

        if not self.first_image_sent:
            rospy.loginfo("Sending initial image...")
            obs = self.env.render(mode='rgb_array')
            self.publish_image(obs)
            self.first_image_sent = True
            rospy.loginfo("Waiting for first wheel command...")
            return
        
        # Only step if we received a new command
        if self.new_command_received:
            rospy.loginfo(f"Stepping simulation with action: linear={self.current_action[0]:.3f}, angular={self.current_action[1]:.3f}")
            
            # Step the simulation
            obs, reward, done, info = self.env.step(self.current_action)
            
            print(f"step_count = {self.env.unwrapped.step_count}, reward={reward:.3f}")
            
            if done:
                print("Episode done! Resetting...")
                self.env.reset()
                obs = self.env.render(mode='rgb_array')
            
            self.env.render()
            
            # Publish new camera image
            self.publish_image(obs)
            
            # Reset flag - wait for next command
            self.new_command_received = False
            rospy.loginfo("Waiting for next wheel command...")
    
    def run(self):
        """Main run loop"""
        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)
        
        try:
            pyglet.app.run()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down...")
        finally:
            self.env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Duckietown-udem1-v0")
    parser.add_argument("--map-name", default="udem1")
    parser.add_argument("--distortion", default=False, action="store_true")
    parser.add_argument("--camera_rand", default=False, action="store_true")
    parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
    parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
    parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
    parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
    parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    
    args = parser.parse_args()
    
    try:
        bridge = DuckietownROSBridge(args)
        bridge.run()
    except rospy.ROSInterruptException:
        pass
