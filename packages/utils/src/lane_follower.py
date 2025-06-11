#!/usr/bin/env python3
import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from collections import deque
import time
# import torch

# Tunable parameters
BASE_SPEED = 0.25  # Base forward speed
CURVE_SPEED = 0.20  # Reduced speed for curves
P_GAIN = 0.4       # Proportional gain
D_GAIN = 0.2       # Derivative gain - helps with curves
MAX_STEER = 0.5    # Maximum steering adjustment
SMOOTHING_STRAIGHT = 3  # Smoothing for straight roads
SMOOTHING_CURVE = 2     # Less smoothing for curves

# Alignment parameters
ALIGNMENT_P_GAIN = 0.1    # Proportional gain for red alignment
ALIGNMENT_MAX_SPEED = 0.1  # Maximum speed during alignment
MIN_RED_WIDTH = 50        # Minimum width of red mask to attempt alignment
MODEL_PATH = 'yolov5'

class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(
            node_name=node_name, node_type=NodeType.VISUALIZATION)

        # Initialize parameters
        self.base_speed = BASE_SPEED
        self.curve_speed = CURVE_SPEED
        self.p_gain = P_GAIN
        self.d_gain = D_GAIN
        self.max_steer = MAX_STEER
        
        # Error tracking for derivative control
        self.prev_error = 0
        self.left_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        self.right_motor_history = deque(maxlen=SMOOTHING_STRAIGHT)
        
        # Setup ROS nodes
        self._vehicle_name = os.environ['VEHICLE_NAME']

        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._window = "camera-reader"
        self.bridge = CvBridge()
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)

        self.sub = rospy.Subscriber(
            self._camera_topic, CompressedImage, self.callback)
        self._publisher = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1)

        self.left_motor = rospy.Publisher("left_motor", Float64, queue_size=1)
        self.right_motor = rospy.Publisher("right_motor", Float64, queue_size=1)

        self.shutting_down = False

        self.red_timer_start = None
        self.red_stop_duration = 3.0
        self.red_cooldown_start = None
        self.red_cooldown_duration = 2.0

        self.turn_state = -1
        self.changed_state = False
        
        # self.model = None
        # yolov5_model_path = 'yolov5.pt' # Assuming this file is in your script's directory
        
        # try:
        #     rospy.loginfo(f"Loading YOLOv5 model from: {yolov5_model_path}")
        #     self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path, force_reload=True)
        #     self.model.eval() # Set to evaluation mode
        #     # Optional: Set inference parameters (tune these as needed)
        #     # self.yolov5_model.conf = 0.5   # Confidence threshold
        #     # self.yolov5_model.iou = 0.45 # NMS IoU threshold
        #     rospy.loginfo("YOLOv5 model loaded successfully.")
        # except Exception as e:
        #     rospy.logerr(f"Failed to load YOLOv5 model: {e}")
        #     rospy.logwarn("YOLOv5 inference will not be available.")

        rospy.on_shutdown(self.shutdown_hook)

    def shutdown_hook(self):
        self.shutting_down = True
        self.left_motor.publish(0)
        self.right_motor.publish(0)
        cv2.destroyAllWindows()
        
    def smooth_motor_value(self, value, history_buffer):
        """Apply smoothing to motor values"""
        history_buffer.append(value)
        return sum(history_buffer) / len(history_buffer)
    
    def calculate_red_alignment(self, mask_red, vis_image):
        """
        Calculate alignment correction based on red mask edges
        Returns alignment_error and whether alignment is possible
        """
        h, w = mask_red.shape
        
        # Focus on the bottom portion where red should be most visible
        roi_start = int(h * 0.7)  # Bottom 30% of image
        roi_mask = mask_red[roi_start:, :]
        
        # Find all non-zero points in the ROI
        red_points = np.where(roi_mask > 0)
        
        if len(red_points[1]) < MIN_RED_WIDTH:  # Not enough red pixels
            return 0, False
            
        # Get the leftmost and rightmost red pixels
        left_edge = np.min(red_points[1])
        right_edge = np.max(red_points[1])
        red_width = right_edge - left_edge
        
        if red_width < MIN_RED_WIDTH:  # Red area too narrow
            return 0, False
            
        # Calculate center of red area
        red_center = (left_edge + right_edge) / 2
        image_center = w / 2
        
        # Calculate alignment error (positive = red is to the right, need to turn left)
        alignment_error = (red_center - image_center) / (w / 2)  # Normalize to [-1, 1]
        
        # Draw visualization
        roi_y = roi_start + roi_mask.shape[0] // 2
        cv2.circle(vis_image, (int(left_edge), roi_y), 5, (255, 0, 0), -1)  # Left edge - blue
        cv2.circle(vis_image, (int(right_edge), roi_y), 5, (0, 255, 0), -1)  # Right edge - green
        cv2.circle(vis_image, (int(red_center), roi_y), 5, (255, 255, 0), -1)  # Center - cyan
        cv2.line(vis_image, (int(image_center), roi_y-10), (int(image_center), roi_y+10), (255, 255, 255), 2)  # Image center
        
        return alignment_error, True
    
    def callback(self, msg):
        if self.shutting_down:
            return
        
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)

        # Process image
        self.image = cv2.bilateralFilter(self.image, 9, 75, 75)  # Less aggressive filtering

        # Create a copy for visualization
        vis_image = self.image.copy()
        
        # Define regions of interest - near field and far field
        h, w = self.image.shape[:2]

        # detections = []
        # if self.model is not None:
        #     try:
        #         # The YOLOv5 model expects RGB. OpenCV reads as BGR.
        #         # Convert BGR to RGB if your model was trained on RGB images.
        #         # Otherwise, if your model was trained on BGR, you might skip this.
        #         # Most pre-trained YOLOv5 models from Ultralytics expect RGB.
        #         image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                
        #         # Perform inference
        #         results = self.model(image_rgb)
                
        #         # Get detections as a pandas DataFrame for easy access
        #         # results.pandas().xyxy[0] returns a DataFrame for the first (and only) image in the batch
        #         detections_df = results.pandas().xyxy[0]
                
        #         # Convert DataFrame to a list of dictionaries or tuples for easier processing
        #         detections = detections_df.to_dict(orient='records')

        #         # Optional: Draw YOLOv5 detections on vis_image
        #         # The `results.render()` method is a convenient way to get an annotated image
        #         # However, it returns a NumPy array, which might overwrite your vis_image if not handled carefully.
        #         # For direct drawing onto `vis_image` without `results.render()`:
        #         for det in detections:
        #             x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        #             label = det['name']
        #             conf = det['confidence']
        #             color = (0, 255, 0) # Green for bounding boxes
        #             cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        #             cv2.putText(vis_image, f"{label} {conf:.2f}", (x1, y1 - 10), 
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        #         cv2.putText(vis_image, f"YOLOv5 Detections: {len(detections)}", (10, h - 30),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        #     except Exception as e:
        #         rospy.logerr(f"Error during YOLOv5 inference: {e}")
        
        # Color space conversions
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hls = cv2.cvtColor(self.image, cv2.COLOR_BGR2HLS)

        # Yellow line detection (usually left boundary)
        lb_yellow_hsv = np.array([22, 120, 120], dtype=np.uint8)
        ub_yellow_hsv = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow_hsv = cv2.inRange(hsv, lb_yellow_hsv, ub_yellow_hsv)

        # Convert to LAB color space for additional filtering
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        # In LAB: L=lightness, A=green-red, B=blue-yellow
        # Yellow has high B values, grass typically has lower B values
        lb_yellow_lab = np.array([50, 100, 130], dtype=np.uint8)  # High B for yellow
        ub_yellow_lab = np.array([255, 140, 200], dtype=np.uint8)
        mask_yellow_lab = cv2.inRange(lab, lb_yellow_lab, ub_yellow_lab)

        # Combine both masks
        mask_yellow = cv2.bitwise_and(mask_yellow_hsv, mask_yellow_lab)

        # Apply region of interest
        mask_yellow[:int(h*0.55), :] = 0

        # Red color has two ranges in HSV due to hue wraparound
        # Lower red range (0-10)
        lb_red_hsv1 = np.array([0, 100, 100], dtype=np.uint8)
        ub_red_hsv1 = np.array([10, 255, 255], dtype=np.uint8)
        mask_red1 = cv2.inRange(hsv, lb_red_hsv1, ub_red_hsv1)

        # Upper red range (170-180)
        lb_red_hsv2 = np.array([170, 100, 100], dtype=np.uint8)
        ub_red_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_red2 = cv2.inRange(hsv, lb_red_hsv2, ub_red_hsv2)

        # Combine both red masks
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Apply region of interest - focus on bottom part of image
        mask_red[:int(h*0.8), :] = 0

        if np.count_nonzero(mask_red) > 50:
            vis_image[mask_red > 0] = [0, 0, 255]

        # White line detection (usually right boundary)
        lb_white = np.array([0, 190, 0], dtype=np.uint8)
        ub_white = np.array([180, 255, 200], dtype=np.uint8)
        mask_white = cv2.inRange(hls, lb_white, ub_white)
        # Apply region of interest - focus on bottom part of image
        mask_white[:int(h*0.55), :] = 0
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((5, 5), np.uint8)
        mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)
        mask_white = cv2.dilate(mask_white, kernel, iterations=1)
        
        # Find contours to detect line shape
        yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # For visualization
        contour_img = np.zeros_like(self.image)
        cv2.drawContours(contour_img, yellow_contours, -1, (0, 255, 255), 2)
        cv2.drawContours(contour_img, white_contours, -1, (255, 255, 255), 2)
        
        # Initialize variables for line detection
        left_line_detected = len(yellow_contours) > 0
        right_line_detected = len(white_contours) > 0
        
        # Split the image into slices to detect curve
        num_slices = 3
        slice_height = int(h * 0.35 / num_slices)
        start_y = int(h * 0.55)
        
        yellow_x_points = []
        white_x_points = []
        y_points = []
        
        for i in range(num_slices):
            y = start_y + i * slice_height + slice_height // 2
            y_points.append(y)
            
            # Find yellow line x-position in this slice
            slice_yellow = mask_yellow[y-5:y+5, :]
            yellow_indices = np.where(slice_yellow > 0)[1]
            if len(yellow_indices) > 0:
                yellow_x = int(np.mean(yellow_indices))
                yellow_x_points.append(yellow_x)
                cv2.circle(vis_image, (yellow_x, y), 5, (0, 255, 255), -1)
            
            # Find white line x-position in this slice
            slice_white = mask_white[y-5:y+5, :]
            white_indices = np.where(slice_white > 0)[1]
            if len(white_indices) > 0:
                white_x = int(np.mean(white_indices))
                white_x_points.append(white_x)
                cv2.circle(vis_image, (white_x, y), 5, (255, 255, 255), -1)
                
        # Detect if we're in a curve by checking the difference in line positions
        is_curve = False
        curve_direction = 0
        
        if len(yellow_x_points) >= 2:
            yellow_diff = yellow_x_points[-1] - yellow_x_points[0]
            if abs(yellow_diff) > 20:  # Threshold for curve detection
                is_curve = True
                curve_direction += np.sign(yellow_diff)
                
        if len(white_x_points) >= 2:
            white_diff = white_x_points[-1] - white_x_points[0]
            if abs(white_diff) > 20:  # Threshold for curve detection
                is_curve = True
                curve_direction += np.sign(white_diff)
        
        # Determine center position and error
        if left_line_detected and right_line_detected and len(yellow_x_points) > 0 and len(white_x_points) > 0:
            # Both lines visible - use both for guidance
            center_position = (yellow_x_points[-1] + white_x_points[-1]) / 2
            ideal_center = w / 2
            error = ideal_center - center_position
        elif left_line_detected and len(yellow_x_points) > 0:
            # Only left (yellow) line visible - estimate center
            error = w/2 - (yellow_x_points[-1] + 160)  # Assume yellow line should be offset left
        elif right_line_detected and len(white_x_points) > 0:
            # Only right (white) line visible - estimate center
            error = w/2 - (white_x_points[-1] - 160)  # Assume white line should be offset right
        else:
            # No lines visible - maintain last direction but be cautious
            error = self.prev_error
        
        # Normalize error to range [-1, 1]
        error = np.clip(error / (w/2), -1, 1)
        
        # Calculate derivative of error for D-term
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # PD control
        steering = self.p_gain * error + self.d_gain * error_diff
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        
        # Adjust speeds based on curve detection
        current_speed = self.curve_speed if is_curve else self.base_speed
        
        # Calculate motor values
        left_motor = current_speed - steering
        right_motor = current_speed + steering
        
        # Special case: if in a tight curve, help by differential steering
        if is_curve and abs(steering) > 0.3:
            # Boost the inside wheel in curves
            if steering > 0:  # Turning left
                right_motor *= 1.3
            else:  # Turning right
                left_motor *= 1.3
        
        # Recovery behavior if no lines detected
        line_pixels_r = np.count_nonzero(mask_red)
        line_pixels_wy = np.count_nonzero(mask_yellow) + np.count_nonzero(mask_white)

        in_cooldown = (self.red_cooldown_start is not None and time.time() - self.red_cooldown_start < self.red_cooldown_duration)
        at_red_stop = False  # Flag to track if we're at a red stop

        if line_pixels_r > 800 and not in_cooldown:
            if self.red_timer_start is None:
                # First time detecting red - start timer
                self.red_timer_start = time.time()
            
            # Check if timer is still running
            if time.time() - self.red_timer_start < self.red_stop_duration:
                # We're at a red stop - perform alignment
                at_red_stop = True
                
                # Calculate red alignment
                alignment_error, can_align = self.calculate_red_alignment(mask_red, vis_image)
                
                if can_align:
                    # Use red alignment to make minor corrections while stopped
                    alignment_correction = ALIGNMENT_P_GAIN * alignment_error
                    alignment_correction = np.clip(alignment_correction, -ALIGNMENT_MAX_SPEED, ALIGNMENT_MAX_SPEED)
                    
                    # Apply small alignment movements
                    left_motor = alignment_correction
                    right_motor = -alignment_correction
                else:
                    # Can't align - just stay stopped
                    left_motor = 0.0
                    right_motor = 0.0
            else:
                # Timer finished - reset and continue normal operation
                self.red_timer_start = None
                self.red_cooldown_start = time.time()

                if self.turn_state == 1:
                    self.red_cooldown_duration = 2.5
                else:
                    self.red_cooldown_duration = 2.0

                self.changed_state = False
        else:
            # No red detected - reset timers
            self.red_timer_start = None

        if in_cooldown:
            if not self.changed_state:
                self.turn_state += 1
                self.changed_state = True

            # Original turning behavior during cooldown (no alignment here)
            if self.turn_state == 0:
                right_motor = 0.05
                left_motor = 0.35   
            elif self.turn_state == 1:
                right_motor = 0.5
                left_motor = 0.4
            else:
                self.turn_state = -1
                pass
        else:
            if line_pixels_wy < 500:  # Very few line pixels detected
                if line_pixels_r <= 100:  # Only if not seeing red
                    left_motor = -0.2
                    right_motor = -0.3

        # Determine smoothing amount based on curve detection
        smoothing_amount = SMOOTHING_CURVE if is_curve else SMOOTHING_STRAIGHT
        self.left_motor_history = deque(self.left_motor_history, maxlen=smoothing_amount)
        self.right_motor_history = deque(self.right_motor_history, maxlen=smoothing_amount)
        
        # Apply smoothing (but not during red alignment as we want precise control)
        if not at_red_stop:
            left_motor = self.smooth_motor_value(left_motor, self.left_motor_history)
            right_motor = self.smooth_motor_value(right_motor, self.right_motor_history)
        
        # Safety bounds
        left_motor = np.clip(left_motor, -1.0, 1.0)
        right_motor = np.clip(right_motor, -1.0, 1.0)
        
        # Publish motor commands
        if not self.shutting_down:
            self.left_motor.publish(left_motor)
            self.right_motor.publish(right_motor)

        # Display visualization
        cv2.putText(vis_image, f"Curve: {is_curve}, Dir: {curve_direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Error: {error:.2f}, Steer: {steering:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"At Red Stop: {at_red_stop}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Cooldown: {in_cooldown}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_image, f"State: {self.turn_state}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show red alignment info if at red stop
        if at_red_stop:
            alignment_error, can_align = self.calculate_red_alignment(mask_red, vis_image)
            cv2.putText(vis_image, f"Red Align: {alignment_error:.2f} {'OK' if can_align else 'NO'}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
        cv2.imshow(self._window, vis_image)
        cv2.waitKey(1)


if __name__ == '__main__':
    node = CameraReaderNode(node_name='camera_reader_node')
    rospy.spin()