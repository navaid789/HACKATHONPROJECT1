---
title: "Chapter 6 - Sensor Simulation: LiDAR, Depth Cameras, and IMUs"
description: "Understanding and implementing sensor simulation for robotics applications"
module: 2
chapter: 6
learning_objectives:
  - Implement realistic LiDAR simulation in robotics environments
  - Create depth camera models for 3D perception
  - Integrate IMU simulation for inertial measurements
  - Apply sensor fusion techniques for enhanced perception
difficulty: advanced
estimated_time: 120
tags:
  - sensors
  - lidar
  - cameras
  - imu
  - perception
  - simulation
  - robotics

authors:
  - Textbook Team
prerequisites:
  - Chapter 1: ROS 2 Fundamentals and Architecture
  - Chapter 4: Gazebo Simulation Fundamentals
  - Chapter 5: Unity Integration and High-Fidelity Rendering
  - Basic understanding of probability and statistics
---

# Chapter 6: Sensor Simulation: LiDAR, Depth Cameras, and IMUs

## Introduction

Sensors form the critical interface between robots and their environment, providing the data necessary for perception, navigation, and interaction. This chapter focuses on the simulation and implementation of three fundamental sensor types in robotics: LiDAR (Light Detection and Ranging), depth cameras, and IMUs (Inertial Measurement Units).

Realistic sensor simulation is crucial for developing and testing robotic systems before deployment on physical hardware. Understanding how to model sensor characteristics, noise patterns, and environmental effects enables more robust algorithm development and reduces the gap between simulation and reality.

## Sensor Fundamentals in Robotics

### Sensor Categories

Robot sensors can be broadly categorized into:

1. **Exteroceptive Sensors**: Sense the external environment (LiDAR, cameras, ultrasonic)
2. **Proprioceptive Sensors**: Sense the robot's own state (encoders, IMUs, joint sensors)
3. **Interoceptive Sensors**: Sense internal conditions (temperature, battery level)

For this chapter, we'll focus on exteroceptive sensors (LiDAR, depth cameras) and proprioceptive sensors (IMUs).

### Sensor Simulation Challenges

Key challenges in sensor simulation include:

- **Noise Modeling**: Accurately representing real sensor noise characteristics
- **Environmental Effects**: Simulating how environmental conditions affect sensor readings
- **Computational Efficiency**: Balancing simulation accuracy with performance
- **Realism**: Minimizing the "reality gap" between simulated and real sensors

## LiDAR Simulation

### LiDAR Principles

LiDAR sensors measure distance by illuminating targets with laser light and measuring the reflection time. In robotics, LiDAR is commonly used for:

- 2D/3D mapping and localization
- Obstacle detection and avoidance
- Navigation planning
- Object recognition and segmentation

### LiDAR Simulation in Gazebo

LiDAR simulation in Gazebo is achieved using the ray sensor plugin:

```xml
<!-- LiDAR sensor in URDF/Xacro -->
<gazebo reference="lidar_link">
  <sensor type="ray" name="main_lidar">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle> <!-- -π -->
          <max_angle>3.14159</max_angle>   <!-- π -->
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <frame_name>lidar_link</frame_name>
      <topic_name>scan</topic_name>
      <min_intensity>0.2</min_intensity>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Message Format

LiDAR data is typically published using the `sensor_msgs/LaserScan` message:

```python
# Example processing of LiDAR data in ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class LiDARProcessor(Node):
    def __init__(self):
        super().__init__('lidar_processor')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)

    def lidar_callback(self, msg):
        # Convert ranges to numpy array for processing
        ranges = np.array(msg.ranges)

        # Filter out invalid readings
        valid_ranges = ranges[(ranges >= msg.range_min) & (ranges <= msg.range_max)]

        # Calculate distance to closest obstacle
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f'Minimum distance: {min_distance:.2f}m')

        # Calculate distances in different directions
        num_ranges = len(ranges)
        front_idx = num_ranges // 2
        left_idx = num_ranges * 3 // 4
        right_idx = num_ranges // 4

        front_distance = ranges[front_idx]
        left_distance = ranges[left_idx]
        right_distance = ranges[right_idx]

        self.get_logger().info(
            f'Front: {front_distance:.2f}m, '
            f'Left: {left_distance:.2f}m, '
            f'Right: {right_distance:.2f}m'
        )

def main(args=None):
    rclpy.init(args=args)
    lidar_processor = LiDARProcessor()
    rclpy.spin(lidar_processor)
    lidar_processor.destroy_node()
    rclpy.shutdown()
```

### 3D LiDAR Simulation

For 3D mapping, sensors like Velodyne are simulated:

```xml
<!-- 3D LiDAR (Velodyne-like) -->
<gazebo reference="velodyne_link">
  <sensor type="ray" name="velodyne_sensor">
    <pose>0 0 0 0 0 0</pose>
    <visualize>false</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>1800</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
        <vertical>
          <samples>16</samples>
          <resolution>1</resolution>
          <min_angle>-0.2618</min_angle> <!-- -15 degrees -->
          <max_angle>0.2618</max_angle>  <!-- 15 degrees -->
        </vertical>
      </scan>
      <range>
        <min>0.1</min>
        <max>100.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="velodyne_controller" filename="libgazebo_ros_velodyne_laser.so">
      <topic_name>points</topic_name>
      <frame_name>velodyne_link</frame_name>
      <min_range>0.1</min_range>
      <max_range>100.0</max_range>
      <gaussian_noise>0.008</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### LiDAR Noise and Error Modeling

Real LiDAR sensors have various error sources:

```python
import numpy as np

class LiDARNoiseModel:
    def __init__(self, range_bias=0.01, range_noise_std=0.01,
                 resolution_noise=0.005, angular_noise_std=0.001):
        self.range_bias = range_bias
        self.range_noise_std = range_noise_std
        self.resolution_noise = resolution_noise
        self.angular_noise_std = angular_noise_std

    def add_noise(self, true_ranges, angles):
        """Add realistic noise to LiDAR measurements"""
        # Add bias error
        noisy_ranges = true_ranges + self.range_bias

        # Add Gaussian noise proportional to range
        range_dependent_noise = self.range_noise_std * noisy_ranges
        noisy_ranges += np.random.normal(0, range_dependent_noise)

        # Add resolution-dependent noise
        noisy_ranges += np.random.normal(0, self.resolution_noise, size=noisy_ranges.shape)

        # Ensure valid range readings
        noisy_ranges = np.clip(noisy_ranges, 0.1, 100.0)

        # Angular noise affects accuracy of bearing
        angular_noise = np.random.normal(0, self.angular_noise_std, size=angles.shape)
        noisy_angles = angles + angular_noise

        return noisy_ranges, noisy_angles

# Example usage
noise_model = LiDARNoiseModel()
true_ranges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
angles = np.linspace(-np.pi, np.pi, len(true_ranges))
noisy_ranges, noisy_angles = noise_model.add_noise(true_ranges, angles)
```

## Depth Camera Simulation

### Depth Camera Principles

Depth cameras provide both color (RGB) and depth information for each pixel. They are essential for:

- 3D scene reconstruction
- Object detection and recognition
- Augmented reality applications
- Navigation in complex environments

### Depth Camera Simulation in Gazebo

Depth cameras are implemented using the depth camera plugin:

```xml
<!-- Depth camera sensor -->
<gazebo reference="camera_link">
  <sensor type="depth" name="depth_camera">
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <baseline>0.2</baseline>
      <always_on>true</always_on>
      <update_rate>30</update_rate>
      <camera_name>camera</camera_name>
      <image_topic_name>rgb/image_raw</image_topic_name>
      <depth_image_topic_name>depth/image_raw</depth_image_topic_name>
      <point_cloud_topic_name>depth/points</point_cloud_topic_name>
      <camera_info_topic_name>rgb/camera_info</camera_info_topic_name>
      <depth_image_camera_info_topic_name>depth/camera_info</depth_image_camera_info_topic_name>
      <frame_name>camera_rgb_optical_frame</frame_name>
      <point_cloud_cutoff>0.1</point_cloud_cutoff>
      <point_cloud_cutoff_max>3.0</point_cloud_cutoff_max>
      <Cx>0</Cx>
      <Cy>0</Cy>
      <focal_length>0</focal_length>
    </plugin>
  </sensor>
</gazebo>
```

### Processing Depth Camera Data

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')
        self.bridge = CvBridge()

        # Subscribers for RGB and depth images
        self.rgb_sub = self.create_subscription(
            Image, 'camera/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, 'camera/depth/image_raw', self.depth_callback, 10)

        # Storage for camera info
        self.camera_info = None

    def rgb_callback(self, msg):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process RGB image here
            processed_image = self.process_rgb_image(cv_image)

            # Display the image (optional)
            cv2.imshow('RGB Image', processed_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        try:
            # Convert depth image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

            # Process depth image
            obstacle_map = self.detect_obstacles(depth_image)

            # Log distance to nearest obstacle
            if obstacle_map.size > 0:
                min_dist = np.min(obstacle_map[obstacle_map > 0])
                self.get_logger().info(f'Minimum obstacle distance: {min_dist:.2f}m')

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def process_rgb_image(self, image):
        # Example processing: edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def detect_obstacles(self, depth_image, threshold=0.5):
        # Create obstacle mask (points closer than threshold)
        obstacle_mask = (depth_image > 0) & (depth_image < threshold)
        return depth_image * obstacle_mask

def main(args=None):
    rclpy.init(args=args)
    processor = DepthCameraProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()
```

### Depth Image Processing Techniques

```python
import numpy as np
import cv2
from scipy import ndimage

class DepthImageProcessor:
    def __init__(self):
        pass

    def segment_planes(self, depth_image, threshold=0.01):
        """Segment planar surfaces using RANSAC-like approach"""
        # Convert to point cloud
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Assuming camera intrinsic parameters (these should be calibrated)
        fx, fy = width / (2 * np.tan(0.5)), height / (2 * np.tan(0.5))  # approx
        cx, cy = width / 2, height / 2

        # Convert to 3D points
        z = depth_image
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Stack into point cloud
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        # Remove invalid points (where depth is 0)
        valid_points = points[points[:, 2] > 0]

        return valid_points

    def detect_edges(self, depth_image, low_threshold=0.1, high_threshold=0.5):
        """Detect depth discontinuities as edges"""
        # Calculate depth gradients
        grad_x = np.gradient(depth_image, axis=1)
        grad_y = np.gradient(depth_image, axis=0)

        # Combine gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Apply thresholds
        edges = (gradient_magnitude > low_threshold) & (gradient_magnitude < high_threshold)

        return edges.astype(np.uint8) * 255

    def fill_depth_holes(self, depth_image, max_hole_size=10):
        """Fill small holes in depth image"""
        # Create mask of invalid regions
        invalid_mask = (depth_image <= 0) | np.isnan(depth_image)

        # Fill holes using morphological operations
        kernel = np.ones((3,3), np.uint8)
        filled_mask = cv2.morphologyEx(invalid_mask.astype(np.uint8),
                                     cv2.MORPH_CLOSE, kernel, iterations=1)

        # Use inpainting to fill holes
        filled_depth = depth_image.copy()
        filled_depth[filled_mask.astype(bool)] = np.nan

        # Use scipy to fill nan values with nearest neighbors
        invalid_pixels = np.where(np.isnan(filled_depth))

        if len(invalid_pixels[0]) > 0:
            # Find nearest valid neighbors for each invalid pixel
            for i in range(len(invalid_pixels[0])):
                y, x = invalid_pixels[0][i], invalid_pixels[1][i]

                # Search in a small window around the invalid pixel
                min_dist = float('inf')
                fill_value = 0

                for dy in range(-max_hole_size, max_hole_size + 1):
                    for dx in range(-max_hole_size, max_hole_size + 1):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < depth_image.shape[0] and
                            0 <= nx < depth_image.shape[1] and
                            not np.isnan(depth_image[ny, nx])):
                            dist = np.sqrt(dy**2 + dx**2)
                            if dist < min_dist:
                                min_dist = dist
                                fill_value = depth_image[ny, nx]

                filled_depth[y, x] = fill_value

        return filled_depth
```

## IMU Simulation

### IMU Principles

An IMU typically contains accelerometers, gyroscopes, and sometimes magnetometers to measure:

- **Linear acceleration** in three axes (x, y, z)
- **Angular velocity** in three axes (roll, pitch, yaw)
- **Magnetic field** (compass heading)

IMUs are crucial for:
- Robot orientation estimation
- Motion detection and tracking
- Stabilization systems
- Dead reckoning navigation

### IMU Simulation in Gazebo

```xml
<!-- IMU sensor in URDF/Xacro -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <topic>__default_topic__</topic>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.002</bias_mean>
            <bias_stddev>0.0003</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.002</bias_mean>
            <bias_stddev>0.0003</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.002</bias_mean>
            <bias_stddev>0.0003</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.01</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.01</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.01</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topic_name>imu</topic_name>
      <body_name>imu_link</body_name>
      <frame_name>imu_link</frame_name>
      <update_rate>100</update_rate>
      <gaussian_noise>0.01</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Data Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # IMU subscriber
        self.subscription = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)

        # Store previous orientation for velocity estimation
        self.prev_orientation = None
        self.prev_time = None

        # Integration parameters
        self.angular_velocity_bias = np.array([0.0, 0.0, 0.0])
        self.linear_acceleration_bias = np.array([0.0, 0.0, 0.0])

    def imu_callback(self, msg):
        # Extract orientation (quaternion)
        orientation_q = np.array([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Extract angular velocity
        angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Extract linear acceleration
        linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Convert quaternion to rotation matrix
        rotation = R.from_quat(orientation_q)
        rotation_matrix = rotation.as_matrix()

        # Remove gravity from linear acceleration
        gravity = np.array([0, 0, 9.81])  # Gravity in world frame
        gravity_in_body = rotation_matrix.T @ gravity  # Gravity in body frame
        acceleration_without_gravity = linear_acceleration - gravity_in_body

        # Log processed values
        self.get_logger().info(
            f'Orientation: ({orientation_q[0]:.3f}, {orientation_q[1]:.3f}, '
            f'{orientation_q[2]:.3f}, {orientation_q[3]:.3f})\n'
            f'Angular Vel: ({angular_velocity[0]:.3f}, {angular_velocity[1]:.3f}, '
            f'{angular_velocity[2]:.3f}) rad/s\n'
            f'Accel (no gravity): ({acceleration_without_gravity[0]:.3f}, '
            f'{acceleration_without_gravity[1]:.3f}, {acceleration_without_gravity[2]:.3f}) m/s²'
        )

def main(args=None):
    rclpy.init(args=args)
    processor = IMUProcessor()
    rclpy.spin(processor)
    processor.destroy_node()
    rclpy.shutdown()
```

### IMU Bias and Drift Correction

```python
import numpy as np
from collections import deque
import statistics

class IMUCalibrator:
    def __init__(self, calibration_samples=1000):
        self.calibration_samples = calibration_samples
        self.gyro_samples = deque(maxlen=calibration_samples)
        self.accel_samples = deque(maxlen=calibration_samples)

        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.gravity_vector = np.array([0, 0, 9.81])  # Assumed gravity

    def add_calibration_sample(self, gyro_data, accel_data):
        """Add sample for bias calculation"""
        self.gyro_samples.append(gyro_data)
        self.accel_samples.append(accel_data)

        # Calculate bias when we have enough samples
        if len(self.gyro_samples) == self.calibration_samples:
            self.calculate_bias()

    def calculate_bias(self):
        """Calculate bias based on collected samples"""
        if len(self.gyro_samples) == self.calibration_samples:
            # Calculate mean for bias estimation
            gyro_array = np.array(self.gyro_samples)
            accel_array = np.array(self.accel_samples)

            self.gyro_bias = np.mean(gyro_array, axis=0)
            self.accel_bias = np.mean(accel_array, axis=0)

            # Adjust accelerometer bias to account for gravity
            # When the IMU is at rest, the accelerometer should read gravity
            self.accel_bias = self.accel_bias - self.gravity_vector

    def correct_gyro(self, raw_gyro):
        """Apply bias correction to gyro readings"""
        return raw_gyro - self.gyro_bias

    def correct_accel(self, raw_accel):
        """Apply bias correction to accelerometer readings"""
        return raw_accel - self.accel_bias

# Extended IMU processor with bias correction
class CorrectedIMUProcessor(IMUProcessor):
    def __init__(self):
        super().__init__()

        # Initialize calibrator
        self.calibrator = IMUCalibrator()
        self.calibration_complete = False
        self.calibration_counter = 0

        # Integration variables
        self.estimated_velocity = np.zeros(3)
        self.estimated_position = np.zeros(3)
        self.prev_time = None

    def imu_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Raw sensor data
        raw_angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        raw_linear_acceleration = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        if not self.calibration_complete:
            # Collect samples for bias calculation
            self.calibrator.add_calibration_sample(
                raw_angular_velocity, raw_linear_acceleration
            )

            self.calibration_counter += 1

            if self.calibration_counter >= self.calibrator.calibration_samples:
                self.calibrator.calculate_bias()
                self.calibration_complete = True
                self.get_logger().info('IMU calibration complete')
                self.get_logger().info(f'Gyro bias: {self.calibrator.gyro_bias}')
                self.get_logger().info(f'Accel bias: {self.calibrator.accel_bias}')
        else:
            # Apply bias correction
            corrected_angular_velocity = self.calibrator.correct_gyro(raw_angular_velocity)
            corrected_linear_acceleration = self.calibrator.correct_accel(raw_linear_acceleration)

            # Integrate to get velocity and position (simplified)
            if self.prev_time is not None:
                dt = current_time - self.prev_time

                # Remove gravity from corrected acceleration
                # (this requires knowing the orientation)
                gravity_removed_accel = corrected_linear_acceleration  # Simplified

                # Update velocity and position (simplified)
                self.estimated_velocity += gravity_removed_accel * dt
                self.estimated_position += self.estimated_velocity * dt + 0.5 * gravity_removed_accel * dt**2

            self.prev_time = current_time

            # Log corrected values
            self.get_logger().info(
                f'Corrected Angular Vel: ({corrected_angular_velocity[0]:.3f}, '
                f'{corrected_angular_velocity[1]:.3f}, {corrected_angular_velocity[2]:.3f}) rad/s\n'
                f'Corrected Accel: ({corrected_linear_acceleration[0]:.3f}, '
                f'{corrected_linear_acceleration[1]:.3f}, {corrected_linear_acceleration[2]:.3f}) m/s²'
            )
```

## Sensor Fusion

### Kalman Filter for Sensor Fusion

```python
import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # Initialize state vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(state_dim)

        # Initialize covariance matrix
        self.P = np.eye(state_dim) * 1000  # Large initial uncertainty

        # Process noise
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise
        self.R = np.eye(measurement_dim) * 1.0

    def predict(self, dt):
        """Prediction step"""
        # State transition model (constant velocity model)
        F = np.eye(self.state_dim)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """Update step with measurement z"""
        # Measurement model (position only)
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0, 0] = 1  # Measure x
        H[1, 1] = 1  # Measure y
        H[2, 2] = 1  # Measure z

        # Innovation
        y = z - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Initialize EKF
        self.ekf = ExtendedKalmanFilter(state_dim=6, measurement_dim=3)  # 3 pos + 3 vel, measure 3 pos

        # Subscribers for different sensors
        self.gps_sub = self.create_subscription(
            Point, 'gps', self.gps_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, 'imu', self.imu_callback, 10)

        # Timer for prediction step
        self.timer = self.create_timer(0.01, self.prediction_step)  # 100 Hz

        self.prev_time = self.get_clock().now().nanoseconds / 1e9

    def prediction_step(self):
        """Prediction step of the filter"""
        current_time = self.get_clock().now().nanoseconds / 1e9
        dt = current_time - self.prev_time

        if dt > 0:
            self.ekf.predict(dt)
            self.prev_time = current_time

    def gps_callback(self, msg):
        """GPS measurement update"""
        # GPS provides position measurement
        z = np.array([msg.x, msg.y, msg.z])
        self.ekf.update(z)

    def odom_callback(self, msg):
        """Odometry measurement update"""
        # Odometry provides position and velocity
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.ekf.update(pos)

    def imu_callback(self, msg):
        """IMU measurement (for acceleration-based updates)"""
        # Can be used for acceleration-based updates
        pass
```

## Sensor Simulation in Unity

### Unity Sensor Simulation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class UnitySensorSimulator : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public float lidarRange = 10.0f;
    public int lidarRays = 360;
    public float lidarAngle = 360f;
    public float lidarUpdateRate = 10f;

    public float cameraFOV = 60f;
    public int cameraWidth = 640;
    public int cameraHeight = 480;
    public float cameraRange = 10f;

    [Header("Noise Parameters")]
    public float rangeNoiseStd = 0.01f;
    public float angularNoiseStd = 0.001f;

    private Camera sensorCamera;
    private float updateInterval;
    private float lastUpdateTime;

    void Start()
    {
        // Setup sensor camera if needed
        SetupSensorCamera();

        // Calculate update interval based on rate
        updateInterval = 1.0f / lidarUpdateRate;
        lastUpdateTime = Time.time;
    }

    void SetupSensorCamera()
    {
        // Create camera for depth simulation
        sensorCamera = GetComponent<Camera>();
        if (sensorCamera == null)
        {
            sensorCamera = gameObject.AddComponent<Camera>();
        }

        sensorCamera.fieldOfView = cameraFOV;
        sensorCamera.nearClipPlane = 0.1f;
        sensorCamera.farClipPlane = cameraRange;
        sensorCamera.enabled = false; // We'll render on demand
    }

    void Update()
    {
        if (Time.time - lastUpdateTime > updateInterval)
        {
            SimulateSensors();
            lastUpdateTime = Time.time;
        }
    }

    void SimulateSensors()
    {
        // Simulate LiDAR
        SimulateLiDAR();

        // Simulate camera
        SimulateCamera();
    }

    float[] SimulateLiDAR()
    {
        float[] ranges = new float[lidarRays];
        float angleStep = lidarAngle / lidarRays;

        for (int i = 0; i < lidarRays; i++)
        {
            float angle = transform.eulerAngles.y + (i * angleStep) - (lidarAngle / 2);
            Vector3 direction = new Vector3(
                Mathf.Sin(angle * Mathf.Deg2Rad),
                0,
                Mathf.Cos(angle * Mathf.Deg2Rad)
            );

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, lidarRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = lidarRange; // No obstacle detected
            }

            // Add noise to range measurement
            ranges[i] += RandomGaussianNoise(rangeNoiseStd);
        }

        return ranges;
    }

    Texture2D SimulateCamera()
    {
        // Create temporary render texture
        RenderTexture tempRT = RenderTexture.GetTemporary(
            cameraWidth, cameraHeight, 24);

        // Set the camera to render to our temporary render texture
        sensorCamera.targetTexture = tempRT;
        sensorCamera.Render();

        // Create a texture to read the render texture
        Texture2D tex = new Texture2D(cameraWidth, cameraHeight,
                                    TextureFormat.RGB24, false);

        // Store the currently active RenderTexture
        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = tempRT;

        // Read the render texture pixels into the texture
        tex.ReadPixels(new Rect(0, 0, cameraWidth, cameraHeight), 0, 0);
        tex.Apply();

        // Restore the previous RenderTexture
        RenderTexture.active = previous;

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(tempRT);

        return tex;
    }

    float RandomGaussianNoise(float stdDev)
    {
        // Box-Muller transform for Gaussian noise
        float u1 = Random.value;
        float u2 = Random.value;
        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return normal * stdDev;
    }

    void OnDrawGizmos()
    {
        // Draw LiDAR rays for visualization
        if (lidarRays <= 100) // Only draw if not too many rays
        {
            float angleStep = lidarAngle / lidarRays;

            for (int i = 0; i < lidarRays; i++)
            {
                float angle = transform.eulerAngles.y + (i * angleStep) - (lidarAngle / 2);
                Vector3 direction = new Vector3(
                    Mathf.Sin(angle * Mathf.Deg2Rad),
                    0,
                    Mathf.Cos(angle * Mathf.Deg2Rad)
                );

                RaycastHit hit;
                if (Physics.Raycast(transform.position, direction, out hit, lidarRange))
                {
                    Gizmos.color = Color.green;
                    Gizmos.DrawLine(transform.position, hit.point);
                }
                else
                {
                    Gizmos.color = Color.red;
                    Gizmos.DrawLine(transform.position,
                                  transform.position + direction * lidarRange);
                }
            }
        }
    }
}
```

## Performance Optimization for Sensor Simulation

### Efficient Sensor Processing

```csharp
using System.Collections.Generic;
using UnityEngine;
using Unity.Jobs;

public class OptimizedSensorSimulator : MonoBehaviour
{
    [Header("Performance Settings")]
    public bool useMultiThreading = true;
    public int maxSensors = 10;

    private List<SensorData> sensorDataList;
    private Queue<SensorReading> readingQueue;
    private System.Threading.Mutex queueMutex;

    [System.Serializable]
    public class SensorData
    {
        public string sensorName;
        public string sensorType; // "lidar", "camera", "imu"
        public float lastUpdateTime;
        public float updateInterval;
        public Vector3 position;
        public Quaternion rotation;
    }

    [System.Serializable]
    public class SensorReading
    {
        public string sensorName;
        public float timestamp;
        public object data; // Could be float[], Texture2D, etc.
    }

    void Start()
    {
        sensorDataList = new List<SensorData>();
        readingQueue = new Queue<SensorReading>();
        queueMutex = new System.Threading.Mutex();
    }

    public void AddSensor(string name, string type, float updateRate)
    {
        SensorData sensor = new SensorData
        {
            sensorName = name,
            sensorType = type,
            updateInterval = 1.0f / updateRate,
            lastUpdateTime = Time.time,
            position = transform.position,
            rotation = transform.rotation
        };

        sensorDataList.Add(sensor);
    }

    void Update()
    {
        // Process sensors based on their update rates
        foreach (var sensor in sensorDataList)
        {
            if (Time.time - sensor.lastUpdateTime > sensor.updateInterval)
            {
                ProcessSensorReading(sensor);
                sensor.lastUpdateTime = Time.time;
            }
        }
    }

    void ProcessSensorReading(SensorData sensor)
    {
        object sensorData = null;

        switch (sensor.sensorType)
        {
            case "lidar":
                sensorData = SimulateLiDARReading(sensor);
                break;
            case "camera":
                sensorData = SimulateCameraReading(sensor);
                break;
            case "imu":
                sensorData = SimulateIMUReading(sensor);
                break;
        }

        if (sensorData != null)
        {
            SensorReading reading = new SensorReading
            {
                sensorName = sensor.sensorName,
                timestamp = Time.time,
                data = sensorData
            };

            // Add to queue thread-safely
            queueMutex.WaitOne();
            readingQueue.Enqueue(reading);
            queueMutex.ReleaseMutex();
        }
    }

    object SimulateLiDARReading(SensorData sensor)
    {
        // Optimized LiDAR simulation using Unity's physics
        float[] ranges = new float[360]; // Simplified for performance

        for (int i = 0; i < ranges.Length; i++)
        {
            float angle = (i * 360f / ranges.Length) * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            RaycastHit hit;
            if (Physics.Raycast(sensor.position, direction, out hit, 10f))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = 10f;
            }
        }

        return ranges;
    }

    object SimulateCameraReading(SensorData sensor)
    {
        // Simplified camera simulation
        return new Texture2D(640, 480); // Placeholder
    }

    object SimulateIMUReading(SensorData sensor)
    {
        // Simulate IMU reading based on Unity's physics
        Vector3 angularVelocity = GetSimulatedAngularVelocity();
        Vector3 linearAcceleration = GetSimulatedLinearAcceleration(sensor.position);

        return new { angular_velocity = angularVelocity, linear_acceleration = linearAcceleration };
    }

    Vector3 GetSimulatedAngularVelocity()
    {
        // Calculate from Unity's rotation change
        // Implementation depends on how you're tracking rotation
        return Vector3.zero; // Placeholder
    }

    Vector3 GetSimulatedLinearAcceleration(Vector3 position)
    {
        // Calculate from Unity's physics simulation
        // Could be based on forces applied to the rigidbody
        return Vector3.zero; // Placeholder
    }

    public bool TryGetSensorReading(out SensorReading reading)
    {
        reading = null;

        queueMutex.WaitOne();
        if (readingQueue.Count > 0)
        {
            reading = readingQueue.Dequeue();
        }
        queueMutex.ReleaseMutex();

        return reading != null;
    }
}
```

## Summary

This chapter covered the simulation and processing of three critical sensor types in robotics:

- **LiDAR Simulation**: Understanding ray sensors, noise modeling, and 2D/3D LiDAR implementation
- **Depth Camera Simulation**: Creating and processing RGB-D data for 3D perception
- **IMU Simulation**: Modeling inertial sensors with proper bias and drift correction
- **Sensor Fusion**: Combining multiple sensors using Kalman filters for improved accuracy

Realistic sensor simulation is crucial for developing robust robotic systems that can operate effectively in real-world conditions. Proper modeling of sensor characteristics, noise patterns, and environmental effects significantly improves the transferability of algorithms from simulation to reality.

## Learning Check

After completing this chapter, you should be able to:
- Implement realistic LiDAR simulation with proper noise modeling
- Create and process depth camera data for 3D perception tasks
- Simulate IMU sensors with bias and drift correction
- Apply sensor fusion techniques to combine multiple sensor readings
- Optimize sensor simulation for performance in complex environments