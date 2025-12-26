---
title: Chapter 9 - Navigation (Nav2) for Humanoid Robots
description: "ROS 2 Navigation 2 (Nav2) for autonomous navigation and path planning"
module: 3
chapter: 9
learning_objectives:
  - Understand ROS 2 Navigation 2 (Nav2) architecture and components
  - Configure navigation for humanoid robot platforms
  - Implement path planning and obstacle avoidance algorithms
  - Integrate perception systems with navigation
difficulty: advanced
estimated_time: 120
tags:
  - nav2
  - navigation
  - path-planning
  - autonomous-navigation
  - robotics
  - ros2

authors:
  - Textbook Team
prerequisites:
  - Chapter 1: ROS 2 Fundamentals and Architecture
  - Chapter 2: Nodes, Topics, and Services
  - Chapter 6: Sensor Simulation and Integration
  - Chapter 7: Isaac Sim for Physical AI
  - Basic understanding of path planning algorithms
---

# Chapter 9: Navigation (Nav2) for Humanoid Robots

## Introduction

ROS 2 Navigation 2 (Nav2) is the next-generation navigation framework designed specifically for ROS 2. It provides a flexible, modular, and robust system for autonomous navigation that can be adapted for various robot platforms, including humanoid robots. Nav2 enables robots to navigate autonomously in complex environments by combining perception, path planning, and control algorithms.

Unlike its predecessor in ROS 1, Nav2 has been completely redesigned with modern software engineering practices, improved performance, and better integration with the ROS 2 ecosystem. It's particularly well-suited for humanoid robots due to its modular architecture and support for complex kinematic constraints.

## Overview of Nav2

### Architecture and Components

Nav2 follows a behavior tree-based architecture that allows for complex navigation behaviors and easy customization. The main components include:

1. **Navigation Server**: Central orchestrator that manages navigation requests
2. **Planners Server**: Handles global and local path planning
3. **Controller Server**: Manages trajectory control and execution
4. **Recovery Server**: Handles navigation recovery behaviors
5. **Lifecycle Manager**: Manages the lifecycle of navigation components

### Key Features

- **Modular Design**: Components can be easily swapped or customized
- **Behavior Trees**: Advanced decision-making capabilities
- **Costmap Integration**: 2D and 3D costmap support for obstacle avoidance
- **Recovery Behaviors**: Automatic recovery from navigation failures
- **Multi-robot Support**: Coordination between multiple robots
- **Plugin Architecture**: Extensible with custom components

## Nav2 Architecture

### Core Components

The Nav2 system consists of several interconnected servers:

```python
# Example Nav2 server configuration
from nav2_behavior_tree.nav2_server import Nav2Server
from nav2_planners_server.planners_server import PlannersServer
from nav2_controller_server.controller_server import ControllerServer

class Nav2System:
    def __init__(self):
        # Initialize core servers
        self.navigation_server = Nav2Server()
        self.planners_server = PlannersServer()
        self.controller_server = ControllerServer()
        self.recovery_server = RecoveryServer()

        # Initialize costmap components
        self.global_costmap = GlobalCostmap()
        self.local_costmap = LocalCostmap()
```

### Behavior Tree Integration

Nav2 uses behavior trees for complex navigation decision-making:

```xml
<!-- Example behavior tree for navigation -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <Sequence name="NavigateToPose">
            <ComputePathToPose goal="{goal}" path="{path}"/>
            <SmoothPath path="{path}" smoothed_path="{smoothed_path}"/>
            <FollowPath path="{smoothed_path}"/>
        </Sequence>
    </BehaviorTree>
</root>
```

### Lifecycle Management

Nav2 components follow the ROS 2 lifecycle pattern:

```python
from lifecycle_msgs.msg import Transition
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

class Nav2LifecycleManager(LifecycleNode):
    def __init__(self):
        super().__init__('nav2_lifecycle_manager')

    def on_configure(self, state):
        # Configure navigation components
        self.get_logger().info('Configuring navigation components')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        # Activate navigation components
        self.get_logger().info('Activating navigation components')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        # Deactivate navigation components
        self.get_logger().info('Deactivating navigation components')
        return TransitionCallbackReturn.SUCCESS
```

## Costmap Configuration for Humanoid Robots

### Global Costmap

The global costmap provides a representation of the known environment:

```yaml
# Global costmap configuration for humanoid robot
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: false
      width: 100
      height: 100
      resolution: 0.05
      origin_x: -25.0
      origin_y: -25.0

      # Robot footprint for humanoid
      footprint: "[[-0.3, -0.2], [-0.3, 0.2], [0.3, 0.2], [0.3, -0.2]]"
      footprint_padding: 0.01

      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

### Local Costmap

The local costmap provides real-time obstacle information:

```yaml
# Local costmap configuration for humanoid robot
local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: true
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05

      # Robot footprint for humanoid
      footprint: "[[-0.3, -0.2], [-0.3, 0.2], [0.3, 0.2], [0.3, -0.2]]"
      footprint_padding: 0.01

      plugins: ["obstacle_layer", "voxel_layer", "inflation_layer"]

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"

      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        publish_voxel_map: true
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
```

## Path Planning Algorithms

### Global Planners

Nav2 supports multiple global planners:

#### A* Planner

```python
# A* path planner implementation
import numpy as np
from nav2_core.global_planner import GlobalPlanner
from geometry_msgs.msg import PoseStamped

class AStarPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()
        self.name = "AStarPlanner"

    def create_plan(self, start, goal):
        # Implement A* algorithm
        # Convert costmap to grid
        grid = self.costmap_to_grid()

        # A* pathfinding
        path = self.a_star_search(grid, start, goal)

        # Convert path to ROS message
        plan = self.path_to_msg(path)
        return plan

    def a_star_search(self, grid, start, goal):
        # A* implementation with heuristic
        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

            if current == goal:
                # Reconstruct path
                return self.reconstruct_path(came_from, current)

            open_set.remove(current)

            for neighbor in self.get_neighbors(current, grid):
                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)

                    if neighbor not in open_set:
                        open_set.append(neighbor)

        return []  # No path found
```

#### Dijkstra Planner

```python
# Dijkstra path planner
class DijkstraPlanner(GlobalPlanner):
    def __init__(self):
        super().__init__()
        self.name = "DijkstraPlanner"

    def create_plan(self, start, goal):
        # Dijkstra algorithm implementation
        grid = self.costmap_to_grid()
        distances = {start: 0}
        previous = {}
        unvisited = set()

        # Initialize distances
        for cell in grid:
            if cell != start:
                distances[cell] = float('inf')
            unvisited.add(cell)

        current = start

        while unvisited:
            # Find unvisited node with smallest distance
            current = min(unvisited, key=lambda x: distances[x])

            if distances[current] == float('inf'):
                break  # Remaining nodes are unreachable

            if current == goal:
                break

            unvisited.remove(current)

            for neighbor in self.get_neighbors(current, grid):
                if neighbor in unvisited:
                    alt = distances[current] + self.distance(current, neighbor)
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        previous[neighbor] = current

        # Reconstruct path
        path = self.reconstruct_path(previous, goal)
        plan = self.path_to_msg(path)
        return plan
```

### Local Planners

#### Trajectory Rollout

```python
# Local trajectory planner for humanoid robots
from nav2_core.controller import Controller
from geometry_msgs.msg import Twist
import numpy as np

class HumanoidLocalPlanner(Controller):
    def __init__(self):
        super().__init__()
        self.name = "HumanoidLocalPlanner"
        self.max_vel_x = 0.5
        self.max_vel_theta = 1.0
        self.min_vel_x = 0.1
        self.min_vel_theta = 0.1

    def compute_velocity_commands(self, pose, velocity, goal_checker):
        # Compute velocity commands for humanoid robot
        # Consider kinematic constraints
        cmd_vel = Twist()

        # Get current goal
        goal = self.get_current_goal()

        # Calculate distance to goal
        dist_to_goal = self.calculate_distance(pose.pose.position, goal.pose.position)

        if dist_to_goal < 0.1:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            return cmd_vel

        # Calculate desired velocity based on distance
        linear_vel = min(self.max_vel_x, dist_to_goal * 0.5)

        # Calculate angular velocity to face goal
        angle_to_goal = self.calculate_angle_to_goal(pose, goal)
        angular_vel = max(-self.max_vel_theta, min(self.max_vel_theta, angle_to_goal * 2.0))

        # Apply safety checks with local costmap
        safe_linear_vel = self.check_linear_velocity(linear_vel, pose, velocity)
        safe_angular_vel = self.check_angular_velocity(angular_vel, pose, velocity)

        cmd_vel.linear.x = safe_linear_vel
        cmd_vel.angular.z = safe_angular_vel

        return cmd_vel

    def check_linear_velocity(self, desired_vel, pose, velocity):
        # Check if desired linear velocity is safe based on local costmap
        # This would involve checking for obstacles in the robot's path
        return desired_vel  # Simplified for example

    def check_angular_velocity(self, desired_vel, pose, velocity):
        # Check if desired angular velocity is safe
        return desired_vel  # Simplified for example
```

## Nav2 Configuration for Humanoid Robots

### Navigation Parameters

```yaml
# Navigation parameters for humanoid robot
bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: true
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_to_pose_bt_xml: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node
    - nav2_controller_cancel_bt_node
    - nav2_path_longer_on_approach_bt_node
    - nav2_wait_cancel_bt_node
```

### Recovery Behaviors

```yaml
# Recovery behaviors configuration
recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
    backup:
      plugin: "nav2_recoveries/BackUp"
    wait:
      plugin: "nav2_recoveries/Wait"
    global_frame: odom
    robot_base_frame: base_link
    transform_timeout: 0.1
    use_sim_time: true
    simulate_ahead_time: 2.0
    max_rotational_vel: 1.0
    min_rotational_vel: 0.4
    rotational_acc_lim: 3.2
```

## Implementing Humanoid-Specific Navigation

### Kinematic Constraints

Humanoid robots have specific kinematic constraints that need to be considered in navigation:

```python
# Humanoid kinematic constraints for navigation
class HumanoidKinematicConstraints:
    def __init__(self):
        # Humanoid-specific parameters
        self.step_height = 0.15  # Maximum step height
        self.step_width = 0.30   # Maximum step width
        self.turning_radius = 0.25  # Minimum turning radius
        self.max_angular_velocity = 0.5  # Maximum angular velocity
        self.max_linear_velocity = 0.3   # Maximum linear velocity
        self.balance_margin = 0.1        # Balance safety margin

    def is_path_valid(self, path):
        # Check if path is valid for humanoid robot
        for i in range(len(path) - 1):
            # Check step height constraints
            if not self.check_step_height(path[i], path[i+1]):
                return False

            # Check turning constraints
            if not self.check_turning_radius(path[i], path[i+1]):
                return False

        return True

    def check_step_height(self, pose1, pose2):
        # Check if step height is within humanoid capabilities
        height_diff = abs(pose1.position.z - pose2.position.z)
        return height_diff <= self.step_height

    def check_turning_radius(self, pose1, pose2):
        # Check if turn is within turning radius constraints
        # Simplified check - actual implementation would be more complex
        distance = self.calculate_distance(pose1.position, pose2.position)
        return distance >= self.turning_radius
```

### Balance and Stability Considerations

```python
# Balance and stability for humanoid navigation
class HumanoidBalanceController:
    def __init__(self):
        self.zmp_reference = [0.0, 0.0]  # Zero Moment Point reference
        self.com_height = 0.8  # Center of mass height
        self.max_com_offset = 0.05  # Maximum COM offset for stability

    def check_balance_feasibility(self, path):
        # Check if navigation path maintains balance
        for pose in path:
            if not self.is_balance_feasible(pose):
                return False
        return True

    def is_balance_feasible(self, pose):
        # Check if pose is balance-feasible
        # Calculate ZMP (Zero Moment Point) for the pose
        zmp = self.calculate_zmp(pose)

        # Check if ZMP is within support polygon
        if self.is_zmp_in_support_polygon(zmp):
            return True
        else:
            return False

    def calculate_zmp(self, pose):
        # Calculate Zero Moment Point for given pose
        # This is a simplified calculation
        return [pose.position.x, pose.position.y]
```

## Integration with Perception Systems

### Sensor Integration

```python
# Integrating sensor data with navigation
class Nav2PerceptionIntegrator:
    def __init__(self):
        self.node = rclpy.create_node('nav2_perception_integrator')

        # Subscribe to sensor topics
        self.lidar_sub = self.node.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.camera_sub = self.node.create_subscription(
            Image, '/camera/depth', self.camera_callback, 10)
        self.imu_sub = self.node.create_subscription(
            Imu, '/imu', self.imu_callback, 10)

    def lidar_callback(self, msg):
        # Process LiDAR data for obstacle detection
        obstacles = self.process_lidar_data(msg)
        self.update_local_costmap(obstacles)

    def camera_callback(self, msg):
        # Process camera data for 3D obstacle detection
        obstacles_3d = self.process_camera_data(msg)
        self.update_voxel_layer(obstacles_3d)

    def imu_callback(self, msg):
        # Process IMU data for robot orientation
        self.update_robot_orientation(msg.orientation)
```

### Dynamic Obstacle Avoidance

```python
# Dynamic obstacle avoidance for humanoid robots
class DynamicObstacleAvoidance:
    def __init__(self):
        self.tracking_window = 2.0  # seconds to track dynamic obstacles
        self.prediction_horizon = 1.0  # seconds to predict obstacle movement
        self.safe_distance = 0.5  # meters from dynamic obstacles

    def predict_obstacle_path(self, obstacle_track):
        # Predict future position of dynamic obstacle
        if len(obstacle_track) < 2:
            return None

        # Calculate velocity from last two positions
        last_pos = obstacle_track[-1]
        prev_pos = obstacle_track[-2]

        dt = last_pos.timestamp - prev_pos.timestamp
        if dt > 0:
            velocity = (last_pos.position - prev_pos.position) / dt
            predicted_pos = last_pos.position + velocity * self.prediction_horizon
            return predicted_pos
        else:
            return last_pos.position

    def adjust_path_for_moving_obstacles(self, original_path, moving_obstacles):
        # Adjust navigation path to avoid moving obstacles
        adjusted_path = []

        for i, waypoint in enumerate(original_path):
            safe_waypoint = self.find_safe_waypoint(waypoint, moving_obstacles)
            adjusted_path.append(safe_waypoint)

        return adjusted_path

    def find_safe_waypoint(self, original_waypoint, moving_obstacles):
        # Find a safe waypoint near the original that avoids moving obstacles
        # Check if original waypoint is safe
        if self.is_waypoint_safe(original_waypoint, moving_obstacles):
            return original_waypoint

        # If not safe, find alternative nearby waypoints
        for offset in [0.1, -0.1, 0.2, -0.2]:  # Try small offsets
            alternative_waypoint = self.offset_waypoint(original_waypoint, offset)
            if self.is_waypoint_safe(alternative_waypoint, moving_obstacles):
                return alternative_waypoint

        # If no safe alternative found, return original (should trigger recovery)
        return original_waypoint
```

## Behavior Trees for Complex Navigation

### Custom Behavior Tree Nodes

```python
# Custom behavior tree node for humanoid navigation
from py_trees import behaviour, common
from py_trees.blackboard import Blackboard

class HumanoidNavigateToPose(behaviour.Behaviour):
    def __init__(self, name="HumanoidNavigateToPose"):
        super(HumanoidNavigateToPose, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, **kwargs):
        # Initialize navigation components
        self.nav_client = kwargs['nav_client']

    def initialise(self):
        # Set up navigation goal
        self.feedback_message = "initialising"

    def update(self):
        # Check if navigation is possible for humanoid
        if not self.is_humanoid_navigation_feasible():
            self.feedback_message = "navigation not feasible for humanoid"
            return common.Status.FAILURE

        # Check balance constraints
        if not self.is_balance_feasible():
            self.feedback_message = "balance constraints violated"
            return common.Status.FAILURE

        # Execute navigation
        nav_result = self.execute_navigation()

        if nav_result == "succeeded":
            self.feedback_message = "navigation succeeded"
            return common.Status.SUCCESS
        elif nav_result == "running":
            self.feedback_message = "navigating"
            return common.Status.RUNNING
        else:
            self.feedback_message = "navigation failed"
            return common.Status.FAILURE

    def is_humanoid_navigation_feasible(self):
        # Check if navigation is feasible for humanoid constraints
        return True  # Simplified check

    def is_balance_feasible(self):
        # Check if navigation maintains balance
        return True  # Simplified check

    def execute_navigation(self):
        # Execute navigation with humanoid-specific constraints
        return "running"  # Simplified
```

### Behavior Tree Configuration

```xml
<!-- Custom behavior tree for humanoid navigation -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <ReactiveSequence name="HumanoidNavigateToPose">
            <CheckHumanoidFeasibility/>
            <CheckBalanceConstraints/>
            <Sequence name="NavigationSequence">
                <ComputePathToPose goal="{goal}" path="{path}"/>
                <SmoothPath path="{path}" smoothed_path="{smoothed_path}"/>
                <FollowPath path="{smoothed_path}"/>
            </Sequence>
        </ReactiveSequence>
    </BehaviorTree>
</root>
```

## Advanced Navigation Features

### Multi-Goal Navigation

```python
# Multi-goal navigation for humanoid robots
class MultiGoalNavigator:
    def __init__(self):
        self.goals = []
        self.current_goal_index = 0
        self.path_cache = {}

    def set_goals(self, goal_list):
        # Set multiple navigation goals
        self.goals = goal_list
        self.current_goal_index = 0
        self.path_cache = {}

    def navigate_to_next_goal(self):
        # Navigate to the next goal in sequence
        if self.current_goal_index >= len(self.goals):
            return "completed_all_goals"

        current_goal = self.goals[self.current_goal_index]

        # Navigate to current goal
        result = self.navigate_to_pose(current_goal)

        if result == "succeeded":
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.goals):
                return "completed_all_goals"
            else:
                return "reached_goal"
        else:
            return result

    def navigate_to_all_goals(self):
        # Navigate through all goals in sequence
        results = []

        for i, goal in enumerate(self.goals):
            self.get_logger().info(f"Navigating to goal {i+1}/{len(self.goals)}")
            result = self.navigate_to_pose(goal)
            results.append(result)

            if result != "succeeded":
                self.get_logger().error(f"Failed to reach goal {i+1}")
                break

        return results
```

### Map Management and Localization

```python
# Map management for humanoid navigation
class HumanoidMapManager:
    def __init__(self):
        self.amcl_client = self.create_client(Localize, 'localize')
        self.map_client = self.create_client(GetMap, 'map_server/get_map')
        self.map_update_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)

    def initialize_localization(self):
        # Initialize AMCL for humanoid robot
        # Use humanoid-specific parameters
        amcl_params = {
            'use_map_topic': True,
            'scan_topic': '/scan',
            'odom_topic': '/odom',
            'base_frame_id': 'base_link',
            'global_frame_id': 'map',
            'transform_tolerance': 0.2,
            'recovery_alpha_slow': 0.001,
            'recovery_alpha_fast': 0.1,
            # Humanoid-specific parameters
            'min_particles': 500,  # More particles for stability
            'max_particles': 2000,
            'kld_err': 0.01,
            'kld_z': 0.99,
            'odom_alpha1': 0.05,  # Reduced for humanoid stability
            'odom_alpha2': 0.05,
            'odom_alpha3': 0.05,
            'odom_alpha4': 0.05,
            'odom_alpha5': 0.05,
            'laser_likelihood_max_dist': 2.0,
            'laser_max_range': 10.0,
            'laser_min_range': 0.1,
            'laser_model_type': 'likelihood_field',
            'update_min_d': 0.1,
            'update_min_a': 0.1,
            'resample_interval': 1,
            'transform_timeout': 0.2,
            'recovery_alpha_slow': 0.001,
            'recovery_alpha_fast': 0.1,
            'initial_pose': {
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'roll': 0.0,
                'pitch': 0.0,
                'yaw': 0.0
            },
            'initial_covariance': [0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.05]
        }

        # Set parameters
        self.set_parameters(amcl_params)

    def update_map_for_humanoid(self, new_map):
        # Update map with humanoid-specific considerations
        # Mark areas as impassable for humanoid (e.g., stairs, narrow passages)
        humanoid_map = self.filter_map_for_humanoid(new_map)
        return humanoid_map

    def filter_map_for_humanoid(self, original_map):
        # Filter map to account for humanoid capabilities
        # Mark areas that are impassable for humanoid robots
        filtered_map = original_map

        # Example: Mark areas with high step height as impassable
        # This would involve analyzing elevation data if available

        return filtered_map
```

## Performance Optimization

### Navigation Performance Tuning

```python
# Performance optimization for humanoid navigation
class Nav2PerformanceOptimizer:
    def __init__(self):
        self.performance_metrics = {
            'planning_time': [],
            'execution_time': [],
            'path_length': [],
            'success_rate': 0.0
        }

    def optimize_planning_frequency(self):
        # Adjust planning frequency based on environment complexity
        if self.is_environment_complex():
            # Reduce planning frequency in complex environments
            self.set_planning_frequency(0.5)  # Plan less frequently
        else:
            # Increase planning frequency in simple environments
            self.set_planning_frequency(2.0)  # Plan more frequently

    def optimize_costmap_resolution(self):
        # Adjust costmap resolution based on navigation requirements
        # Higher resolution for precise navigation, lower for performance
        if precise_navigation_needed:
            self.set_costmap_resolution(0.025)  # Higher resolution
        else:
            self.set_costmap_resolution(0.1)    # Lower resolution for performance

    def optimize_local_planner(self):
        # Optimize local planner parameters for humanoid
        params = {
            'max_vel_x': self.get_optimal_linear_velocity(),
            'min_vel_x': 0.1,
            'max_vel_theta': self.get_optimal_angular_velocity(),
            'min_vel_theta': 0.1,
            'acc_lim_x': 0.5,  # Acceleration limits for humanoid stability
            'acc_lim_theta': 1.0,
            'decel_lim_x': -0.5,
            'decel_lim_theta': -1.0,
            'xy_goal_tolerance': 0.2,  # Larger tolerance for humanoid
            'yaw_goal_tolerance': 0.1,
            'trans_stopped_vel': 0.1,
            'rot_stopped_vel': 0.1
        }

        self.set_local_planner_params(params)

    def get_optimal_linear_velocity(self):
        # Calculate optimal linear velocity based on terrain and stability
        if self.is_rough_terrain():
            return 0.2  # Slower for stability
        else:
            return 0.5  # Faster on smooth terrain

    def get_optimal_angular_velocity(self):
        # Calculate optimal angular velocity based on balance constraints
        return 0.5  # Conservative for humanoid balance
```

## Safety and Recovery Behaviors

### Safety Checks

```python
# Safety checks for humanoid navigation
class HumanoidNavigationSafety:
    def __init__(self):
        self.max_tilt_angle = 15.0  # Maximum tilt in degrees
        self.battery_threshold = 0.2  # 20% battery threshold
        self.emergency_stop_distance = 0.3  # Stop distance for obstacles

    def perform_safety_check(self):
        # Perform comprehensive safety check before navigation
        checks = [
            self.check_balance(),
            self.check_battery_level(),
            self.check_obstacle_distance(),
            self.check_joint_limits(),
            self.check_communication()
        ]

        return all(checks)

    def check_balance(self):
        # Check if robot is within balance limits
        current_tilt = self.get_current_tilt()
        return abs(current_tilt) < self.max_tilt_angle

    def check_battery_level(self):
        # Check if battery level is sufficient for navigation
        battery_level = self.get_battery_level()
        return battery_level > self.battery_threshold

    def check_obstacle_distance(self):
        # Check for immediate obstacles
        min_distance = self.get_min_obstacle_distance()
        return min_distance > self.emergency_stop_distance

    def emergency_stop(self):
        # Emergency stop for safety
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)
        return True
```

### Recovery Behaviors

```python
# Custom recovery behaviors for humanoid robots
class HumanoidRecoveryBehaviors:
    def __init__(self):
        self.recovery_behaviors = {
            'stuck_recovery': self.stuck_recovery,
            'obstacle_recovery': self.obstacle_recovery,
            'balance_recovery': self.balance_recovery
        }

    def stuck_recovery(self):
        # Recovery behavior when robot is stuck
        # For humanoid, this might involve small stepping motions
        self.get_logger().info("Attempting stuck recovery")

        # Small backup motion
        backup_cmd = Twist()
        backup_cmd.linear.x = -0.1
        self.cmd_vel_pub.publish(backup_cmd)
        self.get_clock().sleep_for(Duration(seconds=2.0))

        # Stop
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Try different path
        return "recovery_completed"

    def obstacle_recovery(self):
        # Recovery when obstacle is detected
        # For humanoid, this might involve stepping over or around
        self.get_logger().info("Attempting obstacle recovery")

        # Analyze obstacle characteristics
        obstacle_info = self.analyze_obstacle()

        if obstacle_info['height'] < 0.1:  # Can step over
            return self.step_over_obstacle(obstacle_info)
        else:  # Must go around
            return self.go_around_obstacle(obstacle_info)

    def balance_recovery(self):
        # Recovery when balance is compromised
        self.get_logger().info("Attempting balance recovery")

        # Stop all motion
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        # Adjust posture for balance
        self.adjust_posture_for_balance()

        # Wait for stability
        self.get_clock().sleep_for(Duration(seconds=1.0))

        return "balance_recovered"
```

## Example: Complete Navigation Setup

Here's a complete example of setting up navigation for a humanoid robot:

```python
# complete_humanoid_navigation.py
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Imu
import rclpy.action

class CompleteHumanoidNavigation(Node):
    def __init__(self):
        super().__init__('complete_humanoid_navigation')

        # Initialize navigation action client
        self.nav_client = rclpy.action.ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # Initialize safety systems
        self.safety_system = HumanoidNavigationSafety()
        self.recovery_system = HumanoidRecoveryBehaviors()

        # Subscribe to sensors
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialize humanoid-specific components
        self.kinematic_constraints = HumanoidKinematicConstraints()
        self.balance_controller = HumanoidBalanceController()

        self.get_logger().info('Complete humanoid navigation system initialized')

    def navigate_to_pose(self, x, y, theta):
        # Navigate to specified pose with humanoid constraints
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = theta

        # Check humanoid feasibility before sending goal
        if not self.is_humanoid_navigation_feasible(goal_msg.pose):
            self.get_logger().error('Navigation not feasible for humanoid robot')
            return False

        # Check safety before navigation
        if not self.safety_system.perform_safety_check():
            self.get_logger().error('Safety check failed')
            return False

        # Send navigation goal
        self.nav_client.wait_for_server()
        future = self.nav_client.send_goal_async(goal_msg)

        return future

    def is_humanoid_navigation_feasible(self, pose):
        # Check if navigation to pose is feasible for humanoid
        # This would involve checking kinematic constraints, balance, etc.
        return True

    def lidar_callback(self, msg):
        # Process LiDAR data for obstacle detection
        # Update local costmap with obstacle information
        pass

    def imu_callback(self, msg):
        # Process IMU data for balance monitoring
        # Update balance controller
        pass

def main(args=None):
    rclpy.init(args=args)
    navigation_node = CompleteHumanoidNavigation()

    try:
        # Example: Navigate to a specific pose
        future = navigation_node.navigate_to_pose(5.0, 5.0, 0.0)

        if future:
            rclpy.spin(navigation_node)
        else:
            navigation_node.get_logger().error('Failed to send navigation goal')
    except KeyboardInterrupt:
        pass
    finally:
        navigation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with Isaac ROS

### Isaac ROS Perception Integration

```python
# Integration with Isaac ROS for enhanced perception
class IsaacROSNavIntegration:
    def __init__(self):
        # Initialize Isaac ROS perception nodes
        self.apriltag_detector = self.initialize_apriltag_detector()
        self.stereo_dnn = self.initialize_stereo_dnn()

        # Integrate with Nav2 costmaps
        self.enhanced_costmap = self.create_enhanced_costmap()

    def initialize_apriltag_detector(self):
        # Initialize Isaac ROS Apriltag for precise localization
        # Use GPU acceleration for real-time detection
        pass

    def initialize_stereo_dnn(self):
        # Initialize Isaac ROS Stereo DNN for obstacle detection
        # Use deep learning for semantic obstacle classification
        pass

    def create_enhanced_costmap(self):
        # Create costmap with enhanced perception data
        # Combine traditional sensors with Isaac ROS perception
        enhanced_costmap = {
            'traditional_layer': self.create_traditional_costmap(),
            'semantic_layer': self.create_semantic_costmap(),
            'dynamic_layer': self.create_dynamic_costmap()
        }

        return enhanced_costmap
```

## Troubleshooting and Best Practices

### Common Issues

1. **Localization Problems**: Ensure proper map quality and sensor calibration
2. **Path Planning Failures**: Check costmap configuration and robot footprint
3. **Oscillation**: Adjust controller parameters and increase tolerances
4. **Performance Issues**: Optimize costmap resolution and update frequencies

### Best Practices

1. **Thorough Testing**: Test navigation in simulation before real-world deployment
2. **Parameter Tuning**: Carefully tune parameters for your specific humanoid platform
3. **Safety First**: Implement comprehensive safety checks and emergency stops
4. **Modular Design**: Keep navigation components modular for easy maintenance
5. **Documentation**: Document all custom configurations and parameters

## Summary

This chapter covered the fundamentals of ROS 2 Navigation 2 (Nav2) for humanoid robots:

- Nav2 architecture and core components
- Costmap configuration for humanoid robots
- Path planning algorithms (A*, Dijkstra, local planners)
- Humanoid-specific kinematic constraints
- Integration with perception systems
- Behavior trees for complex navigation
- Advanced features (multi-goal, map management)
- Performance optimization techniques
- Safety and recovery behaviors
- Isaac ROS integration

Nav2 provides a robust framework for autonomous navigation that can be adapted for the unique requirements of humanoid robots, including balance constraints and complex kinematics.

## Learning Check

After completing this chapter, you should be able to:
- Configure Nav2 for humanoid robot platforms
- Implement path planning algorithms with humanoid constraints
- Integrate perception systems with navigation
- Create custom behavior trees for complex navigation tasks
- Optimize navigation performance for humanoid robots
- Implement safety and recovery behaviors
- Troubleshoot common navigation issues
- Design complete navigation systems for humanoid applications