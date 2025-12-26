---
title: Chapter 3 - URDF for Humanoid Robots
description: "Understanding Unified Robot Description Format (URDF) for humanoid robot modeling"
module: 1
chapter: 3
learning_objectives:
  - Create URDF files for humanoid robot models
  - Define robot links, joints, and physical properties
  - Understand the structure and syntax of URDF
  - Apply URDF to humanoid robot kinematics
difficulty: intermediate
estimated_time: 75
tags:
  - ros2
  - urdf
  - humanoid
  - modeling
  - kinematics

authors:
  - Textbook Team
prerequisites:
  - Chapter 1: ROS 2 Fundamentals and Architecture
  - Chapter 2: Nodes, Topics, and Services
  - Basic understanding of 3D geometry and physics
---

# Chapter 3: URDF for Humanoid Robots

## Introduction

Unified Robot Description Format (URDF) is the standard format for representing robot models in ROS. In this chapter, we'll explore how to create URDF files specifically for humanoid robots, covering the essential components, structure, and best practices for modeling complex articulated robots.

Humanoid robots present unique challenges in URDF modeling due to their complex kinematic structure with multiple degrees of freedom, symmetrical limbs, and intricate joint arrangements. Understanding how to properly model these robots is crucial for simulation, visualization, and control.

## What is URDF?

URDF (Unified Robot Description Format) is an XML-based format that describes robot models in ROS. It defines the physical and visual properties of a robot, including its links (rigid parts), joints (connections between links), and materials. URDF files are essential for:
- Robot simulation in Gazebo
- Robot visualization in RViz
- Kinematic analysis and planning
- Robot calibration and control

## URDF Structure for Humanoid Robots

A humanoid robot URDF typically follows this structure:

- **Base Link**: Usually the pelvis or torso
- **Upper Body**: Torso, head, arms
- **Lower Body**: Legs and feet
- **Sensors**: Cameras, IMUs, etc.

### Basic URDF Components

Let's examine the fundamental elements of a humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.3"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </joint>

  <!-- Torso Link -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>
</robot>
```

## Link Definition

A link represents a rigid part of the robot. Each link must have:

1. **Visual**: How the link appears in visualization
2. **Collision**: How the link interacts in simulation
3. **Inertial**: Physical properties for dynamics

### Visual Properties
```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Choose one: box, cylinder, sphere, or mesh -->
    <box size="0.1 0.1 0.1"/>
    <!-- OR -->
    <cylinder radius="0.05" length="0.2"/>
    <!-- OR -->
    <sphere radius="0.05"/>
    <!-- OR -->
    <mesh filename="package://humanoid_description/meshes/shoulder.dae"/>
  </geometry>
  <material name="blue"/>
</visual>
```

### Collision Properties
```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Same geometry types as visual -->
    <box size="0.1 0.1 0.1"/>
  </geometry>
</collision>
```

### Inertial Properties
```xml
<inertial>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <mass value="1.0"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

## Joint Definition

Joints connect links and define their relative motion. For humanoid robots, common joint types include:

1. **Revolute**: Rotational joint with limits
2. **Continuous**: Rotational joint without limits
3. **Prismatic**: Linear joint with limits
4. **Fixed**: No movement (rigid connection)

### Joint Types for Humanoid Robots

```xml
<!-- Revolute joint for shoulder pitch -->
<joint name="l_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="l_upper_arm"/>
  <origin xyz="0.1 0.15 0.1" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>

<!-- Continuous joint for head rotation -->
<joint name="head_yaw" type="continuous">
  <parent link="neck"/>
  <child link="head"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.1"/>
</joint>

<!-- Fixed joint for sensor mount -->
<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.05" rpy="0 0 0"/>
</joint>
```

## Humanoid Robot Kinematic Chain

Humanoid robots typically have a complex kinematic structure. Here's an example of how to model an arm:

```xml
<!-- Left Arm Chain -->
<!-- Upper Arm -->
<link name="l_upper_arm">
  <visual>
    <geometry>
      <cylinder radius="0.04" length="0.3"/>
    </geometry>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <material name="white"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.04" length="0.3"/>
    </geometry>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
  </collision>
  <inertial>
    <mass value="1.5"/>
    <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
  </inertial>
</link>

<!-- Shoulder Joint -->
<joint name="l_shoulder_joint" type="revolute">
  <parent link="torso"/>
  <child link="l_upper_arm"/>
  <origin xyz="0.1 0.1 0.2" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
</joint>

<!-- Lower Arm -->
<link name="l_lower_arm">
  <visual>
    <geometry>
      <cylinder radius="0.03" length="0.25"/>
    </geometry>
    <origin xyz="0 0 0.125" rpy="0 0 0"/>
    <material name="white"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.03" length="0.25"/>
    </geometry>
    <origin xyz="0 0 0.125" rpy="0 0 0"/>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
  </inertial>
</link>

<!-- Elbow Joint -->
<joint name="l_elbow_joint" type="revolute">
  <parent link="l_upper_arm"/>
  <child link="l_lower_arm"/>
  <origin xyz="0 0 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="3.14" effort="30" velocity="2"/>
</joint>

<!-- Hand -->
<link name="l_hand">
  <visual>
    <geometry>
      <box size="0.1 0.08 0.05"/>
    </geometry>
    <material name="black"/>
  </visual>
  <collision>
    <geometry>
      <box size="0.1 0.08 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.3"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
  </inertial>
</link>

<!-- Wrist Joint -->
<joint name="l_wrist_joint" type="revolute">
  <parent link="l_lower_arm"/>
  <child link="l_hand"/>
  <origin xyz="0 0 0.25" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10" velocity="3"/>
</joint>
```

## Using Xacro for Complex Models

Xacro (XML Macros) helps simplify complex URDF models by allowing macros, properties, and mathematical expressions:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="PI" value="3.1415926535897931"/>
  <xacro:property name="arm_length" value="0.3"/>
  <xacro:property name="arm_radius" value="0.04"/>

  <!-- Macro for arm -->
  <xacro:macro name="arm" params="side parent xyz">
    <!-- Upper Arm -->
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder radius="${arm_radius}" length="${arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${arm_length/2}" rpy="0 0 0"/>
        <material name="white"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${arm_radius}" length="${arm_length}"/>
        </geometry>
        <origin xyz="0 0 ${arm_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-${PI/2}" upper="${PI/2}" effort="50" velocity="2"/>
    </joint>
  </xacro:macro>

  <!-- Using the macro -->
  <xacro:arm side="l" parent="torso" xyz="0.1 0.15 0.2"/>
  <xacro:arm side="r" parent="torso" xyz="0.1 -0.15 0.2"/>
</robot>
```

## Sensors in Humanoid Robots

Humanoid robots often have various sensors integrated into their structure:

```xml
<!-- IMU Sensor -->
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.05" rpy="0 0 0"/>
</joint>

<!-- Camera Sensor -->
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.03 0.01"/>
    </geometry>
    <material name="black"/>
  </visual>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>
```

## Best Practices for Humanoid URDF

1. **Symmetry**: Use Xacro macros to avoid duplicating symmetrical parts (left/right arms/legs)
2. **Realistic Inertias**: Calculate or estimate realistic inertial properties
3. **Consistent Naming**: Use consistent naming conventions (e.g., l_upper_arm, r_upper_arm)
4. **Proper Origins**: Ensure joint origins are correctly positioned
5. **Collision vs Visual**: Use simplified collision geometry for better simulation performance
6. **Kinematic Loops**: For closed chains (like when a robot grasps an object), consider using transmission elements

## Validation and Testing

After creating your URDF, it's important to validate it:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Visualize the robot
ros2 run rviz2 rviz2

# Use the robot_state_publisher to publish transforms
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat robot.urdf)'
```

## Example: Complete Humanoid Robot URDF

Here's a simplified example of a complete humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="PI" value="3.1415926535897931"/>

  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>

  <!-- Base/Pelvis Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.175" rpy="0 0 0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="${-PI/4}" upper="${PI/4}" effort="10" velocity="2"/>
  </joint>

  <!-- Include arms using macro -->
  <xacro:include filename="arm.urdf.xacro"/>
  <xacro:left_arm parent="torso" xyz="0.1 0.1 0.1" rpy="0 0 0"/>
  <xacro:right_arm parent="torso" xyz="0.1 -0.1 0.1" rpy="0 0 0"/>

  <!-- Include legs using macro -->
  <xacro:include filename="leg.urdf.xacro"/>
  <xacro:left_leg parent="base_link" xyz="0 0.05 -0.075" rpy="0 0 0"/>
  <xacro:right_leg parent="base_link" xyz="0 -0.05 -0.075" rpy="0 0 0"/>
</robot>
```

## Summary

This chapter covered the fundamentals of creating URDF files for humanoid robots, including:

- Basic URDF structure and components
- Link and joint definitions
- Visual, collision, and inertial properties
- Xacro macros for simplifying complex models
- Sensor integration
- Best practices for humanoid robot modeling

Proper URDF modeling is essential for successful simulation, visualization, and control of humanoid robots in ROS.

## Learning Check

After completing this chapter, you should be able to:
- Create URDF files for humanoid robot models
- Define links, joints, and their properties
- Use Xacro to simplify complex robot descriptions
- Integrate sensors into robot models
- Apply best practices for humanoid robot URDF modeling