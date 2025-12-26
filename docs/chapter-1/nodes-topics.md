---
title: "Chapter 2 - Nodes, Topics, and Services"
description: "Understanding ROS 2 communication patterns: nodes, topics, and services"
module: 1
chapter: 2
learning_objectives:
  - Implement ROS 2 nodes for different functionalities
  - Create and use topics for publish-subscribe communication
  - Design and implement services for request-response communication
  - Understand the differences between topics and services
difficulty: intermediate
estimated_time: 60
tags:
  - ros2
  - nodes
  - topics
  - services
  - communication

authors:
  - Textbook Team
prerequisites:
  - Chapter 1: ROS 2 Fundamentals and Architecture
  - Basic understanding of Python or C++
---

# Chapter 2: Nodes, Topics, and Services

## Introduction

In this chapter, we'll explore the fundamental communication patterns in ROS 2: nodes, topics, and services. These building blocks form the backbone of any ROS 2 system, enabling different components to communicate and coordinate with each other.

Understanding these communication patterns is crucial for designing modular, maintainable, and scalable robotic systems. We'll cover how to create nodes, implement publish-subscribe communication with topics, and provide request-response communication with services.

## Nodes in ROS 2

A node is a fundamental building block of a ROS 2 system that performs computation. Nodes are organized into packages to make sharing of code easier. Each node is designed to perform a specific task and can communicate with other nodes through topics, services, actions, and parameters.

### Creating a Node

Let's look at how to create a node in both Python and C++:

**Python Example:**
```python
import rclpy
from rclpy.node import Node

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info('Robot Controller Node has been started')

        # Initialize any required components here
        self.robot_position = [0.0, 0.0, 0.0]

    def update_position(self, x, y, z):
        self.robot_position = [x, y, z]
        self.get_logger().info(f'Robot position updated: {self.robot_position}')

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()

    # Perform some operations
    node.update_position(1.0, 2.0, 0.0)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**C++ Example:**
```cpp
#include "rclcpp/rclcpp.hpp"

class RobotControllerNode : public rclcpp::Node
{
public:
    RobotControllerNode() : Node("robot_controller")
    {
        RCLCPP_INFO(this->get_logger(), "Robot Controller Node has been started");

        // Initialize any required components here
        robot_position_ = {0.0, 0.0, 0.0};
    }

    void update_position(double x, double y, double z) {
        robot_position_ = {x, y, z};
        RCLCPP_INFO(this->get_logger(),
                   "Robot position updated: [%f, %f, %f]",
                   robot_position_[0], robot_position_[1], robot_position_[2]);
    }

private:
    std::vector<double> robot_position_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotControllerNode>());
    rclcpp::shutdown();
    return 0;
}
```

## Topics - Publish-Subscribe Communication

Topics implement a publish-subscribe communication pattern where publishers send messages to a topic and subscribers receive messages from that topic. This is an asynchronous, one-way communication method.

### Creating Publishers and Subscribers

**Publisher Example:**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher_ = self.create_publisher(String, 'sensor_data', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Sensor reading: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()
    rclpy.spin(sensor_publisher)
    sensor_publisher.destroy_node()
    rclpy.shutdown()
```

**Subscriber Example:**
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DataProcessor(Node):
    def __init__(self):
        super().__init__('data_processor')
        self.subscription = self.create_subscription(
            String,
            'sensor_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')
        # Process the received data
        processed_data = msg.data.upper()
        self.get_logger().info(f'Processed: "{processed_data}"')

def main(args=None):
    rclpy.init(args=args)
    data_processor = DataProcessor()
    rclpy.spin(data_processor)
    data_processor.destroy_node()
    rclpy.shutdown()
```

### Message Types

ROS 2 provides a rich set of standard message types defined in ROS interfaces (rosidl). Common message types include:

- **std_msgs**: Basic data types (Int8, Float64, String, etc.)
- **geometry_msgs**: Geometric primitives (Point, Pose, Twist, etc.)
- **sensor_msgs**: Sensor data (LaserScan, Image, JointState, etc.)
- **nav_msgs**: Navigation messages (Odometry, Path, OccupancyGrid, etc.)

## Services - Request-Response Communication

Services implement a synchronous request-response communication pattern. A client sends a request to a service server, which processes the request and sends back a response.

### Creating Services

**Service Definition (example.srv):**
```
# Request
string name
int64 age
---
# Response
bool success
string message
```

**Service Server:**
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()
```

**Service Client:**
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()
```

## Advanced Topic Features

### Quality of Service (QoS) Configuration

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Configure QoS for a publisher
qos_profile = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST
)

publisher = node.create_publisher(String, 'topic', qos_profile)
```

### Latching Topics

Latching allows the last message published on a topic to be stored and sent to new subscribers:

```python
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

qos_profile = QoSProfile(
    depth=10,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL  # Latching equivalent
)
latching_publisher = node.create_publisher(String, 'latched_topic', qos_profile)
```

## Best Practices for Communication Design

### Node Design Principles

1. **Single Responsibility**: Each node should have a single, well-defined purpose
2. **Modularity**: Design nodes to be reusable and replaceable
3. **Error Handling**: Implement proper error handling and logging
4. **Configuration**: Use parameters for configurable behavior

### Topic Design Guidelines

1. **Naming Convention**: Use descriptive, consistent names (e.g., `/robot/sensors/laser_scan`)
2. **Message Frequency**: Consider the impact of message rate on system performance
3. **Data Size**: Be mindful of message size to avoid network congestion
4. **QoS Selection**: Choose appropriate QoS settings based on application requirements

### Service Design Guidelines

1. **Use Cases**: Use services for operations that have a clear request-response pattern
2. **Response Time**: Services should respond quickly; long operations should use actions
3. **Statelessness**: Design services to be stateless when possible
4. **Error Handling**: Provide meaningful error responses

## Practical Example: Robot Control System

Let's look at a practical example combining nodes, topics, and services:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from example_interfaces.srv import Trigger

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_publisher = self.create_publisher(String, '/robot_status', 10)

        # Subscribers
        self.odom_subscription = self.create_subscription(
            String, '/odom', self.odom_callback, 10)

        # Services
        self.move_service = self.create_service(
            Trigger, '/move_robot', self.move_robot_callback)

        self.robot_status = "IDLE"

    def odom_callback(self, msg):
        self.get_logger().info(f'Odometry: {msg.data}')

    def move_robot_callback(self, request, response):
        # Send movement command
        cmd = Twist()
        cmd.linear.x = 1.0  # Move forward
        cmd.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd)

        # Update status
        self.robot_status = "MOVING"
        status_msg = String()
        status_msg.data = self.robot_status
        self.status_publisher.publish(status_msg)

        response.success = True
        response.message = "Robot movement command sent"
        return response

def main(args=None):
    rclpy.init(args=args)
    robot_control_node = RobotControlNode()
    rclpy.spin(robot_control_node)
    robot_control_node.destroy_node()
    rclpy.shutdown()
```

## Summary

This chapter covered the core communication patterns in ROS 2:

- **Nodes**: The fundamental computational units in ROS 2
- **Topics**: Asynchronous publish-subscribe communication
- **Services**: Synchronous request-response communication

These patterns enable the development of modular, distributed robotic systems where different components can be developed and maintained independently.

## Learning Check

After completing this chapter, you should be able to:
- Create and implement ROS 2 nodes
- Design and use topics for publish-subscribe communication
- Implement services for request-response communication
- Apply appropriate Quality of Service settings
- Follow best practices for communication design