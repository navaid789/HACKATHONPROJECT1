---
title: Curriculum and Course Sequence Recommendations
description: Recommended course sequences and dependencies for Physical AI & Humanoid Robotics
---

# Curriculum and Course Sequence Recommendations

## Overview

This document provides educators with recommended course sequences and dependency information for teaching the Physical AI & Humanoid Robotics textbook. The recommendations are based on the logical progression of concepts and technical dependencies between chapters.

## Module Dependencies

### Module 1: The Robotic Nervous System (ROS 2)
- **Prerequisites**: None (can be taught independently)
- **Dependencies**: None
- **Follows**: Standalone module
- **Enables**: All subsequent modules

**Chapter Sequence:**
1. Chapter 1: ROS 2 Fundamentals and Architecture (45 minutes)
2. Chapter 2: Nodes, Topics, and Services (60 minutes)
3. Chapter 3: URDF for Humanoid Robots (75 minutes)

### Module 2: The Digital Twin (Gazebo & Unity)
- **Prerequisites**: Module 1 (especially Chapters 1 and 2)
- **Dependencies**: ROS 2 knowledge for simulation integration
- **Follows**: Module 1
- **Enables**: Advanced simulation concepts in Modules 3 and 4

**Chapter Sequence:**
1. Chapter 4: Gazebo Simulation Fundamentals (90 minutes)
2. Chapter 5: Unity Integration and High-Fidelity Rendering (120 minutes)
3. Chapter 6: Sensor Simulation and Integration (90 minutes)

### Module 3: The AI-Robot Brain (NVIDIA Isaac™)
- **Prerequisites**: Modules 1 and 2
- **Dependencies**: ROS 2 communication, simulation environments
- **Follows**: Module 2
- **Enables**: Vision-Language-Action systems in Module 4

**Chapter Sequence:**
1. Chapter 7: Isaac Sim for Physical AI (120 minutes)
2. Chapter 8: Isaac ROS for GPU-Accelerated Perception (120 minutes)
3. Chapter 9: Navigation (Nav2) for Humanoid Robots (120 minutes)

### Module 4: Vision-Language-Action (VLA)
- **Prerequisites**: All previous modules
- **Dependencies**: Full stack knowledge (ROS 2, simulation, AI integration)
- **Follows**: Module 3
- **Enables**: Complete Physical AI systems

**Chapter Sequence:**
1. Chapter 10: Vision-Language-Action (VLA) Systems (150 minutes)

## Recommended Course Sequences

### Full Course Sequence (Recommended)
This sequence covers all content in the recommended order:

1. **Module 1: The Robotic Nervous System**
   - Chapter 1: ROS 2 Fundamentals and Architecture
   - Chapter 2: Nodes, Topics, and Services
   - Chapter 3: URDF for Humanoid Robots

2. **Module 2: The Digital Twin**
   - Chapter 4: Gazebo Simulation Fundamentals
   - Chapter 5: Unity Integration and High-Fidelity Rendering
   - Chapter 6: Sensor Simulation and Integration

3. **Module 3: The AI-Robot Brain**
   - Chapter 7: Isaac Sim for Physical AI
   - Chapter 8: Isaac ROS for GPU-Accelerated Perception
   - Chapter 9: Navigation (Nav2) for Humanoid Robots

4. **Module 4: Vision-Language-Action**
   - Chapter 10: Vision-Language-Action (VLA) Systems

### Accelerated Sequence (Advanced Students)
For students with prior robotics experience:

1. Module 1, Chapters 1-2 (condensed)
2. Module 2, Chapters 4-6 (focused on simulation)
3. Module 3, Chapters 7-9 (focused on AI integration)
4. Module 4, Chapter 10 (VLA systems)

### Modular Sequence (Flexible Implementation)
Each module can be taught independently with the following prerequisites:

- **Module 1 Only**: No prerequisites
- **Module 2 Only**: Basic programming knowledge
- **Module 3 Only**: ROS 2 knowledge (Chapter 1-2) + Simulation basics (Chapter 4)
- **Module 4 Only**: Complete knowledge of all previous modules

## Prerequisites by Chapter

### Chapter 1: ROS 2 Fundamentals and Architecture
- **Prerequisites**: Basic programming knowledge (Python/C++)
- **Learning Outcomes**: Understanding of ROS 2 architecture and communication
- **Time Required**: 45 minutes
- **Resources Needed**: Computer with ROS 2 installed

### Chapter 2: Nodes, Topics, and Services
- **Prerequisites**: Chapter 1 (ROS 2 fundamentals)
- **Learning Outcomes**: Ability to create ROS 2 nodes and communication patterns
- **Time Required**: 60 minutes
- **Resources Needed**: ROS 2 environment, basic Python/C++ knowledge

### Chapter 3: URDF for Humanoid Robots
- **Prerequisites**: Chapter 1, basic understanding of 3D geometry
- **Learning Outcomes**: Ability to create robot models in URDF
- **Time Required**: 75 minutes
- **Resources Needed**: URDF validation tools, 3D visualization

### Chapter 4: Gazebo Simulation Fundamentals
- **Prerequisites**: Chapters 1-2 (ROS 2 communication), Chapter 3 (URDF)
- **Learning Outcomes**: Ability to create simulation environments
- **Time Required**: 90 minutes
- **Resources Needed**: Gazebo installation, physics simulation understanding

### Chapter 5: Unity Integration and High-Fidelity Rendering
- **Prerequisites**: Chapter 4 (simulation concepts), basic C# knowledge
- **Learning Outcomes**: Integration of Unity with robotics systems
- **Time Required**: 120 minutes
- **Resources Needed**: Unity installation, graphics hardware

### Chapter 6: Sensor Simulation and Integration
- **Prerequisites**: Chapters 4-5 (simulation), Chapter 2 (ROS communication)
- **Learning Outcomes**: Implementation of sensor simulation and fusion
- **Time Required**: 90 minutes
- **Resources Needed**: Simulation environment, sensor models

### Chapter 7: Isaac Sim for Physical AI
- **Prerequisites**: Chapters 1-4 (ROS 2, simulation fundamentals)
- **Learning Outcomes**: Advanced simulation for AI training
- **Time Required**: 120 minutes
- **Resources Needed**: NVIDIA GPU, Isaac Sim installation

### Chapter 8: Isaac ROS for GPU-Accelerated Perception
- **Prerequisites**: Chapters 1-2 (ROS 2), Chapter 7 (Isaac Sim)
- **Learning Outcomes**: GPU-accelerated perception systems
- **Time Required**: 120 minutes
- **Resources Needed**: NVIDIA GPU, Isaac ROS packages

### Chapter 9: Navigation (Nav2) for Humanoid Robots
- **Prerequisites**: Chapters 1-3 (ROS 2, URDF), Chapters 4-6 (simulation, sensors)
- **Learning Outcomes**: Autonomous navigation systems
- **Time Required**: 120 minutes
- **Resources Needed**: Navigation simulation environment

### Chapter 10: Vision-Language-Action (VLA) Systems
- **Prerequisites**: All previous chapters
- **Learning Outcomes**: Complete VLA systems for embodied AI
- **Time Required**: 150 minutes
- **Resources Needed**: Full technology stack, AI training environment

## Suggested Learning Pathways

### Pathway 1: Simulation Focus
For educators focusing on simulation and digital twin technologies:
- Modules 1 and 2 (recommended sequence)
- Optional: Chapter 7 (Isaac Sim)
- Optional: Chapter 10 (VLA for simulation applications)

### Pathway 2: AI Integration Focus
For educators focusing on AI and perception systems:
- Module 1 (ROS 2 foundation)
- Module 2, Chapter 4 (simulation for AI training)
- Module 3 (AI integration)
- Module 4 (VLA systems)

### Pathway 3: Hardware Integration Focus
For educators with access to physical robots:
- Module 1 (ROS 2 for hardware control)
- Module 2 (simulation for testing)
- Module 3 (navigation and perception)
- Select topics from Module 4 (VLA for human-robot interaction)

## Assessment Recommendations

### Module 1 Assessment
- Quiz on ROS 2 architecture concepts
- Simple node implementation exercise
- URDF validation task

### Module 2 Assessment
- Simulation environment creation
- Sensor integration in simulation
- ROS 2 communication between simulated components

### Module 3 Assessment
- AI model training in simulation
- Perception system implementation
- Navigation task completion

### Module 4 Assessment
- Complete VLA system implementation
- Natural language command execution
- Integration of all components

## Time Management Suggestions

### Class Period Structure (90 minutes)
- 15 minutes: Review previous concepts
- 45 minutes: New content delivery
- 20 minutes: Hands-on practice
- 10 minutes: Q&A and wrap-up

### Lab Session Structure (3 hours)
- 30 minutes: Setup and environment configuration
- 90 minutes: Guided practice and implementation
- 60 minutes: Independent project work

### Course Duration Options

#### Intensive (2 weeks)
- 6-8 hours per day
- 4-5 chapters per day
- Heavy emphasis on hands-on practice

#### Standard (8 weeks)
- 3-4 hours per week
- 1-2 chapters per week
- Balanced theory and practice

#### Extended (16 weeks)
- 1.5-2 hours per week
- 1 chapter every 1-2 weeks
- Emphasis on deep understanding and project work

## Prerequisites Assessment

Before starting Module 1, students should demonstrate:
- Basic programming competency (Python preferred)
- Understanding of basic robotics concepts
- Familiarity with Linux command line
- Access to required hardware/software

## Technical Prerequisites

### Software Requirements
- ROS 2 Humble Hawksbill
- Gazebo Garden or newer
- Unity (Personal or Pro)
- NVIDIA Isaac Sim and Isaac ROS
- Development environment (VS Code recommended)

### Hardware Requirements
- Modern computer (8+ cores, 16GB+ RAM)
- NVIDIA GPU for Isaac components
- Internet access for package installation

## Accessibility Considerations

### For Students with Different Learning Needs
- Provide multiple modalities (visual, textual, hands-on)
- Offer flexible pacing options
- Include alternative examples and explanations
- Provide additional support resources

### For Remote Learning
- Emphasize simulation components over hardware
- Provide cloud-based access options where possible
- Include detailed documentation and video resources
- Enable remote lab access where feasible