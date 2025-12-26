# Data Model: Physical AI & Humanoid Robotics Textbook

**Date**: 2025-12-26 | **Feature**: 001-docusaurus-deployment | **Status**: Draft

## Overview

This document defines the data model for the Physical AI & Humanoid Robotics textbook deployed using Docusaurus. The model represents the content structure, metadata, and relationships between different elements of the educational content.

## Core Entities

### 1. Textbook Chapter
**Definition**: The primary organizational unit of the textbook, representing one of the 10 planned chapters.

**Attributes**:
- `id`: Unique identifier (e.g., "chapter-1-ros-fundamentals")
- `title`: Display title of the chapter
- `slug`: URL-friendly identifier
- `module`: Source module (1-4) that this chapter derives from
- `number`: Chapter number (1-10)
- `description`: Brief description of chapter content
- `learningObjectives`: Array of learning objectives for the chapter
- `prerequisites`: Array of prerequisite knowledge requirements
- `estimatedTime`: Estimated time to complete the chapter (minutes)
- `difficulty`: Difficulty level (beginner, intermediate, advanced)
- `tags`: Array of content tags for search and categorization
- `createdAt`: Creation timestamp
- `updatedAt`: Last modification timestamp
- `authors`: Array of author names
- `reviewers`: Array of reviewer names

**Relationships**:
- Parent: Module (one-to-many)
- Children: Sections (one-to-many)
- Related: Other chapters (many-to-many for cross-references)

### 2. Content Section
**Definition**: A subsection within a chapter that organizes content into digestible segments.

**Attributes**:
- `id`: Unique identifier
- `title`: Section title
- `slug`: URL-friendly identifier
- `chapterId`: Reference to parent chapter
- `order`: Display order within the chapter
- `type`: Content type (text, code, diagram, exercise, example)
- `content`: Markdown content
- `metadata`: Additional metadata specific to content type

**Relationships**:
- Parent: Chapter (many-to-one)
- Children: Content Blocks (one-to-many)

### 3. Content Block
**Definition**: The smallest unit of content that can be individually styled, referenced, or interacted with.

**Attributes**:
- `id`: Unique identifier
- `type`: Block type (paragraph, heading, code, image, diagram, exercise, etc.)
- `content`: The actual content (Markdown, HTML, or component props)
- `sectionId`: Reference to parent section
- `order`: Display order within the section
- `metadata`: Type-specific metadata

### 4. Interactive Element
**Definition**: A dynamic component that provides interactive functionality within the static site.

**Attributes**:
- `id`: Unique identifier
- `type`: Element type (diagram-viewer, code-playground, simulation-viewer, etc.)
- `props`: Configuration properties for the component
- `chapterId`: Reference to containing chapter
- `sectionId`: Reference to containing section
- `title`: Display title for the interactive element
- `description`: Brief description of functionality

**Relationships**:
- Parent: Content Block or Section (many-to-one)

### 5. Module
**Definition**: The original 4 modules that serve as the source for the 10 chapters.

**Attributes**:
- `id`: Unique identifier (e.g., "module-1-ros")
- `title`: Module title
- `number`: Module number (1-4)
- `description`: Brief description of the module
- `chapters`: Array of chapter IDs derived from this module

**Relationships**:
- Children: Chapters (one-to-many)

## Content Structure Hierarchy

```
Textbook
├── Module 1: The Robotic Nervous System (ROS 2)
│   ├── Chapter 1: ROS 2 fundamentals and architecture
│   │   ├── Sections: Introduction, Architecture, Nodes, Topics, Services, Summary
│   │   └── Interactive: ROS 2 Architecture Diagram
│   ├── Chapter 2: Nodes, Topics, Services and rclpy integration
│   │   ├── Sections: Node Structure, Topics and Messages, Services, rclpy, Examples
│   │   └── Interactive: Code Playground for rclpy
│   └── Chapter 3: URDF for humanoid robots
│       ├── Sections: URDF Basics, Humanoid URDF, Examples, Best Practices
│       └── Interactive: URDF Visualization
├── Module 2: The Digital Twin (Gazebo & Unity)
│   ├── Chapter 4: Gazebo simulation fundamentals
│   ├── Chapter 5: Unity integration and high-fidelity rendering
│   └── Chapter 6: Sensor simulation: LiDAR, Depth Cameras, IMUs
├── Module 3: The AI-Robot Brain (NVIDIA Isaac™)
│   ├── Chapter 7: NVIDIA Isaac Sim and synthetic data
│   ├── Chapter 8: Isaac ROS and VSLAM
│   └── Chapter 9: Nav2 and path planning
└── Module 4: Vision-Language-Action (VLA)
    └── Chapter 10: VLA systems, voice-to-action, and cognitive planning
```

## Metadata Schema

### Frontmatter for Markdown Files
```yaml
title: Chapter Title
description: Brief description of the chapter
module: 1  # Module number (1-4)
chapter: 1 # Chapter number (1-10)
learning_objectives:
  - Objective 1
  - Objective 2
difficulty: beginner | intermediate | advanced
estimated_time: 45 # in minutes
tags:
  - ros2
  - architecture
  - nodes
authors:
  - Author Name
prerequisites:
  - Previous chapter
  - Basic Python knowledge
```

## Content Relationships

### Cross-Chapter References
- Links between related concepts across chapters
- Prerequisite tracking for learning progression
- Related content suggestions

### Interactive Element Relationships
- Links between interactive elements and their data sources
- Dependencies between different interactive components
- Shared resources for similar interactive elements

## Search and Indexing Model

### Searchable Content Structure
- Chapter titles and descriptions
- Section headings and content
- Interactive element titles and descriptions
- Tags and metadata
- Code examples and comments

### Indexing Strategy
- Full-text search across all content
- Faceted search by module, chapter, difficulty
- Tag-based filtering
- Content type filtering (text, code, diagrams)

## Validation Rules

1. Each chapter must belong to exactly one module
2. Chapter numbers within a module must be sequential
3. All content must have proper metadata
4. Interactive elements must have fallback content for accessibility
5. All images must have alternative text
6. Content must follow accessibility guidelines (WCAG 2.1 AA)

## File Organization Mapping

The data model maps to the file structure as follows:

```
docs/
├── _modules/
│   ├── module-1.md
│   ├── module-2.md
│   ├── module-3.md
│   └── module-4.md
├── chapter-1/
│   ├── index.md          # Chapter 1: ROS 2 fundamentals
│   ├── nodes-topics.md   # Chapter 2: Nodes, Topics, Services
│   └── urdf.md          # Chapter 3: URDF for humanoid robots
├── chapter-2/
│   ├── gazebo.md        # Chapter 4: Gazebo simulation
│   ├── unity.md         # Chapter 5: Unity integration
│   └── sensors.md       # Chapter 6: Sensor simulation
├── chapter-3/
│   ├── isaac-sim.md     # Chapter 7: Isaac Sim and synthetic data
│   ├── isaac-ros.md     # Chapter 8: Isaac ROS and VSLAM
│   └── nav2.md          # Chapter 9: Nav2 and path planning
└── chapter-4/
    └── vla.md           # Chapter 10: VLA systems
```

## Implementation Considerations

### Docusaurus Integration
- Use frontmatter for metadata
- Leverage MDX for interactive components
- Implement custom remark plugins for specialized content
- Use Docusaurus' built-in sidebar generation

### Content Management
- Maintain consistency in metadata schema
- Implement content validation tools
- Establish clear content authoring guidelines
- Plan for content versioning and updates

This data model provides the foundation for organizing the Physical AI & Humanoid Robotics textbook content in a structured, searchable, and maintainable way within the Docusaurus framework.