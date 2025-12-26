// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals and Architecture',
      items: [
        'chapter-1/index',      // Chapter 1: ROS 2 Fundamentals and Architecture
        'chapter-1/nodes-topics', // Chapter 2: Nodes, Topics, and Services
        'chapter-1/urdf',       // Chapter 3: URDF for Humanoid Robots
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Digital Twin and Simulation',
      items: [
        'chapter-2/gazebo',     // Chapter 4: Gazebo Simulation Fundamentals
        'chapter-2/unity',      // Chapter 5: Unity Integration and High-Fidelity Rendering
        'chapter-2/sensors',    // Chapter 6: Sensor Simulation and Integration
      ],
    },
    {
      type: 'category',
      label: 'Module 3: AI-Robot Brain Integration',
      items: [
        'chapter-3/isaac-sim',  // Chapter 7: Isaac Sim for Physical AI
        'chapter-3/isaac-ros',  // Chapter 8: Isaac ROS for GPU-Accelerated Perception
        'chapter-3/nav2',       // Chapter 9: Navigation (Nav2) for Humanoid Robots
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems',
      items: [
        'chapter-4/vla',        // Chapter 10: Vision-Language-Action (VLA) Systems
      ],
    },
    {
      type: 'category',
      label: 'Instructor Resources',
      items: [
        'instructor/index',
        'instructor/curriculum',
        'instructor/curriculum-map',
      ],
    },
  ],
};

export default sidebars;