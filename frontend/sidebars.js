// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: ROS 2 Fundamentals',
      items: [
        {
          type: 'category',
          label: 'Week 1: ROS 2 Architecture',
          items: [
            'module1/week1/ros2-intro',
            'module1/week1/nodes',
            'module1/week1/architecture'
          ],
        },
        {
          type: 'category',
          label: 'Week 2: Communication',
          items: [
            'module1/week2/topics',
            'module1/week2/services',
            'module1/week2/actions'
          ],
        },
        {
          type: 'category',
          label: 'Week 3: Configuration',
          items: [
            'module1/week3/launch-files',
            'module1/week3/parameters',
            'module1/week3/exercises'
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 2: Simulation Environments',
      items: [
        {
          type: 'category',
          label: 'Week 4: Gazebo Simulation',
          items: [
            'module2/week4/gazebo-intro',
            'module2/week4/physics',
            'module2/week4/setup'
          ],
        },
        {
          type: 'category',
          label: 'Week 5: Robot Modeling',
          items: [
            'module2/week5/urdf',
            'module2/week5/sensors',
            'module2/week5/models'
          ],
        },
        {
          type: 'category',
          label: 'Week 6: Controllers',
          items: [
            'module2/week6/controllers',
            'module2/week6/plugins',
            'module2/week6/exercises'
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 3: NVIDIA Isaac Integration',
      items: [
        {
          type: 'category',
          label: 'Week 7: Isaac Overview',
          items: [
            'module3/week7/isaac-intro',
            'module3/week7/setup',
            'module3/week7/architecture'
          ],
        },
        {
          type: 'category',
          label: 'Week 8: Perception Pipeline',
          items: [
            'module3/week8/perception',
            'module3/week8/sensors',
            'module3/week8/vision'
          ],
        },
        {
          type: 'category',
          label: 'Week 9: Planning and Control',
          items: [
            'module3/week9/planning',
            'module3/week9/control',
            'module3/week9/exercises'
          ],
        },
        {
          type: 'category',
          label: 'Week 10: AI Integration',
          items: [
            'module3/week10/ai-integration',
            'module3/week10/deep-learning',
            'module3/week10/exercises'
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action Systems',
      items: [
        {
          type: 'category',
          label: 'Week 11: Vision Systems',
          items: [
            'module4/week11/vision-intro',
            'module4/week11/object-detection',
            'module4/week11/vision-exercises'
          ],
        },
        {
          type: 'category',
          label: 'Week 12: Language Understanding',
          items: [
            'module4/week12/language-intro',
            'module4/week12/nlp',
            'module4/week12/language-exercises'
          ],
        },
        {
          type: 'category',
          label: 'Week 13: Action Execution',
          items: [
            'module4/week13/action-intro',
            'module4/week13/manipulation',
            'module4/week13/vla-project'
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Capstone Project',
      items: [
        'capstone/intro',
        'capstone/project-requirements',
        'capstone/submission',
        'capstone/evaluation'
      ],
    }
  ],
};

module.exports = sidebars;