# Quickstart Guide: Physical AI & Humanoid Robotics Textbook

**Date**: 2025-12-26 | **Feature**: 001-docusaurus-deployment

## Overview

This quickstart guide provides the essential steps to set up, develop, and deploy the Physical AI & Humanoid Robotics textbook using Docusaurus. This guide is designed for developers who need to get the project up and running quickly.

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn package manager
- Git
- GitHub account for deployment
- Basic knowledge of Markdown and React (for interactive components)

## Setup Instructions

### 1. Clone and Initialize

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
npm install
# OR
yarn install
```

### 2. Local Development

```bash
# Start local development server
npm run start
# OR
yarn start

# This command starts a local development server and opens the website in your browser
# The site will automatically reload when you make changes to the content
```

### 3. Project Structure Overview

```
.
├── docs/                    # Content directory
│   ├── intro.md            # Introduction page
│   ├── chapter-1/          # Module 1: The Robotic Nervous System
│   │   ├── index.md        # Chapter 1: ROS 2 fundamentals
│   │   ├── nodes-topics.md # Chapter 2: Nodes, Topics, Services
│   │   └── urdf.md         # Chapter 3: URDF for humanoid robots
│   ├── chapter-2/          # Module 2: The Digital Twin
│   │   ├── gazebo.md       # Chapter 4: Gazebo simulation
│   │   ├── unity.md        # Chapter 5: Unity integration
│   │   └── sensors.md      # Chapter 6: Sensor simulation
│   ├── chapter-3/          # Module 3: The AI-Robot Brain
│   │   ├── isaac-sim.md    # Chapter 7: Isaac Sim and synthetic data
│   │   ├── isaac-ros.md    # Chapter 8: Isaac ROS and VSLAM
│   │   └── nav2.md         # Chapter 9: Nav2 and path planning
│   └── chapter-4/          # Module 4: Vision-Language-Action
│       └── vla.md          # Chapter 10: VLA systems
├── src/                    # Custom React components
│   └── components/         # Interactive elements
├── static/                 # Static assets
├── docusaurus.config.js    # Main configuration
├── sidebars.js            # Navigation configuration
└── package.json           # Dependencies and scripts
```

## Content Creation

### 1. Adding a New Chapter

To add a new chapter, create a new Markdown file in the appropriate chapter directory:

```markdown
---
title: Chapter Title
description: Brief description of the chapter
module: 1  # Module number (1-4)
chapter: 1 # Chapter number (1-10)
learning_objectives:
  - Understand basic concepts
  - Learn practical applications
difficulty: beginner
estimated_time: 45
tags:
  - ros2
  - architecture
---

# Chapter Title

## Introduction

Your chapter content here...

## Section Title

More content...

```

### 2. Creating Interactive Elements

To add interactive components, use MDX syntax in your Markdown files:

```jsx
import DiagramViewer from '@site/src/components/DiagramViewer';

<DiagramViewer
  title="ROS 2 Architecture"
  description="Interactive diagram showing ROS 2 architecture"
  diagramType="architecture"
  data={/* diagram data */}
/>
```

### 3. Adding Code Examples

Use standard Markdown code blocks with syntax highlighting:

```python
# ROS 2 Python example
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
```

## Configuration

### Main Configuration (`docusaurus.config.js`)

```javascript
module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'A comprehensive textbook on embodied intelligence',
  url: 'https://your-username.github.io',
  baseUrl: '/physical-ai-textbook/',
  organizationName: 'your-username',
  projectName: 'physical-ai-textbook',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',

  // GitHub Pages deployment
  organizationName: 'your-org', // Usually your GitHub org/user name
  projectName: 'physical-ai-textbook', // Usually your repo name

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/your-org/physical-ai-textbook/edit/main/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Physical AI Textbook',
        logo: {
          alt: 'Physical AI Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'intro',
            position: 'left',
            label: 'Textbook',
          },
          {
            href: 'https://github.com/your-org/physical-ai-textbook',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Content',
            items: [
              {
                label: 'Introduction',
                to: '/docs/intro',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Textbook. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer/themes/github'),
        darkTheme: require('prism-react-renderer/themes/dracula'),
        additionalLanguages: ['python', 'bash', 'json', 'yaml'],
      },
    }),
};
```

## Building and Deployment

### 1. Build for Production

```bash
# Build the static files for production
npm run build
# OR
yarn build

# The built files will be in the build/ directory
```

### 2. Local Preview of Production Build

```bash
# Serve the production build locally for testing
npm run serve
# OR
yarn serve

# This serves the built site on http://localhost:3000
```

### 3. Deployment to GitHub Pages

The project is configured for GitHub Pages deployment using GitHub Actions. To deploy:

1. Ensure your `docusaurus.config.js` has the correct `organizationName` and `projectName`
2. Push your changes to the `main` branch
3. The GitHub Actions workflow will automatically build and deploy the site

Alternatively, you can deploy manually:

```bash
# Deploy to GitHub Pages
npm run deploy
# OR
yarn deploy

# This command builds the site and pushes the static files to the gh-pages branch
```

## Development Workflow

### 1. Content Development

1. Create or edit Markdown files in the `docs/` directory
2. Use the local development server to preview changes
3. Follow the content structure guidelines in the data model
4. Ensure all content has proper metadata in the frontmatter

### 2. Component Development

1. Create new React components in `src/components/`
2. Test components in the development environment
3. Use MDX to embed components in Markdown files
4. Ensure components are accessible and responsive

### 3. Testing

```bash
# Run unit tests
npm test
# OR
yarn test

# Run linting
npm run lint
# OR
yarn lint

# Build to catch any build-time errors
npm run build
```

## Common Tasks

### Update Navigation

Edit `sidebars.js` to update the navigation structure:

```javascript
module.exports = {
  textbook: [
    'intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System',
      items: [
        'chapter-1/index',
        'chapter-1/nodes-topics',
        'chapter-1/urdf',
      ],
    },
    // ... other chapters
  ],
};
```

### Add New Assets

1. Place images in `static/img/`
2. Place other static files in the appropriate `static/` subdirectory
3. Reference assets using `/img/filename.ext` path format

### Custom Styling

1. Add custom CSS to `src/css/custom.css`
2. Or create new components with styled components
3. Use Docusaurus' theme customization options

## Troubleshooting

### Common Issues

**Q: Site doesn't reload automatically during development**
A: Ensure file watching is enabled and check for permission issues

**Q: Build fails with "out of memory" error**
A: Increase Node.js memory limit: `export NODE_OPTIONS="--max_old_space_size=4096"`

**Q: Images not displaying**
A: Check that images are in the `static/` directory and use correct paths

**Q: Search not working**
A: Ensure Algolia search is configured properly in `docusaurus.config.js`

## Next Steps

1. Review the [full specification](./spec.md) for detailed requirements
2. Check the [implementation plan](./plan.md) for development phases
3. Follow the [tasks list](./tasks.md) for implementation steps
4. Refer to the [data model](./data-model.md) for content structure
5. Consult the [research document](./research.md) for technical decisions

This quickstart guide provides the essential information to begin working with the Physical AI & Humanoid Robotics textbook project. For more detailed information, refer to the other documentation files in this specification directory.