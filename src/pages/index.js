import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import clsx from 'clsx';

function Homepage() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="A comprehensive textbook on Physical AI & Humanoid Robotics">
      <main>
        {/* Hero Section */}
        <div className={clsx('hero hero--primary', 'text--center')}>
          <div className="container">
            <h1 className="hero__title">{siteConfig.title}</h1>
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <div className="padding-top--md">
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                Start Reading Now
              </Link>
            </div>
          </div>
        </div>

        {/* Welcome Section */}
        <div className="container padding-vert--lg">
          <div className="row">
            <div className="col col--8 col--offset-2 text--center">
              <div className="padding-vert--lg">
                <h2>Explore the Future of Robotics</h2>
                <p className="hero__subtitle">
                  Welcome to the comprehensive textbook on Physical AI & Humanoid Robotics -
                  where artificial intelligence meets embodied systems.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="container padding-vert--lg">
          <div className="row">
            <div className="col col--10 col--offset-1">
              <div className="row">
                <div className="col col--4 margin-vert--md">
                  <div className="card">
                    <div className="card__header text--center">
                      <h3>🤖 Fundamentals</h3>
                    </div>
                    <div className="card__body text--center">
                      <p>Understand the foundations of embodied intelligence and robotics</p>
                    </div>
                    <div className="card__footer text--center">
                      <Link to="/docs/chapter-1" className="button button--sm button--primary">
                        Learn More
                      </Link>
                    </div>
                  </div>
                </div>

                <div className="col col--4 margin-vert--md">
                  <div className="card">
                    <div className="card__header text--center">
                      <h3>🧠 AI Integration</h3>
                    </div>
                    <div className="card__body text--center">
                      <p>Learn how machine learning applies to physical systems</p>
                    </div>
                    <div className="card__footer text--center">
                      <Link to="/docs/chapter-3/isaac-ros" className="button button--sm button--primary">
                        Learn More
                      </Link>
                    </div>
                  </div>
                </div>

                <div className="col col--4 margin-vert--md">
                  <div className="card">
                    <div className="card__header text--center">
                      <h3>🦾 Practical Applications</h3>
                    </div>
                    <div className="card__body text--center">
                      <p>Discover real-world implementations and case studies</p>
                    </div>
                    <div className="card__footer text--center">
                      <Link to="/docs/chapter-4/vla" className="button button--sm button--primary">
                        Learn More
                      </Link>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Content Preview Section */}
        <div className="container padding-vert--lg">
          <div className="row">
            <div className="col col--8 col--offset-2">
              <h2 className="text--center padding-bottom--lg">What You'll Learn</h2>
              <div className="row">
                <div className="col col--6">
                  <ul>
                    <li>Foundations of embodied intelligence</li>
                    <li>Robotics control systems and algorithms</li>
                    <li>Machine learning applications in robotics</li>
                    <li>Humanoid robot design and mechanics</li>
                  </ul>
                </div>
                <div className="col col--6">
                  <ul>
                    <li>Simulation environments and real-world deployment</li>
                    <li>ROS2 integration and communication</li>
                    <li>Safety and ethical considerations</li>
                    <li>Advanced locomotion and manipulation</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className={clsx('hero hero--secondary', 'text--center')}>
          <div className="container padding-vert--lg">
            <h2 className="hero__title">Ready to Dive In?</h2>
            <p className="hero__subtitle">
              Start your journey into the world of Physical AI and Humanoid Robotics today
            </p>
            <div className="padding-top--md">
              <Link
                className="button button--primary button--lg margin-right--md"
                to="/docs/intro">
                Begin Reading
              </Link>
              <Link
                className="button button--secondary button--lg"
                to="/docs/intro">
                Explore Topics
              </Link>
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default Homepage;