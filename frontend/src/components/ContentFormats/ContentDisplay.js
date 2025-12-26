import React, { useState, useEffect } from 'react';
import styles from './ContentDisplay.module.css';

const ContentDisplay = ({ content, format = 'auto' }) => {
  const [activeTab, setActiveTab] = useState('content');
  const [codeLanguage, setCodeLanguage] = useState('python');
  const [codeTheme, setCodeTheme] = useState('light');

  useEffect(() => {
    // Determine the best display format based on content type
    if (format === 'auto' && content) {
      if (content.type === 'code' || content.code) {
        setActiveTab('code');
      } else if (content.type === 'simulation' || content.simulation) {
        setActiveTab('simulation');
      } else if (content.type === 'interactive' || content.interactive) {
        setActiveTab('interactive');
      } else {
        setActiveTab('content');
      }
    }
  }, [content, format]);

  const renderTextContent = () => {
    if (!content) return null;

    // If content has markdown, render it as HTML
    if (content.markdown) {
      return (
        <div
          className={styles.textContent}
          dangerouslySetInnerHTML={{ __html: content.markdown }}
        />
      );
    }

    // Otherwise render as plain text or with basic formatting
    return (
      <div className={styles.textContent}>
        {content.title && <h1>{content.title}</h1>}
        {content.subtitle && <h2>{content.subtitle}</h2>}
        {content.content && <p>{content.content}</p>}
        {content.description && <p>{content.description}</p>}
        {content.paragraphs && content.paragraphs.map((para, index) => (
          <p key={index}>{para}</p>
        ))}
      </div>
    );
  };

  const renderCodeContent = () => {
    if (!content || !content.code) return null;

    return (
      <div className={styles.codeContent}>
        <div className={styles.codeHeader}>
          <div className={styles.codeControls}>
            <select
              value={codeLanguage}
              onChange={(e) => setCodeLanguage(e.target.value)}
              className={styles.languageSelect}
            >
              <option value="python">Python</option>
              <option value="cpp">C++</option>
              <option value="bash">Bash</option>
              <option value="javascript">JavaScript</option>
              <option value="json">JSON</option>
              <option value="yaml">YAML</option>
            </select>
            <select
              value={codeTheme}
              onChange={(e) => setCodeTheme(e.target.value)}
              className={styles.themeSelect}
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
            </select>
          </div>
          <div className={styles.codeActions}>
            <button className={styles.copyButton}>
              Copy
            </button>
            <button className={styles.runButton}>
              Run
            </button>
          </div>
        </div>

        <pre className={`${styles.codeBlock} ${styles[codeTheme]}`}>
          <code className={`language-${codeLanguage}`}>
            {content.code}
          </code>
        </pre>

        {content.codeDescription && (
          <div className={styles.codeDescription}>
            {content.codeDescription}
          </div>
        )}

        {content.codeExplanations && (
          <div className={styles.codeExplanations}>
            <h4>Code Explanations:</h4>
            <ul>
              {content.codeExplanations.map((explanation, index) => (
                <li key={index}>
                  <strong>Line {explanation.line}:</strong> {explanation.description}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  };

  const renderSimulationContent = () => {
    if (!content || !content.simulation) return null;

    return (
      <div className={styles.simulationContent}>
        <div className={styles.simulationHeader}>
          <h3>{content.simulation.title || 'Simulation Environment'}</h3>
          <div className={styles.simulationControls}>
            <button className={styles.simulationButton}>
              Start
            </button>
            <button className={styles.simulationButton}>
              Pause
            </button>
            <button className={styles.simulationButton}>
              Reset
            </button>
          </div>
        </div>

        <div className={styles.simulationViewer}>
          {/* Placeholder for simulation viewer */}
          <div className={styles.simulationPlaceholder}>
            <div className={styles.simulationVisual}>
              {content.simulation.type === 'gazebo' && '🤖 Gazebo Simulation Environment'}
              {content.simulation.type === 'isaac' && '🎮 NVIDIA Isaac Simulation'}
              {content.simulation.type === 'custom' && '🔬 Custom Simulation Environment'}
            </div>
            <div className={styles.simulationInfo}>
              <p><strong>Type:</strong> {content.simulation.type || 'Unknown'}</p>
              <p><strong>Environment:</strong> {content.simulation.environment || 'Default'}</p>
              <p><strong>Duration:</strong> {content.simulation.duration || 'N/A'}</p>
            </div>
          </div>
        </div>

        <div className={styles.simulationDescription}>
          <h4>Simulation Details</h4>
          <p>{content.simulation.description || 'No description available.'}</p>

          {content.simulation.objectives && (
            <div className={styles.simulationObjectives}>
              <h5>Learning Objectives:</h5>
              <ul>
                {content.simulation.objectives.map((objective, index) => (
                  <li key={index}>{objective}</li>
                ))}
              </ul>
            </div>
          )}

          {content.simulation.instructions && (
            <div className={styles.simulationInstructions}>
              <h5>Instructions:</h5>
              <ol>
                {content.simulation.instructions.map((instruction, index) => (
                  <li key={index}>{instruction}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderInteractiveContent = () => {
    if (!content || !content.interactive) return null;

    return (
      <div className={styles.interactiveContent}>
        <h3>{content.interactive.title || 'Interactive Content'}</h3>

        {content.interactive.type === 'quiz' && (
          <div className={styles.quizContainer}>
            {content.interactive.questions && content.interactive.questions.map((question, qIndex) => (
              <div key={qIndex} className={styles.question}>
                <h4>{question.question}</h4>
                <div className={styles.answers}>
                  {question.options && question.options.map((option, oIndex) => (
                    <label key={oIndex} className={styles.answerOption}>
                      <input
                        type="radio"
                        name={`question-${qIndex}`}
                        value={oIndex}
                        className={styles.answerInput}
                      />
                      <span className={styles.answerText}>{option}</span>
                    </label>
                  ))}
                </div>
              </div>
            ))}
            <button className={styles.submitQuizButton}>
              Submit Answers
            </button>
          </div>
        )}

        {content.interactive.type === 'exercise' && (
          <div className={styles.exerciseContainer}>
            <h4>Exercise: {content.interactive.title}</h4>
            <p>{content.interactive.description}</p>
            <div className={styles.exerciseInput}>
              <textarea
                placeholder="Enter your solution here..."
                rows={8}
                className={styles.solutionInput}
              />
            </div>
            <button className={styles.submitExerciseButton}>
              Submit Solution
            </button>
          </div>
        )}
      </div>
    );
  };

  const hasMultipleFormats = () => {
    return content && (
      (content.content || content.markdown) ||
      content.code ||
      content.simulation ||
      content.interactive
    );
  };

  return (
    <div className={styles.contentDisplay}>
      {hasMultipleFormats() && (
        <div className={styles.tabContainer}>
          <div className={styles.tabs}>
            {(content.content || content.markdown) && (
              <button
                className={`${styles.tab} ${activeTab === 'content' ? styles.activeTab : ''}`}
                onClick={() => setActiveTab('content')}
              >
                Content
              </button>
            )}
            {content.code && (
              <button
                className={`${styles.tab} ${activeTab === 'code' ? styles.activeTab : ''}`}
                onClick={() => setActiveTab('code')}
              >
                Code
              </button>
            )}
            {content.simulation && (
              <button
                className={`${styles.tab} ${activeTab === 'simulation' ? styles.activeTab : ''}`}
                onClick={() => setActiveTab('simulation')}
              >
                Simulation
              </button>
            )}
            {content.interactive && (
              <button
                className={`${styles.tab} ${activeTab === 'interactive' ? styles.activeTab : ''}`}
                onClick={() => setActiveTab('interactive')}
              >
                Interactive
              </button>
            )}
          </div>
        </div>
      )}

      <div className={styles.tabContent}>
        {activeTab === 'content' && renderTextContent()}
        {activeTab === 'code' && renderCodeContent()}
        {activeTab === 'simulation' && renderSimulationContent()}
        {activeTab === 'interactive' && renderInteractiveContent()}
      </div>

      {content && content.resources && content.resources.length > 0 && (
        <div className={styles.resources}>
          <h4>Additional Resources</h4>
          <ul>
            {content.resources.map((resource, index) => (
              <li key={index}>
                <a href={resource.url} target="_blank" rel="noopener noreferrer">
                  {resource.title}
                </a>
                {resource.description && <p>{resource.description}</p>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ContentDisplay;