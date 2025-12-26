import React, { useState } from 'react';
import styles from './AssessmentRubrics.module.css';

const AssessmentRubrics = ({ rubrics, assessment, onRubricSelect }) => {
  const [selectedRubric, setSelectedRubric] = useState(null);
  const [expandedCriteria, setExpandedCriteria] = useState({});

  const toggleCriteria = (criterionId) => {
    setExpandedCriteria(prev => ({
      ...prev,
      [criterionId]: !prev[criterionId]
    }));
  };

  const handleRubricSelect = (rubric) => {
    setSelectedRubric(rubric);
    if (onRubricSelect) {
      onRubricSelect(rubric);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 90) return styles.excellent;
    if (score >= 80) return styles.good;
    if (score >= 70) return styles.satisfactory;
    if (score >= 60) return styles.needsImprovement;
    return styles.inadequate;
  };

  const getScoreLabel = (score) => {
    if (score >= 90) return 'Excellent';
    if (score >= 80) return 'Good';
    if (score >= 70) return 'Satisfactory';
    if (score >= 60) return 'Needs Improvement';
    return 'Inadequate';
  };

  const renderRubricOverview = () => (
    <div className={styles.rubricOverview}>
      <h2>Assessment Rubrics</h2>
      <p className={styles.description}>
        Rubrics provide clear expectations and standards for evaluating your work.
        Each rubric outlines specific criteria and performance levels.
      </p>

      <div className={styles.rubricsGrid}>
        {rubrics?.map((rubric, index) => (
          <div
            key={rubric.id || index}
            className={`${styles.rubricCard} ${selectedRubric?.id === rubric.id ? styles.selected : ''}`}
            onClick={() => handleRubricSelect(rubric)}
          >
            <div className={styles.rubricHeader}>
              <h3>{rubric.title || `Rubric ${index + 1}`}</h3>
              <span className={styles.rubricType}>{rubric.type || 'General'}</span>
            </div>

            <div className={styles.rubricSummary}>
              <div className={styles.summaryItem}>
                <span className={styles.label}>Criteria:</span>
                <span className={styles.value}>{rubric.criteria?.length || 0}</span>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.label}>Max Score:</span>
                <span className={styles.value}>{rubric.maxScore || 100}</span>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.label}>Type:</span>
                <span className={styles.value}>{rubric.format || 'Point-based'}</span>
              </div>
            </div>

            <div className={styles.rubricDescription}>
              {rubric.description || 'No description provided.'}
            </div>

            <div className={styles.rubricActions}>
              <button className={styles.viewButton}>
                View Details
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderDetailedRubric = () => {
    if (!selectedRubric) return renderRubricOverview();

    return (
      <div className={styles.detailedRubric}>
        <div className={styles.rubricHeader}>
          <button className={styles.backButton} onClick={() => setSelectedRubric(null)}>
            ← Back to Rubrics
          </button>
          <h2>{selectedRubric.title}</h2>
          <div className={styles.rubricMeta}>
            <span className={styles.rubricType}>{selectedRubric.type}</span>
            <span className={styles.maxScore}>Max: {selectedRubric.maxScore || 100} points</span>
          </div>
        </div>

        <div className={styles.rubricDescription}>
          <h3>Description</h3>
          <p>{selectedRubric.description || 'No description provided.'}</p>
        </div>

        <div className={styles.criteriaSection}>
          <h3>Evaluation Criteria</h3>
          <div className={styles.criteriaList}>
            {selectedRubric.criteria?.map((criterion, index) => (
              <div key={criterion.id || index} className={styles.criterion}>
                <div
                  className={styles.criterionHeader}
                  onClick={() => toggleCriteria(criterion.id || index)}
                >
                  <div className={styles.criterionInfo}>
                    <h4>{criterion.name || `Criterion ${index + 1}`}</h4>
                    <p className={styles.criterionDescription}>{criterion.description}</p>
                  </div>
                  <div className={styles.criterionScore}>
                    <span>Max: {criterion.maxPoints || 0} pts</span>
                  </div>
                  <div className={styles.expandIcon}>
                    {expandedCriteria[criterion.id || index] ? '▲' : '▼'}
                  </div>
                </div>

                {expandedCriteria[criterion.id || index] && (
                  <div className={styles.criterionDetails}>
                    <div className={styles.performanceLevels}>
                      <h5>Performance Levels</h5>
                      <div className={styles.levelsGrid}>
                        {criterion.levels?.map((level, levelIndex) => (
                          <div key={levelIndex} className={styles.levelCard}>
                            <div className={`${styles.levelHeader} ${getScoreColor(level.score)}`}>
                              <span className={styles.levelName}>{level.name}</span>
                              <span className={styles.levelScore}>{level.score}%</span>
                            </div>
                            <div className={styles.levelDescription}>
                              <p>{level.description}</p>
                              <p className={styles.levelExample}>
                                <strong>Example:</strong> {level.example || 'No example provided.'}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {criterion.standards && (
                      <div className={styles.standardsSection}>
                        <h5>Standards</h5>
                        <ul className={styles.standardsList}>
                          {criterion.standards.map((standard, stdIndex) => (
                            <li key={stdIndex} className={styles.standardItem}>
                              {standard}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className={styles.rubricSummaryTable}>
          <h3>Rubric Summary</h3>
          <div className={styles.summaryGrid}>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Total Criteria:</span>
              <span className={styles.value}>{selectedRubric.criteria?.length || 0}</span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Maximum Score:</span>
              <span className={styles.value}>{selectedRubric.maxScore || 100}</span>
            </div>
            <div className={styles.summaryItem}>
              <span className={styles.label}>Assessment Type:</span>
              <span className={styles.value}>{selectedRubric.type || 'General'}</span>
            </div>
          </div>
        </div>

        <div className={styles.rubricActions}>
          <button className={styles.printButton}>
            Print Rubric
          </button>
          <button className={styles.shareButton}>
            Share Rubric
          </button>
          <button className={styles.exportButton}>
            Export as PDF
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className={styles.assessmentRubrics}>
      {selectedRubric ? renderDetailedRubric() : renderRubricOverview()}
    </div>
  );
};

export default AssessmentRubrics;