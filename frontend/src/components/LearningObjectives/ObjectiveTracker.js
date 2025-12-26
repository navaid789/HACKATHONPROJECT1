import React, { useState, useEffect } from 'react';
import styles from './ObjectiveTracker.module.css';

const ObjectiveTracker = ({ objectives, module, week, userProgress }) => {
  const [selectedModule, setSelectedModule] = useState(module);
  const [selectedWeek, setSelectedWeek] = useState(week);
  const [filter, setFilter] = useState('all'); // 'all', 'completed', 'in-progress', 'not-started'
  const [expandedObjectives, setExpandedObjectives] = useState({});

  useEffect(() => {
    if (module) setSelectedModule(module);
    if (week) setSelectedWeek(week);
  }, [module, week]);

  const toggleObjective = (objectiveId) => {
    setExpandedObjectives(prev => ({
      ...prev,
      [objectiveId]: !prev[objectiveId]
    }));
  };

  const getStatus = (objective) => {
    const progress = userProgress?.find(p => p.objectiveId === objective.id);
    if (!progress) return 'not-started';

    if (progress.mastery >= 0.8) return 'completed';
    if (progress.mastery >= 0.3) return 'in-progress';
    return 'not-started';
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'beginner': return styles.beginner;
      case 'intermediate': return styles.intermediate;
      case 'advanced': return styles.advanced;
      default: return styles.intermediate;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return styles.completed;
      case 'in-progress': return styles.inProgress;
      case 'not-started': return styles.notStarted;
      default: return styles.notStarted;
    }
  };

  const getMasteryLevel = (mastery) => {
    if (mastery >= 0.8) return 'Mastered';
    if (mastery >= 0.6) return 'Proficient';
    if (mastery >= 0.4) return 'Developing';
    if (mastery >= 0.2) return 'Beginning';
    return 'Not Started';
  };

  const getFilteredObjectives = () => {
    if (!objectives) return [];

    return objectives.filter(objective => {
      const status = getStatus(objective);
      if (filter === 'all') return true;
      return status === filter;
    });
  };

  const calculateOverallProgress = () => {
    if (!objectives || objectives.length === 0) return 0;

    const completed = objectives.filter(obj => {
      const progress = userProgress?.find(p => p.objectiveId === obj.id);
      return progress && progress.mastery >= 0.8;
    }).length;

    return Math.round((completed / objectives.length) * 100);
  };

  const calculateModuleProgress = () => {
    if (!objectives) return 0;

    const totalMastery = objectives.reduce((sum, obj) => {
      const progress = userProgress?.find(p => p.objectiveId === obj.id);
      return sum + (progress?.mastery || 0);
    }, 0);

    return Math.round((totalMastery / objectives.length) * 100);
  };

  const renderOverview = () => (
    <div className={styles.overview}>
      <div className={styles.header}>
        <h1>Learning Objectives Tracker</h1>
        <div className={styles.controls}>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className={styles.filterSelect}
          >
            <option value="all">All Objectives</option>
            <option value="completed">Completed</option>
            <option value="in-progress">In Progress</option>
            <option value="not-started">Not Started</option>
          </select>
        </div>
      </div>

      <div className={styles.overviewStats}>
        <div className={styles.statCard}>
          <h3>{calculateOverallProgress()}%</h3>
          <p>Overall Completion</p>
        </div>
        <div className={styles.statCard}>
          <h3>{calculateModuleProgress()}%</h3>
          <p>Average Mastery</p>
        </div>
        <div className={styles.statCard}>
          <h3>{objectives?.length || 0}</h3>
          <p>Total Objectives</p>
        </div>
        <div className={styles.statCard}>
          <h3>{userProgress?.filter(p => p.mastery >= 0.8).length || 0}</h3>
          <p>Mastered Objectives</p>
        </div>
      </div>

      <div className={styles.modulesList}>
        {objectives && objectives.length > 0 ? (
          getFilteredObjectives().map((objective, index) => {
            const status = getStatus(objective);
            const progress = userProgress?.find(p => p.objectiveId === objective.id);
            const mastery = progress?.mastery || 0;

            return (
              <div
                key={objective.id || index}
                className={`${styles.objectiveCard} ${styles[status]}`}
                onClick={() => toggleObjective(objective.id || index)}
              >
                <div className={styles.objectiveHeader}>
                  <div className={styles.objectiveTitle}>
                    <span className={`${styles.weekIndicator} ${getDifficultyColor(objective.difficulty || 'intermediate')}`}>
                      {objective.difficulty?.toUpperCase() || 'INT'}
                    </span>
                    <h3>{objective.title || `Objective ${index + 1}`}</h3>
                  </div>
                  <div className={styles.objectiveStatus}>
                    <span className={`${styles.statusBadge} ${getStatusColor(status)}`}>
                      {status.replace('-', ' ')}
                    </span>
                    <span className={`${styles.difficultyBadge} ${getDifficultyColor(objective.difficulty || 'intermediate')}`}>
                      {objective.difficulty || 'Intermediate'}
                    </span>
                  </div>
                </div>

                <div className={styles.masteryBar}>
                  <div className={styles.masteryFill} style={{ width: `${mastery * 100}%` }}></div>
                  <span className={styles.masteryText}>{Math.round(mastery * 100)}%</span>
                </div>

                <div className={styles.objectiveDetails}>
                  <div className={styles.detailRow}>
                    <span className={styles.detailLabel}>Mastery Level:</span>
                    <span className={styles.detailValue}>{getMasteryLevel(mastery)}</span>
                  </div>
                  <div className={styles.detailRow}>
                    <span className={styles.detailLabel}>Module:</span>
                    <span className={styles.detailValue}>{objective.module || 'N/A'}</span>
                  </div>
                  <div className={styles.detailRow}>
                    <span className={styles.detailLabel}>Week:</span>
                    <span className={styles.detailValue}>{objective.week || 'N/A'}</span>
                  </div>
                </div>
              </div>
            );
          })
        ) : (
          <div className={styles.noObjectives}>
            <p>No learning objectives available for this module.</p>
          </div>
        )}
      </div>
    </div>
  );

  const renderDetailedView = () => {
    if (!objectives || objectives.length === 0) {
      return (
        <div className={styles.noObjectives}>
          <p>No learning objectives available.</p>
        </div>
      );
    }

    return (
      <div className={styles.detailedView}>
        {getFilteredObjectives().map((objective, index) => {
          const status = getStatus(objective);
          const progress = userProgress?.find(p => p.objectiveId === objective.id);
          const mastery = progress?.mastery || 0;
          const expanded = expandedObjectives[objective.id || index];

          return (
            <div key={objective.id || index} className={styles.objectiveSection}>
              <div
                className={`${styles.objectiveCard} ${styles[status]}`}
                onClick={() => toggleObjective(objective.id || index)}
              >
                <div className={styles.objectiveHeader}>
                  <div className={styles.objectiveTitle}>
                    <span className={`${styles.weekIndicator} ${getDifficultyColor(objective.difficulty || 'intermediate')}`}>
                      {objective.difficulty?.toUpperCase() || 'INT'}
                    </span>
                    <h3>{objective.title || `Objective ${index + 1}`}</h3>
                  </div>
                  <div className={styles.objectiveStatus}>
                    <span className={`${styles.statusBadge} ${getStatusColor(status)}`}>
                      {status.replace('-', ' ')}
                    </span>
                    <span className={`${styles.difficultyBadge} ${getDifficultyColor(objective.difficulty || 'intermediate')}`}>
                      {objective.difficulty || 'Intermediate'}
                    </span>
                  </div>
                </div>

                <div className={styles.masteryBar}>
                  <div className={styles.masteryFill} style={{ width: `${mastery * 100}%` }}></div>
                  <span className={styles.masteryText}>{Math.round(mastery * 100)}%</span>
                </div>
              </div>

              {expanded && (
                <div className={styles.objectiveDetails}>
                  <div className={styles.objectiveDescription}>
                    <p>{objective.description || 'No description available.'}</p>
                  </div>

                  {objective.relatedTopics && objective.relatedTopics.length > 0 && (
                    <div className={styles.relatedTopics}>
                      <span className={styles.relatedLabel}>Related Topics:</span>
                      <div className={styles.topicsContainer}>
                        {objective.relatedTopics.map((topic, topicIndex) => (
                          <span key={topicIndex} className={styles.topicTag}>
                            {topic}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {objective.assessments && objective.assessments.length > 0 && (
                    <div className={styles.assessments}>
                      <h4>Assessments</h4>
                      <div className={styles.assessmentsList}>
                        {objective.assessments.map((assessment, assessmentIndex) => {
                          const assessmentProgress = progress?.assessments?.find(a => a.id === assessment.id);
                          return (
                            <div key={assessmentIndex} className={styles.assessmentItem}>
                              <span className={styles.assessmentTitle}>{assessment.title}</span>
                              <span className={`${styles.assessmentStatus} ${assessmentProgress?.completed ? styles.completed : styles.notCompleted}`}>
                                {assessmentProgress?.completed ? 'Completed' : 'Pending'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {objective.resources && objective.resources.length > 0 && (
                    <div className={styles.resources}>
                      <h4>Resources</h4>
                      <ul>
                        {objective.resources.map((resource, resourceIndex) => (
                          <li key={resourceIndex}>
                            <a href={resource.url} target="_blank" rel="noopener noreferrer">
                              {resource.title}
                            </a>
                            {resource.description && <p>{resource.description}</p>}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div className={styles.objectiveActions}>
                    <button className={styles.startButton}>
                      Start Learning
                    </button>
                    <button className={styles.trackProgressButton}>
                      Track Progress
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className={styles.objectiveTracker}>
      {renderOverview()}
    </div>
  );
};

export default ObjectiveTracker;