import React, { useState } from 'react';
import styles from './TextbookNavigation.module.css';

const TextbookNavigation = ({ modules, currentModule, currentWeek, onModuleChange, onWeekChange }) => {
  const [expandedModules, setExpandedModules] = useState({});

  const toggleModule = (moduleId) => {
    setExpandedModules(prev => ({
      ...prev,
      [moduleId]: !prev[moduleId]
    }));
  };

  const handleModuleSelect = (moduleId) => {
    if (onModuleChange) {
      onModuleChange(moduleId);
    }
    // Expand the selected module
    setExpandedModules(prev => ({
      ...prev,
      [moduleId]: true
    }));
  };

  const handleWeekSelect = (moduleId, weekId) => {
    if (onWeekChange) {
      onWeekChange(moduleId, weekId);
    }
  };

  return (
    <div className={styles.textbookNavigation}>
      <div className={styles.navigationHeader}>
        <h3>Textbook Navigation</h3>
        <div className={styles.navigationStats}>
          <span className={styles.moduleCount}>{modules.length} Modules</span>
        </div>
      </div>

      <div className={styles.modulesList}>
        {modules.map((module) => (
          <div key={module.id} className={styles.moduleItem}>
            <div
              className={`${styles.moduleHeader} ${
                currentModule === module.id ? styles.currentModule : ''
              }`}
              onClick={() => toggleModule(module.id)}
            >
              <div className={styles.moduleInfo}>
                <h4 className={styles.moduleTitle}>{module.title}</h4>
                <div className={styles.moduleMeta}>
                  <span className={styles.weekCount}>{module.weeks.length} Weeks</span>
                  <span className={styles.lessonCount}>{module.totalLessons} Lessons</span>
                </div>
              </div>
              <div className={styles.moduleActions}>
                <button
                  className={`${styles.expandButton} ${
                    expandedModules[module.id] ? styles.expanded : ''
                  }`}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleModule(module.id);
                  }}
                >
                  <span className={styles.expandIcon}>▼</span>
                </button>
              </div>
            </div>

            {expandedModules[module.id] && (
              <div className={styles.weeksList}>
                {module.weeks.map((week) => (
                  <div
                    key={week.id}
                    className={`${styles.weekItem} ${
                      currentModule === module.id && currentWeek === week.id
                        ? styles.currentWeek
                        : ''
                    }`}
                    onClick={() => handleWeekSelect(module.id, week.id)}
                  >
                    <div className={styles.weekInfo}>
                      <span className={styles.weekNumber}>Week {week.number}</span>
                      <span className={styles.weekTitle}>{week.title}</span>
                    </div>
                    <div className={styles.weekMeta}>
                      <span className={styles.lessonCount}>{week.lessons.length} Lessons</span>
                      {week.exercise && (
                        <span className={styles.exerciseBadge}>Exercise</span>
                      )}
                      {week.quiz && (
                        <span className={styles.quizBadge}>Quiz</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className={styles.progressSection}>
        <h4>Learning Progress</h4>
        <div className={styles.progressOverview}>
          <div className={styles.progressItem}>
            <span className={styles.progressLabel}>Modules Completed</span>
            <span className={styles.progressValue}>2/4</span>
          </div>
          <div className={styles.progressItem}>
            <span className={styles.progressLabel}>Weeks Completed</span>
            <span className={styles.progressValue}>6/13</span>
          </div>
          <div className={styles.progressItem}>
            <span className={styles.progressLabel}>Overall Progress</span>
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{ width: '46%' }}></div>
            </div>
            <span className={styles.progressPercentage}>46%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TextbookNavigation;