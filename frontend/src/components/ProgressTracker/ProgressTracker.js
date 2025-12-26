import React, { useState, useEffect } from 'react';
import styles from './ProgressTracker.module.css';

const ProgressTracker = ({ userProgress, modules, onModuleSelect, onWeekSelect }) => {
  const [selectedModule, setSelectedModule] = useState(null);
  const [viewMode, setViewMode] = useState('overview'); // 'overview', 'module', 'detailed'

  useEffect(() => {
    if (modules && modules.length > 0 && !selectedModule) {
      setSelectedModule(modules[0]);
    }
  }, [modules, selectedModule]);

  const getOverallProgress = () => {
    if (!userProgress) return 0;

    const total = userProgress.totalLessons || 1;
    const completed = userProgress.completedLessons || 0;
    return Math.round((completed / total) * 100);
  };

  const getModuleProgress = (module) => {
    const total = module.totalLessons || 1;
    const completed = module.completedLessons || 0;
    return Math.round((completed / total) * 100);
  };

  const getWeekProgress = (week) => {
    const total = week.totalLessons || 1;
    const completed = week.completedLessons || 0;
    return Math.round((completed / total) * 100);
  };

  const getPerformanceMetrics = () => {
    if (!userProgress) return {};

    return {
      avgScore: userProgress.averageScore || 0,
      totalExercises: userProgress.totalExercises || 0,
      completedExercises: userProgress.completedExercises || 0,
      passingRate: userProgress.passingRate || 0
    };
  };

  const renderOverview = () => (
    <div className={styles.overview}>
      <div className={styles.overviewHeader}>
        <h3>Learning Progress Overview</h3>
        <div className={styles.timeframeSelector}>
          <button
            className={viewMode === 'overview' ? styles.active : ''}
            onClick={() => setViewMode('overview')}
          >
            Overview
          </button>
          <button
            className={viewMode === 'detailed' ? styles.active : ''}
            onClick={() => setViewMode('detailed')}
          >
            Detailed
          </button>
        </div>
      </div>

      <div className={styles.overviewStats}>
        <div className={styles.statCard}>
          <h4>Overall Progress</h4>
          <div className={styles.progressCircle}>
            <svg viewBox="0 0 36 36" className={styles.circularChart}>
              <path
                className={styles.circleBackground}
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="#e6e6e6"
                strokeWidth="3"
              />
              <path
                className={styles.circleProgress}
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="#3498db"
                strokeWidth="3"
                strokeDasharray={`${getOverallProgress()}, 100`}
              />
            </svg>
            <span className={styles.progressText}>{getOverallProgress()}%</span>
          </div>
          <p>{userProgress?.completedLessons || 0} of {userProgress?.totalLessons || 1} lessons completed</p>
        </div>

        <div className={styles.statCard}>
          <h4>Performance</h4>
          <div className={styles.performanceMetrics}>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{getPerformanceMetrics().avgScore}%</span>
              <span className={styles.metricLabel}>Avg. Score</span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{getPerformanceMetrics().passingRate}%</span>
              <span className={styles.metricLabel}>Passing Rate</span>
            </div>
          </div>
        </div>

        <div className={styles.statCard}>
          <h4>Engagement</h4>
          <div className={styles.engagementMetrics}>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{userProgress?.totalExercises || 0}</span>
              <span className={styles.metricLabel}>Exercises</span>
            </div>
            <div className={styles.metric}>
              <span className={styles.metricValue}>{userProgress?.totalHours || 0}h</span>
              <span className={styles.metricLabel}>Time Spent</span>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.modulesProgress}>
        <h4>Module Progress</h4>
        <div className={styles.modulesList}>
          {modules?.map((module, index) => (
            <div
              key={module.id}
              className={`${styles.moduleCard} ${selectedModule?.id === module.id ? styles.selected : ''}`}
              onClick={() => {
                setSelectedModule(module);
                setViewMode('detailed');
                if (onModuleSelect) onModuleSelect(module);
              }}
            >
              <div className={styles.moduleHeader}>
                <h5>{module.title}</h5>
                <span className={styles.moduleProgress}>{getModuleProgress(module)}%</span>
              </div>
              <div className={styles.moduleBar}>
                <div
                  className={styles.moduleFill}
                  style={{ width: `${getModuleProgress(module)}%` }}
                ></div>
              </div>
              <div className={styles.moduleDetails}>
                <span>{module.completedLessons || 0}/{module.totalLessons || 1} lessons</span>
                <span>{module.completedExercises || 0}/{module.totalExercises || 1} exercises</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderDetailedView = () => {
    if (!selectedModule) return renderOverview();

    return (
      <div className={styles.detailedView}>
        <div className={styles.detailedHeader}>
          <button
            className={styles.backButton}
            onClick={() => setViewMode('overview')}
          >
            ← Back to Overview
          </button>
          <h3>{selectedModule.title}</h3>
          <div className={styles.moduleProgress}>
            <span>Progress: {getModuleProgress(selectedModule)}%</span>
          </div>
        </div>

        <div className={styles.weeksProgress}>
          <h4>Weekly Progress</h4>
          <div className={styles.weeksList}>
            {selectedModule.weeks?.map((week, index) => (
              <div
                key={week.id}
                className={styles.weekCard}
                onClick={() => {
                  if (onWeekSelect) onWeekSelect(selectedModule, week);
                }}
              >
                <div className={styles.weekHeader}>
                  <h5>Week {week.number}: {week.title}</h5>
                  <span className={styles.weekProgress}>{getWeekProgress(week)}%</span>
                </div>
                <div className={styles.weekBar}>
                  <div
                    className={styles.weekFill}
                    style={{ width: `${getWeekProgress(week)}%` }}
                  ></div>
                </div>
                <div className={styles.weekDetails}>
                  <span>{week.completedLessons || 0}/{week.totalLessons || 1} lessons</span>
                  <span>{week.exerciseCount || 0} exercises</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className={styles.lessonsProgress}>
          <h4>Lesson Progress</h4>
          <div className={styles.lessonsGrid}>
            {selectedModule.lessons?.map((lesson, index) => (
              <div
                key={lesson.id}
                className={`${styles.lessonCard} ${lesson.completed ? styles.completed : ''}`}
              >
                <div className={styles.lessonHeader}>
                  <span className={styles.lessonTitle}>{lesson.title}</span>
                  <span className={`${styles.lessonStatus} ${lesson.completed ? styles.completed : styles.notCompleted}`}>
                    {lesson.completed ? '✓' : '○'}
                  </span>
                </div>
                <div className={styles.lessonDetails}>
                  <span className={styles.lessonType}>{lesson.type}</span>
                  <span className={styles.lessonDuration}>{lesson.duration} min</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className={styles.progressTracker}>
      {viewMode === 'overview' || !selectedModule ? renderOverview() : renderDetailedView()}
    </div>
  );
};

export default ProgressTracker;