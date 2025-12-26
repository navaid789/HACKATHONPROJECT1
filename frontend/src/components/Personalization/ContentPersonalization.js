import React, { useState, useEffect } from 'react';
import styles from './ContentPersonalization.module.css';

const ContentPersonalization = ({ user, content, onContentFilter }) => {
  const [preferences, setPreferences] = useState({
    learningStyle: 'mixed',
    difficultyLevel: 'intermediate',
    focusArea: 'balanced',
    learningPace: 'moderate',
    preferredTopics: []
  });

  const [filteredContent, setFilteredContent] = useState(content || []);
  const [activeFilters, setActiveFilters] = useState({});

  useEffect(() => {
    if (user && user.learningPreferences) {
      setPreferences(prev => ({
        ...prev,
        ...user.learningPreferences
      }));
    }
  }, [user]);

  useEffect(() => {
    applyFilters();
  }, [preferences, content]);

  const applyFilters = () => {
    if (!content) {
      setFilteredContent([]);
      return;
    }

    let filtered = [...content];

    // Apply learning style filter
    if (preferences.learningStyle && preferences.learningStyle !== 'mixed') {
      filtered = filtered.filter(item =>
        item.tags?.includes(preferences.learningStyle) ||
        item.metadata?.learningStyle === preferences.learningStyle
      );
    }

    // Apply difficulty level filter
    if (preferences.difficultyLevel) {
      filtered = filtered.filter(item =>
        item.difficulty === preferences.difficultyLevel ||
        item.metadata?.difficulty === preferences.difficultyLevel
      );
    }

    // Apply focus area filter
    if (preferences.focusArea && preferences.focusArea !== 'balanced') {
      filtered = filtered.filter(item =>
        item.tags?.includes(preferences.focusArea) ||
        item.metadata?.focusArea === preferences.focusArea
      );
    }

    setFilteredContent(filtered);

    if (onContentFilter) {
      onContentFilter(filtered);
    }
  };

  const handlePreferenceChange = (preference, value) => {
    setPreferences(prev => ({
      ...prev,
      [preference]: value
    }));
  };

  const toggleFilter = (filterType, filterValue) => {
    setActiveFilters(prev => {
      const newFilters = { ...prev };
      if (newFilters[filterType] === filterValue) {
        delete newFilters[filterType];
      } else {
        newFilters[filterType] = filterValue;
      }
      return newFilters;
    });
  };

  const getRecommendationReason = (contentItem) => {
    // Generate a reason for why this content is recommended
    const reasons = [];

    if (preferences.learningStyle && contentItem.tags?.includes(preferences.learningStyle)) {
      reasons.push(`matches your ${preferences.learningStyle} learning style`);
    }

    if (preferences.focusArea && contentItem.tags?.includes(preferences.focusArea)) {
      reasons.push(`aligns with your ${preferences.focusArea} focus`);
    }

    if (contentItem.difficulty === preferences.difficultyLevel) {
      reasons.push(`at your preferred difficulty level`);
    }

    return reasons.length > 0 ? `Recommended because it ${reasons.join(' and ')}.` : '';
  };

  return (
    <div className={styles.contentPersonalization}>
      <div className={styles.personalizationHeader}>
        <h3>Personalized Learning Experience</h3>
        <p>Content tailored to your learning preferences and progress</p>
      </div>

      <div className={styles.preferencesPanel}>
        <div className={styles.preferenceGroup}>
          <label className={styles.preferenceLabel}>Learning Style</label>
          <div className={styles.preferenceOptions}>
            {['visual', 'hands_on', 'theoretical', 'mixed'].map(style => (
              <button
                key={style}
                className={`${styles.preferenceOption} ${
                  preferences.learningStyle === style ? styles.active : ''
                }`}
                onClick={() => handlePreferenceChange('learningStyle', style)}
              >
                {style.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.preferenceGroup}>
          <label className={styles.preferenceLabel}>Difficulty Level</label>
          <div className={styles.preferenceOptions}>
            {['beginner', 'intermediate', 'advanced'].map(level => (
              <button
                key={level}
                className={`${styles.preferenceOption} ${
                  preferences.difficultyLevel === level ? styles.active : ''
                }`}
                onClick={() => handlePreferenceChange('difficultyLevel', level)}
              >
                {level.charAt(0).toUpperCase() + level.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.preferenceGroup}>
          <label className={styles.preferenceLabel}>Focus Area</label>
          <div className={styles.preferenceOptions}>
            {['theory', 'practice', 'balanced'].map(area => (
              <button
                key={area}
                className={`${styles.preferenceOption} ${
                  preferences.focusArea === area ? styles.active : ''
                }`}
                onClick={() => handlePreferenceChange('focusArea', area)}
              >
                {area.charAt(0).toUpperCase() + area.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.preferenceGroup}>
          <label className={styles.preferenceLabel}>Learning Pace</label>
          <div className={styles.preferenceOptions}>
            {['slow', 'moderate', 'fast'].map(pace => (
              <button
                key={pace}
                className={`${styles.preferenceOption} ${
                  preferences.learningPace === pace ? styles.active : ''
                }`}
                onClick={() => handlePreferenceChange('learningPace', pace)}
              >
                {pace.charAt(0).toUpperCase() + pace.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.contentRecommendations}>
        <div className={styles.recommendationsHeader}>
          <h4>Recommended Content</h4>
          <span className={styles.recommendationCount}>
            {filteredContent.length} items recommended
          </span>
        </div>

        <div className={styles.recommendationsList}>
          {filteredContent.length > 0 ? (
            filteredContent.map((item, index) => (
              <div key={index} className={styles.recommendationItem}>
                <div className={styles.recommendationContent}>
                  <h5 className={styles.recommendationTitle}>{item.title}</h5>
                  <p className={styles.recommendationDescription}>
                    {item.description || item.content?.substring(0, 150) + '...'}
                  </p>
                  <div className={styles.recommendationTags}>
                    {item.tags?.map((tag, tagIndex) => (
                      <span key={tagIndex} className={styles.tag}>
                        {tag}
                      </span>
                    ))}
                  </div>
                  <p className={styles.recommendationReason}>
                    {getRecommendationReason(item)}
                  </p>
                </div>
                <div className={styles.recommendationActions}>
                  <button className={styles.startButton}>
                    Start Learning
                  </button>
                  <button className={styles.saveButton}>
                    Save for Later
                  </button>
                </div>
              </div>
            ))
          ) : (
            <div className={styles.noRecommendations}>
              <p>No content matches your current preferences. Try adjusting your filters.</p>
            </div>
          )}
        </div>
      </div>

      <div className={styles.learningPath}>
        <h4>Your Learning Path</h4>
        <div className={styles.pathProgress}>
          <div className={styles.pathItem}>
            <div className={styles.pathCircle}>1</div>
            <div className={styles.pathContent}>
              <h5>Complete Profile</h5>
              <p>Set your learning preferences</p>
            </div>
          </div>
          <div className={styles.pathItem}>
            <div className={styles.pathCircle}>2</div>
            <div className={styles.pathContent}>
              <h5>Explore Content</h5>
              <p>Review recommended materials</p>
            </div>
          </div>
          <div className={styles.pathItem}>
            <div className={styles.pathCircle}>3</div>
            <div className={styles.pathContent}>
              <h5>Start Learning</h5>
              <p>Begin with recommended content</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContentPersonalization;