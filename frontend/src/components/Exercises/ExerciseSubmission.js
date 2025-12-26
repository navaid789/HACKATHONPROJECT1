import React, { useState } from 'react';
import styles from './ExerciseSubmission.module.css';

const ExerciseSubmission = ({ exercise, onSubmit, onValidate }) => {
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('python');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionResult, setSubmissionResult] = useState(null);
  const [activeTab, setActiveTab] = useState('editor');

  const handleSubmit = async () => {
    if (!code.trim()) {
      alert('Please enter your solution before submitting.');
      return;
    }

    setIsSubmitting(true);

    try {
      // In a real implementation, this would call the backend API
      // For now, we'll simulate a submission
      const submissionData = {
        exercise_id: exercise.id,
        submission_content: code,
        language: language,
        submitted_at: new Date().toISOString()
      };

      if (onSubmit) {
        const result = await onSubmit(submissionData);
        setSubmissionResult(result);
      } else {
        // Simulate a successful submission
        setTimeout(() => {
          setSubmissionResult({
            success: true,
            message: 'Exercise submitted successfully!',
            submission_id: Math.floor(Math.random() * 10000),
            submitted_at: new Date().toISOString()
          });
          setIsSubmitting(false);
        }, 1500);
      }
    } catch (error) {
      setSubmissionResult({
        success: false,
        message: 'Error submitting exercise: ' + error.message
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleValidate = async () => {
    if (!code.trim()) {
      alert('Please enter your solution before validating.');
      return;
    }

    try {
      // In a real implementation, this would call the validation API
      // For now, we'll simulate validation
      if (onValidate) {
        const result = await onValidate(code, language);
        setSubmissionResult(result);
      } else {
        // Simulate validation result
        setSubmissionResult({
          success: true,
          message: 'Validation completed!',
          feedback: 'Your solution looks good. Consider adding more comments for clarity.',
          score: 85,
          validation_passed: true
        });
      }
    } catch (error) {
      setSubmissionResult({
        success: false,
        message: 'Error validating exercise: ' + error.message
      });
    }
  };

  const handleReset = () => {
    setCode('');
    setSubmissionResult(null);
  };

  const handleLanguageChange = (newLanguage) => {
    setLanguage(newLanguage);
  };

  const insertTemplate = (template) => {
    setCode(prev => prev + template);
  };

  return (
    <div className={styles.exerciseSubmission}>
      <div className={styles.exerciseHeader}>
        <h2>{exercise?.title || 'Exercise'}</h2>
        <div className={styles.exerciseMeta}>
          <span className={styles.moduleTag}>Module: {exercise?.module || 'N/A'}</span>
          <span className={styles.difficultyTag}>Difficulty: {exercise?.difficulty || 'Intermediate'}</span>
        </div>
      </div>

      <div className={styles.exerciseDescription}>
        <h3>Exercise Description</h3>
        <p>{exercise?.description || 'No description available.'}</p>
        {exercise?.requirements && (
          <div className={styles.requirements}>
            <h4>Requirements:</h4>
            <ul>
              {exercise.requirements.map((req, index) => (
                <li key={index}>{req}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className={styles.submissionArea}>
        <div className={styles.editorHeader}>
          <div className={styles.editorControls}>
            <div className={styles.languageSelector}>
              <label htmlFor="language">Language:</label>
              <select
                id="language"
                value={language}
                onChange={(e) => handleLanguageChange(e.target.value)}
                className={styles.languageSelect}
              >
                <option value="python">Python</option>
                <option value="cpp">C++</option>
                <option value="javascript">JavaScript</option>
                <option value="java">Java</option>
              </select>
            </div>
            <div className={styles.editorTabs}>
              <button
                className={`${styles.tabButton} ${activeTab === 'editor' ? styles.activeTab : ''}`}
                onClick={() => setActiveTab('editor')}
              >
                Editor
              </button>
              <button
                className={`${styles.tabButton} ${activeTab === 'preview' ? styles.activeTab : ''}`}
                onClick={() => setActiveTab('preview')}
              >
                Preview
              </button>
            </div>
          </div>
          <div className={styles.editorActions}>
            <button
              className={styles.templateButton}
              onClick={() => insertTemplate('# Start your solution here\n')}
            >
              Insert Template
            </button>
          </div>
        </div>

        <div className={styles.editorContainer}>
          {activeTab === 'editor' ? (
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              className={styles.codeEditor}
              placeholder={`// Write your ${language} solution here...`}
              rows={20}
            />
          ) : (
            <div className={styles.codePreview}>
              <pre className={styles.previewContent}>
                <code>{code || '// No code to preview'}</code>
              </pre>
            </div>
          )}
        </div>

        <div className={styles.submissionControls}>
          <button
            className={styles.validateButton}
            onClick={handleValidate}
            disabled={isSubmitting}
          >
            Validate Solution
          </button>
          <button
            className={styles.submitButton}
            onClick={handleSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Submitting...' : 'Submit Solution'}
          </button>
          <button
            className={styles.resetButton}
            onClick={handleReset}
          >
            Reset
          </button>
        </div>
      </div>

      {submissionResult && (
        <div className={`${styles.submissionResult} ${submissionResult.success ? styles.success : styles.error}`}>
          <div className={styles.resultHeader}>
            <h3>{submissionResult.success ? 'Success!' : 'Error'}</h3>
            {submissionResult.score && (
              <div className={styles.scoreBadge}>
                Score: {submissionResult.score}/100
              </div>
            )}
          </div>
          <p>{submissionResult.message}</p>
          {submissionResult.feedback && (
            <div className={styles.feedback}>
              <h4>Feedback:</h4>
              <p>{submissionResult.feedback}</p>
            </div>
          )}
          {submissionResult.validation_passed === false && (
            <div className={styles.validationIssues}>
              <h4>Issues Found:</h4>
              <ul>
                <li>Missing required functionality</li>
                <li>Code does not meet requirements</li>
                <li>Consider refactoring for better efficiency</li>
              </ul>
            </div>
          )}
        </div>
      )}

      {exercise?.example_solution && (
        <div className={styles.exampleSolution}>
          <h3>Example Solution</h3>
          <pre className={styles.exampleCode}>
            <code>{exercise.example_solution}</code>
          </pre>
        </div>
      )}
    </div>
  );
};

export default ExerciseSubmission;