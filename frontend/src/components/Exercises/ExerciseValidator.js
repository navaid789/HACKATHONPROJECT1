import React, { useState, useEffect } from 'react';
import styles from './ExerciseValidator.module.css';

const ExerciseValidator = ({ exercise, submission, onValidationComplete }) => {
  const [validationResult, setValidationResult] = useState(null);
  const [isChecking, setIsChecking] = useState(false);
  const [validationSteps, setValidationSteps] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (submission && submission.submission) {
      validateSubmission(submission.submission);
    }
  }, [submission]);

  const validateSubmission = async (code) => {
    setIsChecking(true);

    // Simulate multi-step validation process
    const steps = [
      { name: 'Syntax Check', status: 'pending', description: 'Checking for syntax errors' },
      { name: 'Logic Validation', status: 'pending', description: 'Validating program logic' },
      { name: 'Output Verification', status: 'pending', description: 'Verifying expected output' },
      { name: 'Performance Check', status: 'pending', description: 'Checking for efficiency' },
      { name: 'Style Review', status: 'pending', description: 'Reviewing code style and best practices' }
    ];

    setValidationSteps(steps);

    // Simulate each validation step
    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i);
      setValidationSteps(prev => prev.map((step, idx) =>
        idx === i ? { ...step, status: 'checking' } : step
      ));

      // Simulate API call to validation service
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Determine if this step passes (random for demo)
      const passes = Math.random() > 0.2; // 80% success rate for demo

      setValidationSteps(prev => prev.map((step, idx) =>
        idx === i ? { ...step, status: passes ? 'passed' : 'failed' } : step
      ));
    }

    // Generate final validation result
    const passedSteps = validationSteps.filter(step => step.status === 'passed').length;
    const totalSteps = validationSteps.length;
    const overallScore = Math.round((passedSteps / totalSteps) * 100);

    const result = {
      passed: overallScore >= 60,
      score: overallScore,
      totalSteps,
      passedSteps,
      failedSteps: totalSteps - passedSteps,
      feedback: generateFeedback(overallScore, validationSteps),
      detailedFeedback: validationSteps,
      timestamp: new Date().toISOString()
    };

    setValidationResult(result);
    setIsChecking(false);

    if (onValidationComplete) {
      onValidationComplete(result);
    }
  };

  const generateFeedback = (score, steps) => {
    if (score >= 90) {
      return "Excellent work! Your solution meets all requirements and follows best practices.";
    } else if (score >= 70) {
      return "Good job! Your solution works correctly but could benefit from some improvements.";
    } else if (score >= 50) {
      return "Your solution has some issues that need to be addressed. Review the feedback below.";
    } else {
      return "Your solution needs significant improvements. Please review the requirements and feedback.";
    }
  };

  const getStepColor = (status) => {
    switch (status) {
      case 'passed': return 'var(--ifm-color-success)';
      case 'failed': return 'var(--ifm-color-danger)';
      case 'checking': return 'var(--ifm-color-warning)';
      default: return 'var(--ifm-color-emphasis-400)';
    }
  };

  const getStepIcon = (status) => {
    switch (status) {
      case 'passed': return '✓';
      case 'failed': return '✗';
      case 'checking': return '⏳';
      default: return '○';
    }
  };

  const getStepLabel = (status) => {
    switch (status) {
      case 'passed': return 'Passed';
      case 'failed': return 'Failed';
      case 'checking': return 'Checking...';
      default: return 'Pending';
    }
  };

  if (!submission) {
    return (
      <div className={styles.exerciseValidator}>
        <div className={styles.noSubmission}>
          <p>No submission to validate.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.exerciseValidator}>
      <div className={styles.header}>
        <h2>Exercise Validation</h2>
        <div className={styles.exerciseInfo}>
          <span className={styles.exerciseTitle}>{exercise.title}</span>
          <span className={styles.exerciseModule}>Module: {exercise.module}</span>
        </div>
      </div>

      <div className={styles.submissionInfo}>
        <div className={styles.infoItem}>
          <span className={styles.label}>Submitted:</span>
          <span className={styles.value}>
            {submission.submittedAt ? new Date(submission.submittedAt).toLocaleString() : 'Just now'}
          </span>
        </div>
        <div className={styles.infoItem}>
          <span className={styles.label}>Language:</span>
          <span className={styles.value}>{submission.language || 'Python'}</span>
        </div>
      </div>

      {isChecking && (
        <div className={styles.validationProgress}>
          <h3>Validating Submission...</h3>
          <div className={styles.progressSteps}>
            {validationSteps.map((step, index) => (
              <div
                key={index}
                className={`${styles.progressStep} ${index <= currentStep ? styles.active : ''}`}
              >
                <div
                  className={styles.stepIndicator}
                  style={{
                    backgroundColor: index < currentStep ? 'var(--ifm-color-success)' :
                                   index === currentStep ? 'var(--ifm-color-warning)' :
                                   'var(--ifm-color-emphasis-200)'
                  }}
                >
                  {index < currentStep ? '✓' : index === currentStep ? '⏳' : index + 1}
                </div>
                <div className={styles.stepText}>
                  <div className={styles.stepName}>{step.name}</div>
                  <div className={styles.stepStatus}>
                    {index < currentStep ? 'Completed' :
                     index === currentStep ? 'In Progress' :
                     'Pending'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {validationResult && (
        <div className={styles.validationResults}>
          <div className={`${styles.overallResult} ${validationResult.passed ? styles.passed : styles.failed}`}>
            <div className={styles.resultHeader}>
              <h3>Validation Results</h3>
              <div className={styles.scoreBadge}>
                <span className={styles.scoreValue}>{validationResult.score}%</span>
                <span className={styles.scoreLabel}>Score</span>
              </div>
            </div>

            <div className={styles.resultSummary}>
              <div className={styles.summaryItem}>
                <span className={styles.summaryNumber}>{validationResult.passedSteps}</span>
                <span className={styles.summaryLabel}>Passed</span>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryNumber}>{validationResult.failedSteps}</span>
                <span className={styles.summaryLabel}>Failed</span>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryNumber}>{validationResult.totalSteps}</span>
                <span className={styles.summaryLabel}>Total</span>
              </div>
            </div>

            <div className={styles.overallFeedback}>
              <h4>Overall Feedback</h4>
              <p>{validationResult.feedback}</p>
            </div>
          </div>

          <div className={styles.detailedResults}>
            <h3>Detailed Validation Steps</h3>
            <div className={styles.validationStepsList}>
              {validationResult.detailedFeedback.map((step, index) => (
                <div
                  key={index}
                  className={`${styles.validationStep} ${styles[step.status]}`}
                  style={{ borderLeft: `4px solid ${getStepColor(step.status)}` }}
                >
                  <div className={styles.stepHeader}>
                    <div className={styles.stepIcon} style={{ color: getStepColor(step.status) }}>
                      {getStepIcon(step.status)}
                    </div>
                    <div className={styles.stepInfo}>
                      <h4>{step.name}</h4>
                      <span className={styles.stepStatus} style={{ color: getStepColor(step.status) }}>
                        {getStepLabel(step.status)}
                      </span>
                    </div>
                  </div>

                  <div className={styles.stepDescription}>
                    {step.description}
                  </div>

                  {step.status === 'failed' && (
                    <div className={styles.failureDetails}>
                      <h5>Issues Found:</h5>
                      <ul>
                        <li>Example issue: {step.name} did not meet requirements</li>
                        <li>Recommendation: Review the {step.name.toLowerCase()} guidelines</li>
                      </ul>
                    </div>
                  )}

                  {step.status === 'passed' && (
                    <div className={styles.successDetails}>
                      <p>✓ Requirements successfully met</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className={styles.recommendations}>
            <h3>Recommendations</h3>
            <ul>
              {validationResult.failedSteps > 0 && (
                <li>Review the failed validation steps and address the issues</li>
              )}
              {validationResult.score < 80 && (
                <li>Consider refactoring your code for better performance and readability</li>
              )}
              {validationResult.score >= 80 && (
                <li>Great work! Your solution meets the requirements.</li>
              )}
              <li>Test your solution with additional edge cases</li>
            </ul>
          </div>

          <div className={styles.actions}>
            <button className={styles.resubmitButton}>
              Resubmit Solution
            </button>
            <button className={styles.reviewButton}>
              Review Requirements
            </button>
            <button className={styles.downloadButton}>
              Download Report
            </button>
          </div>
        </div>
      )}

      <div className={styles.submissionCode}>
        <h3>Submitted Code</h3>
        <pre className={styles.codeBlock}>
          <code>
            {submission.submission || 'No code submitted'}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default ExerciseValidator;