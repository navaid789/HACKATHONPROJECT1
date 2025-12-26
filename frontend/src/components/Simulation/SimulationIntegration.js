import React, { useState, useEffect } from 'react';
import styles from './SimulationIntegration.module.css';

const SimulationIntegration = ({ simulation, onRun, onStop, onReset }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [sessionStatus, setSessionStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [parameters, setParameters] = useState({});
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    if (simulation?.defaultParameters) {
      setParameters(simulation.defaultParameters);
    }
  }, [simulation]);

  const handleRun = async () => {
    setIsRunning(true);
    setSessionStatus('running');
    setProgress(0);

    // Add log entry
    addLog('Simulation started', 'info');

    try {
      if (onRun) {
        await onRun(simulation.id, parameters);
      }

      // Simulate progress
      const interval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + 5;
          if (newProgress >= 100) {
            clearInterval(interval);
            setIsRunning(false);
            setSessionStatus('completed');
            addLog('Simulation completed successfully', 'success');

            // Generate mock results
            setResults({
              success: true,
              metrics: {
                completionTime: '2.5s',
                accuracy: '98.7%',
                efficiency: '94.2%'
              },
              summary: 'Simulation executed successfully with optimal results.'
            });
            return 100;
          }
          return newProgress;
        });
      }, 200);
    } catch (error) {
      setSessionStatus('error');
      addLog(`Error running simulation: ${error.message}`, 'error');
    }
  };

  const handleStop = () => {
    setIsRunning(false);
    setIsPaused(false);
    setSessionStatus('stopped');
    addLog('Simulation stopped by user', 'info');

    if (onStop) {
      onStop(simulation.id);
    }
  };

  const handlePause = () => {
    setIsPaused(!isPaused);
    addLog(isPaused ? 'Simulation resumed' : 'Simulation paused', 'info');
  };

  const handleReset = () => {
    setIsRunning(false);
    setIsPaused(false);
    setSessionStatus('idle');
    setProgress(0);
    setResults(null);
    addLog('Simulation reset', 'info');

    if (onReset) {
      onReset(simulation.id);
    }
  };

  const handleParameterChange = (paramName, value) => {
    setParameters(prev => ({
      ...prev,
      [paramName]: value
    }));
  };

  const addLog = (message, level = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog = { timestamp, message, level };
    setLogs(prev => [newLog, ...prev.slice(0, 49)]); // Keep only last 50 logs
  };

  const renderParameterControls = () => {
    if (!simulation?.parameters) return null;

    return (
      <div className={styles.parametersSection}>
        <h4>Simulation Parameters</h4>
        <div className={styles.parametersGrid}>
          {Object.entries(simulation.parameters).map(([paramName, paramConfig]) => (
            <div key={paramName} className={styles.parameterControl}>
              <label className={styles.parameterLabel}>
                {paramConfig.label || paramName}
                {paramConfig.unit && <span className={styles.unit}>{paramConfig.unit}</span>}
              </label>
              {paramConfig.type === 'slider' ? (
                <input
                  type="range"
                  min={paramConfig.min}
                  max={paramConfig.max}
                  step={paramConfig.step}
                  value={parameters[paramName] || paramConfig.default}
                  onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value))}
                  className={styles.slider}
                />
              ) : paramConfig.type === 'select' ? (
                <select
                  value={parameters[paramName] || paramConfig.default}
                  onChange={(e) => handleParameterChange(paramName, e.target.value)}
                  className={styles.select}
                >
                  {paramConfig.options?.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={parameters[paramName] || paramConfig.default}
                  onChange={(e) => handleParameterChange(paramName, parseFloat(e.target.value))}
                  className={styles.numberInput}
                />
              )}
              <span className={styles.parameterValue}>
                {parameters[paramName] || paramConfig.default}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className={styles.simulationIntegration}>
      <div className={styles.simulationHeader}>
        <h2>{simulation?.title || 'Simulation Environment'}</h2>
        <div className={styles.simulationMeta}>
          <span className={styles.simulationType}>
            Type: {simulation?.type || 'Custom'}
          </span>
          <span className={styles.simulationStatus}>
            Status:
            <span className={`${styles.statusBadge} ${styles[sessionStatus]}`}>
              {sessionStatus.charAt(0).toUpperCase() + sessionStatus.slice(1)}
            </span>
          </span>
        </div>
      </div>

      <div className={styles.simulationDescription}>
        <h3>About this Simulation</h3>
        <p>{simulation?.description || 'No description available.'}</p>
      </div>

      {renderParameterControls()}

      <div className={styles.simulationControls}>
        <div className={styles.controlButtons}>
          <button
            className={`${styles.controlButton} ${styles.runButton}`}
            onClick={handleRun}
            disabled={isRunning}
          >
            {isRunning ? 'Running...' : 'Run Simulation'}
          </button>

          <button
            className={`${styles.controlButton} ${styles.pauseButton}`}
            onClick={handlePause}
            disabled={!isRunning}
          >
            {isPaused ? 'Resume' : 'Pause'}
          </button>

          <button
            className={`${styles.controlButton} ${styles.stopButton}`}
            onClick={handleStop}
            disabled={!isRunning}
          >
            Stop
          </button>

          <button
            className={`${styles.controlButton} ${styles.resetButton}`}
            onClick={handleReset}
          >
            Reset
          </button>
        </div>

        {isRunning && (
          <div className={styles.progressBar}>
            <div
              className={styles.progressFill}
              style={{ width: `${progress}%` }}
            ></div>
            <span className={styles.progressText}>{Math.round(progress)}%</span>
          </div>
        )}
      </div>

      <div className={styles.simulationViewer}>
        <div className={styles.simulationPlaceholder}>
          <div className={styles.simulationVisual}>
            {simulation?.type === 'gazebo' && '🤖 Gazebo Simulation Environment'}
            {simulation?.type === 'isaac' && '🎮 NVIDIA Isaac Simulation'}
            {simulation?.type === 'custom' && '🔬 Custom Simulation Environment'}
            {(!simulation?.type || simulation.type === 'default') && '🧪 Simulation Environment'}
          </div>
          <div className={styles.simulationInfo}>
            <p><strong>Environment:</strong> {simulation?.environment || 'Default'}</p>
            <p><strong>Duration:</strong> {simulation?.duration || 'N/A'}</p>
            <p><strong>Version:</strong> {simulation?.version || '1.0.0'}</p>
          </div>
          {isRunning && (
            <div className={styles.runningIndicator}>
              <div className={styles.spinner}></div>
              <p>Simulation Running...</p>
            </div>
          )}
        </div>
      </div>

      {results && (
        <div className={styles.resultsSection}>
          <h3>Simulation Results</h3>
          <div className={styles.resultsGrid}>
            {results.metrics && Object.entries(results.metrics).map(([key, value]) => (
              <div key={key} className={styles.metricCard}>
                <h4>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</h4>
                <p>{value}</p>
              </div>
            ))}
          </div>
          {results.summary && (
            <div className={styles.resultsSummary}>
              <h4>Summary</h4>
              <p>{results.summary}</p>
            </div>
          )}
        </div>
      )}

      <div className={styles.logsSection}>
        <h3>Simulation Logs</h3>
        <div className={styles.logsContainer}>
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div key={index} className={`${styles.logEntry} ${styles[log.level]}`}>
                <span className={styles.logTimestamp}>[{log.timestamp}]</span>
                <span className={styles.logMessage}>{log.message}</span>
              </div>
            ))
          ) : (
            <p className={styles.noLogs}>No logs yet. Start a simulation to see logs.</p>
          )}
        </div>
      </div>

      <div className={styles.simulationInstructions}>
        <h3>Instructions</h3>
        <ol>
          <li>Configure simulation parameters as needed</li>
          <li>Click "Run Simulation" to start the simulation</li>
          <li>Monitor progress and logs in real-time</li>
          <li>Use Pause/Stop/Reset controls as needed</li>
          <li>Review results after simulation completes</li>
        </ol>
      </div>
    </div>
  );
};

export default SimulationIntegration;