import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import json
from pathlib import Path

class LoggingService:
    """
    Comprehensive logging service for the Physical AI & Humanoid Robotics Textbook platform
    """

    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize the logging service
        """
        self.logger = logging.getLogger("physical_ai_textbook")
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler if specified
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def log_info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log an info message
        """
        self._log_message(logging.INFO, message, extra)

    def log_warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log a warning message
        """
        self._log_message(logging.WARNING, message, extra)

    def log_error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log an error message
        """
        self._log_message(logging.ERROR, message, extra)

    def log_debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log a debug message
        """
        self._log_message(logging.DEBUG, message, extra)

    def log_exception(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Log an exception with traceback
        """
        self.logger.exception(message, extra=extra)

    def _log_message(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Internal method to log messages
        """
        if extra:
            message = f"{message} | Extra: {json.dumps(extra, default=str)}"
        self.logger.log(level, message)

    def log_api_request(self, method: str, endpoint: str, user_id: Optional[int] = None,
                      response_status: Optional[int] = None, execution_time: Optional[float] = None):
        """
        Log API request details
        """
        extra = {
            "method": method,
            "endpoint": endpoint,
            "user_id": user_id,
            "response_status": response_status,
            "execution_time_ms": execution_time
        }
        self.log_info("API Request", extra=extra)

    def log_user_action(self, user_id: int, action: str, details: Optional[Dict[str, Any]] = None):
        """
        Log user actions for analytics and monitoring
        """
        extra = {
            "user_id": user_id,
            "action": action,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.log_info("User Action", extra=extra)

    def log_content_access(self, user_id: int, content_id: int, content_type: str, module: str):
        """
        Log content access for tracking user engagement
        """
        extra = {
            "user_id": user_id,
            "content_id": content_id,
            "content_type": content_type,
            "module": module,
            "access_time": datetime.utcnow().isoformat()
        }
        self.log_info("Content Access", extra=extra)

    def log_exercise_submission(self, user_id: int, exercise_id: int, grade: Optional[float] = None):
        """
        Log exercise submission for progress tracking
        """
        extra = {
            "user_id": user_id,
            "exercise_id": exercise_id,
            "grade": grade,
            "submission_time": datetime.utcnow().isoformat()
        }
        self.log_info("Exercise Submission", extra=extra)

    def log_simulation_session(self, user_id: int, simulation_name: str, status: str,
                            duration: Optional[float] = None):
        """
        Log simulation session for monitoring usage
        """
        extra = {
            "user_id": user_id,
            "simulation_name": simulation_name,
            "status": status,
            "duration_seconds": duration,
            "session_time": datetime.utcnow().isoformat()
        }
        self.log_info("Simulation Session", extra=extra)

    def log_ai_interaction(self, user_id: Optional[int], query: str, response_length: int,
                         processing_time: float, model_used: str):
        """
        Log AI interactions for performance monitoring
        """
        extra = {
            "user_id": user_id,
            "query_length": len(query),
            "response_length": response_length,
            "processing_time_ms": processing_time,
            "model_used": model_used,
            "interaction_time": datetime.utcnow().isoformat()
        }
        self.log_info("AI Interaction", extra=extra)

    def log_system_metric(self, metric_name: str, value: float, unit: str = "",
                         tags: Optional[Dict[str, str]] = None):
        """
        Log system metrics for monitoring
        """
        extra = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "tags": tags,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.log_info("System Metric", extra=extra)

    def log_performance(self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics for operations
        """
        extra = {
            "operation": operation,
            "duration_ms": duration,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.log_info("Performance", extra=extra)

    def log_security_event(self, event_type: str, user_id: Optional[int], ip_address: Optional[str] = None,
                         details: Optional[Dict[str, Any]] = None):
        """
        Log security-related events
        """
        extra = {
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.log_warning("Security Event", extra=extra)

# Global logging service instance
logging_service = LoggingService()

def get_logger():
    """
    Get the global logging service instance
    """
    return logging_service