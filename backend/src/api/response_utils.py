from typing import Any, Dict, Optional
from datetime import datetime
from fastapi import HTTPException, status
import json

def create_success_response(
    data: Any = None,
    message: str = "Request successful",
    status_code: int = 200
) -> Dict[str, Any]:
    """
    Create a standardized success response
    """
    return {
        "success": True,
        "status_code": status_code,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }


def create_error_response(
    message: str = "An error occurred",
    status_code: int = 400,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized error response
    """
    response = {
        "success": False,
        "status_code": status_code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }

    if error_code:
        response["error_code"] = error_code

    if details:
        response["details"] = details

    return response


def format_content_response(content) -> Dict[str, Any]:
    """
    Format content object for API response
    """
    return {
        "id": content.id,
        "title": content.title,
        "slug": content.slug,
        "content_type": content.content_type,
        "module": content.module,
        "week": content.week,
        "content": content.content,
        "is_published": content.is_published,
        "author_id": content.author_id,
        "prerequisites": content.prerequisites,
        "learning_objectives": content.learning_objectives,
        "duration_minutes": content.duration_minutes,
        "created_at": content.created_at.isoformat() if content.created_at else None,
        "updated_at": content.updated_at.isoformat() if content.updated_at else None
    }


def format_user_response(user, include_email: bool = False) -> Dict[str, Any]:
    """
    Format user object for API response
    """
    response = {
        "id": user.id,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": user.is_active,
        "is_verified": user.is_verified,
        "background": user.background,
        "learning_preferences": user.learning_preferences,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None
    }

    if include_email:
        response["email"] = user.email

    return response


def format_exercise_submission_response(submission) -> Dict[str, Any]:
    """
    Format exercise submission object for API response
    """
    return {
        "id": submission.id,
        "user_id": submission.user_id,
        "exercise_id": submission.exercise_id,
        "grade": submission.grade,
        "feedback": submission.feedback,
        "is_graded": submission.is_graded,
        "submitted_at": submission.submitted_at.isoformat() if submission.submitted_at else None,
        "graded_at": submission.graded_at.isoformat() if submission.graded_at else None,
        "created_at": submission.created_at.isoformat() if submission.created_at else None,
        "updated_at": submission.updated_at.isoformat() if submission.updated_at else None
    }


def format_simulation_session_response(session) -> Dict[str, Any]:
    """
    Format simulation session object for API response
    """
    return {
        "id": session.id,
        "user_id": session.user_id,
        "simulation_name": session.simulation_name,
        "simulation_type": session.simulation_type,
        "parameters": session.parameters,
        "results": session.results,
        "status": session.status,
        "started_at": session.started_at.isoformat() if session.started_at else None,
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "duration": session.duration,
        "is_active": session.is_active,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None
    }


def handle_error(
    error: Exception,
    message: str = "An error occurred processing your request",
    status_code: int = 500
) -> Dict[str, Any]:
    """
    Handle exceptions and return standardized error response
    """
    print(f"Error occurred: {str(error)}")  # Log the error for debugging
    return create_error_response(message=message, status_code=status_code)


def validate_and_format_input(data: Dict[str, Any], required_fields: list) -> Dict[str, Any]:
    """
    Validate required fields in input data and format it
    """
    for field in required_fields:
        if field not in data or data[field] is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Required field '{field}' is missing"
            )

    # Additional validation can be added here
    return data


def paginate_response(
    items: list,
    page: int,
    page_size: int,
    total_count: int
) -> Dict[str, Any]:
    """
    Create a paginated response
    """
    total_pages = (total_count + page_size - 1) // page_size  # Ceiling division

    return {
        "data": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_count": total_count,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }