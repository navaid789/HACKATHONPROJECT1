from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, Optional
from datetime import datetime
import json
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..database.database import get_db
from ..models.exercise_submission import ExerciseSubmission
from ..models.content import Content
from ..models.user import User
from .response_utils import (
    create_success_response,
    create_error_response,
    format_exercise_submission_response,
    handle_error,
    validate_and_format_input
)
from ..services.rag_service import RAGService
from ..config.settings import settings

router = APIRouter(prefix="/api/exercises", tags=["exercises"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@router.post("/submit")
async def submit_exercise(
    submission_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Submit an exercise solution for validation and grading
    """
    try:
        # Validate required fields
        required_fields = ["user_id", "exercise_id", "submission_content"]
        validate_and_format_input(submission_data, required_fields)

        user_id = submission_data["user_id"]
        exercise_id = submission_data["exercise_id"]
        submission_content = submission_data["submission_content"]
        language = submission_data.get("language", "python")
        metadata = submission_data.get("metadata", {})

        # Verify user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Verify exercise exists
        exercise = db.query(Content).filter(Content.id == exercise_id).first()
        if not exercise:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Exercise not found"
            )

        # Create exercise submission record
        exercise_submission = ExerciseSubmission(
            user_id=user_id,
            exercise_id=exercise_id,
            submission_content=submission_content,
            submission_metadata=metadata,
            is_graded=False
        )

        db.add(exercise_submission)
        db.commit()
        db.refresh(exercise_submission)

        return create_success_response(
            data=format_exercise_submission_response(exercise_submission),
            message="Exercise submitted successfully",
            status_code=status.HTTP_201_CREATED
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting exercise: {str(e)}")
        return handle_error(e, "Failed to submit exercise", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/submission/{submission_id}")
async def get_submission(
    submission_id: int,
    db: Session = Depends(get_db)
):
    """
    Get details of a specific exercise submission
    """
    try:
        submission = db.query(ExerciseSubmission).filter(
            ExerciseSubmission.id == submission_id
        ).first()

        if not submission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Submission not found"
            )

        return create_success_response(
            data=format_exercise_submission_response(submission),
            message="Submission retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting submission: {str(e)}")
        return handle_error(e, "Failed to retrieve submission", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/user/{user_id}/exercise/{exercise_id}/submissions")
async def get_user_exercise_submissions(
    user_id: int,
    exercise_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all submissions for a specific user and exercise
    """
    try:
        submissions = db.query(ExerciseSubmission).filter(
            and_(
                ExerciseSubmission.user_id == user_id,
                ExerciseSubmission.exercise_id == exercise_id
            )
        ).order_by(ExerciseSubmission.submitted_at.desc()).all()

        formatted_submissions = [
            format_exercise_submission_response(sub) for sub in submissions
        ]

        return create_success_response(
            data=formatted_submissions,
            message="Submissions retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting user exercise submissions: {str(e)}")
        return handle_error(e, "Failed to retrieve submissions", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/user/{user_id}/progress")
async def get_user_exercise_progress(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get overall exercise progress for a user
    """
    try:
        # Get all submissions for the user
        all_submissions = db.query(ExerciseSubmission).filter(
            ExerciseSubmission.user_id == user_id
        ).all()

        # Calculate progress statistics
        total_submissions = len(all_submissions)
        graded_submissions = [sub for sub in all_submissions if sub.is_graded]
        passed_submissions = [sub for sub in graded_submissions if sub.grade and sub.grade >= 60]

        avg_grade = 0
        if graded_submissions:
            avg_grade = sum(sub.grade for sub in graded_submissions if sub.grade) / len(graded_submissions)

        progress_data = {
            "total_submissions": total_submissions,
            "graded_submissions": len(graded_submissions),
            "passed_submissions": len(passed_submissions),
            "failed_submissions": len(graded_submissions) - len(passed_submissions),
            "average_grade": round(avg_grade, 2) if avg_grade else 0,
            "completion_rate": round((len(graded_submissions) / total_submissions * 100) if total_submissions > 0 else 0, 2)
        }

        return create_success_response(
            data=progress_data,
            message="User progress retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Error getting user progress: {str(e)}")
        return handle_error(e, "Failed to retrieve user progress", status.HTTP_500_INTERNAL_SERVER_ERROR)