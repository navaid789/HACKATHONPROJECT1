from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class ExerciseSubmission(Base):
    __tablename__ = "exercise_submissions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exercise_id = Column(Integer, ForeignKey("content.id"), nullable=False)
    submission_content = Column(Text, nullable=False)  # student's code or answer
    submission_metadata = Column(JSON, nullable=True)  # additional submission data
    grade = Column(Float, nullable=True)  # numeric grade (0-100)
    feedback = Column(Text, nullable=True)  # instructor feedback
    is_graded = Column(Boolean, default=False)
    submitted_at = Column(DateTime, default=datetime.utcnow)
    graded_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<ExerciseSubmission(id={self.id}, user_id={self.user_id}, exercise_id={self.exercise_id})>"