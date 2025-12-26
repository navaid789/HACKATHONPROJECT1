from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Content(Base):
    __tablename__ = "content"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    slug = Column(String(200), unique=True, index=True, nullable=False)
    content_type = Column(String(50), nullable=False)  # module, week, lesson, exercise, simulation
    module = Column(String(100), nullable=False)  # Module identifier (e.g., "ROS 2 Fundamentals")
    week = Column(Integer, nullable=True)  # Week number within module
    content = Column(Text, nullable=True)  # Main content in markdown or HTML
    metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    author_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    prerequisites = Column(Text, nullable=True)  # JSON string for prerequisite content IDs
    learning_objectives = Column(Text, nullable=True)  # JSON string for learning objectives
    duration_minutes = Column(Integer, default=60)  # Estimated completion time
    is_published = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Content(id={self.id}, title={self.title}, module={self.module})>"