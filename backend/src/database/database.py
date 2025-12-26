from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import os
from typing import Generator

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/physical_ai_textbook")

# Create engine
engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Number of connection objects to maintain in the pool
    max_overflow=30,  # Number of additional connections beyond pool_size
    pool_pre_ping=True,  # Verify connections before using them
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for models
Base = declarative_base()

@contextmanager
def get_db_session() -> Generator:
    """
    Context manager for database sessions
    Ensures proper cleanup of database connections
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db():
    """
    Dependency for FastAPI to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Initialize the database by creating all tables
    """
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()