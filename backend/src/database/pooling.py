from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from typing import Optional
import os

class DatabasePool:
    """
    Database connection pooling configuration
    """
    def __init__(self):
        # Get database URL from environment or use default
        self.database_url = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/physical_ai_textbook")

        # Create engine with connection pooling settings
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=20,  # Number of connection objects to maintain in the pool
            max_overflow=30,  # Number of additional connections beyond pool_size
            pool_pre_ping=True,  # Verify connections before using them
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_timeout=30,  # Number of seconds to wait before giving up on getting a connection
            echo=False  # Set to True to log SQL statements (useful for debugging)
        )

    def get_engine(self):
        """
        Get the configured engine with connection pooling
        """
        return self.engine

# Create a global instance
db_pool = DatabasePool()

def get_db_engine():
    """
    Get the database engine with connection pooling
    """
    return db_pool.get_engine()