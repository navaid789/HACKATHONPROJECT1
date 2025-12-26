from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Physical AI & Humanoid Robotics Textbook API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Database settings
    database_url: str = os.getenv("DATABASE_URL", "postgresql://username:password@localhost/physical_ai_textbook")

    # Qdrant settings for vector database
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_https: bool = os.getenv("QDRANT_HTTPS", "False").lower() == "true"

    # Authentication settings
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    algorithm: str = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    refresh_token_expire_days: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # AI/ML settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # Content settings
    max_content_size: int = int(os.getenv("MAX_CONTENT_SIZE", "1048576"))  # 1MB in bytes
    allowed_content_types: str = os.getenv("ALLOWED_CONTENT_TYPES", "text/markdown,text/html,application/json")

    # Simulation settings
    simulation_timeout: int = int(os.getenv("SIMULATION_TIMEOUT", "300"))  # 5 minutes in seconds

    # CORS settings
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    class Config:
        env_file = ".env"

# Create settings instance
settings = Settings()