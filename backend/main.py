from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from src.database.database import init_db
from src.api.content_routes import router as content_router
from src.api.exercise_routes import router as exercise_router
from src.api.response_utils import create_success_response
from src.api.error_handlers import init_error_handlers
from src.config.settings import settings
from src.services.logging_service import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown
    """
    # Startup
    logger.log_info("Starting Physical AI & Humanoid Robotics Textbook API")
    init_db()
    logger.log_info("Database initialized successfully")

    yield

    # Shutdown
    logger.log_info("Shutting down Physical AI & Humanoid Robotics Textbook API")

# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook API",
    description="An interactive learning platform for Physical AI and Humanoid Robotics",
    version="1.0.0",
    lifespan=lifespan
)

# Initialize error handlers
init_error_handlers(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Add this for auth token access
    expose_headers=["Access-Control-Allow-Origin"]
)

# Include API routers
app.include_router(content_router)
app.include_router(exercise_router)

@app.get("/")
async def root():
    """
    Root endpoint for health check
    """
    return create_success_response(
        data={"message": "Physical AI & Humanoid Robotics Textbook API is running"},
        message="API is healthy"
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return create_success_response(
        data={
            "status": "healthy",
            "version": "1.0.0",
            "service": "Physical AI & Humanoid Robotics Textbook API"
        },
        message="Health check passed"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )