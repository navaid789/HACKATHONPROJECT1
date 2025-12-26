from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Dict, Any
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_error_handlers(app: FastAPI):
    """
    Initialize error handlers for the FastAPI application
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """
        Handle HTTP exceptions
        """
        logger.error(f"HTTP {exc.status_code} error at {request.url}: {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Handle request validation errors
        """
        logger.error(f"Validation error at {request.url}: {exc.errors()}")

        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "status_code": 422,
                "message": "Validation error",
                "details": exc.errors(),
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url)
            }
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """
        Handle general exceptions
        """
        logger.error(f"General error at {request.url}: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        # In production, don't expose internal error details to clients
        error_message = "Internal server error" if not app.debug else str(exc)
        error_details = traceback.format_exc() if app.debug else None

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status_code": 500,
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url),
                "details": error_details
            }
        )

    # Add a middleware to log all requests
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """
        Middleware to log incoming requests
        """
        logger.info(f"Request: {request.method} {request.url}")

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request failed: {request.method} {request.url} - {str(e)}")
            raise

        logger.info(f"Response: {response.status_code}")
        return response

def format_error_response(status_code: int, message: str, details: Any = None) -> Dict[str, Any]:
    """
    Format a standardized error response
    """
    response = {
        "success": False,
        "status_code": status_code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }

    if details:
        response["details"] = details

    return response

def format_success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """
    Format a standardized success response
    """
    return {
        "success": True,
        "status_code": 200,
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }