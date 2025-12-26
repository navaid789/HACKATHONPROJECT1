from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..database.database import get_db
from ..models.content import Content
from ..models.user import User
from ..services.auth_service import get_current_active_user
from .response_utils import (
    create_success_response,
    create_error_response,
    format_content_response,
    handle_error,
    validate_and_format_input,
    paginate_response
)

router = APIRouter(prefix="/api/content", tags=["content"])

@router.post("/")
async def create_content(
    content_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create new content (lesson, exercise, etc.)
    """
    try:
        required_fields = ["title", "content_type", "module"]
        validate_and_format_input(content_data, required_fields)

        # Check if user has permission to create content (instructors and admins only)
        if current_user.role not in ["instructor", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only instructors and admins can create content"
            )

        # Check if slug already exists
        slug = content_data.get("slug")
        if not slug:
            # Generate slug from title if not provided
            title = content_data["title"]
            slug = title.lower().replace(" ", "-").replace("_", "-")

        existing_content = db.query(Content).filter(Content.slug == slug).first()
        if existing_content:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Content with this slug already exists"
            )

        # Create content object
        content = Content(
            title=content_data["title"],
            slug=slug,
            content_type=content_data["content_type"],
            module=content_data["module"],
            week=content_data.get("week"),
            content=content_data.get("content", ""),
            metadata=content_data.get("metadata"),
            author_id=current_user.id,
            prerequisites=content_data.get("prerequisites"),
            learning_objectives=content_data.get("learning_objectives"),
            duration_minutes=content_data.get("duration_minutes", 60),
            is_published=content_data.get("is_published", False)
        )

        db.add(content)
        db.commit()
        db.refresh(content)

        return create_success_response(
            data=format_content_response(content),
            message="Content created successfully",
            status_code=status.HTTP_201_CREATED
        )

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "Failed to create content", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/{content_id}")
async def get_content(
    content_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get content by ID
    """
    try:
        content = db.query(Content).filter(
            and_(
                Content.id == content_id,
                or_(Content.is_published == True, Content.author_id == current_user.id)
            )
        ).first()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )

        return create_success_response(
            data=format_content_response(content),
            message="Content retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "Failed to retrieve content", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/")
async def get_content_list(
    module: str = None,
    content_type: str = None,
    week: int = None,
    is_published: bool = True,
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of content with optional filters
    """
    try:
        query = db.query(Content)

        # Apply filters
        if module:
            query = query.filter(Content.module == module)
        if content_type:
            query = query.filter(Content.content_type == content_type)
        if week is not None:
            query = query.filter(Content.week == week)

        # For students, only show published content; for authors, show their own content too
        if current_user.role == "student":
            query = query.filter(Content.is_published == is_published)
        else:
            query = query.filter(
                or_(
                    Content.is_published == is_published,
                    Content.author_id == current_user.id
                )
            )

        # Calculate pagination
        total_count = query.count()
        offset = (page - 1) * page_size
        content_list = query.offset(offset).limit(page_size).all()

        formatted_content = [format_content_response(content) for content in content_list]

        return create_success_response(
            data=paginate_response(
                items=formatted_content,
                page=page,
                page_size=page_size,
                total_count=total_count
            ),
            message="Content list retrieved successfully"
        )

    except Exception as e:
        return handle_error(e, "Failed to retrieve content list", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.put("/{content_id}")
async def update_content(
    content_id: int,
    content_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Update content by ID
    """
    try:
        # Get existing content
        content = db.query(Content).filter(Content.id == content_id).first()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )

        # Check if user has permission to update (must be author or admin)
        if current_user.role != "admin" and content.author_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this content"
            )

        # Update fields if provided in content_data
        updatable_fields = [
            "title", "content_type", "module", "week", "content", "metadata",
            "prerequisites", "learning_objectives", "duration_minutes", "is_published"
        ]

        for field in updatable_fields:
            if field in content_data:
                setattr(content, field, content_data[field])

        # If title changed and slug wasn't explicitly provided, update slug
        if "title" in content_data and "slug" not in content_data:
            content.slug = content_data["title"].lower().replace(" ", "-").replace("_", "-")

        db.commit()
        db.refresh(content)

        return create_success_response(
            data=format_content_response(content),
            message="Content updated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "Failed to update content", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.delete("/{content_id}")
async def delete_content(
    content_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete content by ID
    """
    try:
        content = db.query(Content).filter(Content.id == content_id).first()

        if not content:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Content not found"
            )

        # Check if user has permission to delete (must be author or admin)
        if current_user.role != "admin" and content.author_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this content"
            )

        db.delete(content)
        db.commit()

        return create_success_response(
            message="Content deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        return handle_error(e, "Failed to delete content", status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.get("/module/{module_name}")
async def get_content_by_module(
    module_name: str,
    week: int = None,
    content_type: str = None,
    is_published: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get content by module with optional week and content type filters
    """
    try:
        query = db.query(Content).filter(Content.module == module_name)

        # Apply additional filters
        if week is not None:
            query = query.filter(Content.week == week)
        if content_type:
            query = query.filter(Content.content_type == content_type)

        # Apply publish status filter based on user role
        if current_user.role == "student":
            query = query.filter(Content.is_published == is_published)
        else:
            query = query.filter(
                or_(
                    Content.is_published == is_published,
                    Content.author_id == current_user.id
                )
            )

        content_list = query.all()
        formatted_content = [format_content_response(content) for content in content_list]

        return create_success_response(
            data=formatted_content,
            message=f"Content for module {module_name} retrieved successfully"
        )

    except Exception as e:
        return handle_error(e, "Failed to retrieve module content", status.HTTP_500_INTERNAL_SERVER_ERROR)