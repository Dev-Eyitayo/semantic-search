from fastapi import APIRouter, Depends, HTTPException, Query, Header, Form, File, UploadFile, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, desc
from datetime import datetime, timezone
import uuid

from api.deps import get_current_user
from db.session import get_db
from db.models.user import User, RefreshToken
from db.models.property import SavedSearch
from schemas.user import (
    UserResponse, UserUpdate, ChangePasswordRequest,
    SavedSearchCreate, SavedSearchResponse, SavedSearchList
)
from schemas.base import StandardResponse
from core.security import verify_password, get_password_hash
from core.phone_utils import normalize_nigerian_phone
from services.cloudinary_service import CloudinaryService
from loguru import logger
from typing import Optional

router = APIRouter()


@router.get("/me", response_model=StandardResponse[UserResponse])
async def get_current_user_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve the authenticated user's profile details including role and verification status.
    """
    logger.info(f"Profile retrieved for user: {current_user.email}")
    
    return StandardResponse(
        message="User profile retrieved successfully",
        data=UserResponse.model_validate(current_user)
    )



@router.put("/me", response_model=StandardResponse[dict], status_code=200)
async def update_profile(
    user_data: UserUpdate = Depends(UserUpdate.as_form),
    avatar: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):

    logger.info(f"Profile update initiated for: {current_user.email}")
    
    updates_made = []
    avatar_url = None

    if avatar:
        try:
            file_content = await avatar.read()
            # Validate size/type via Cloudinary service
            is_valid, error_msg = CloudinaryService.validate_file(
                file_content, avatar.filename, "avatar", 2
            )
            if not is_valid:
                raise HTTPException(status_code=400, detail=error_msg)

            # Upload
            upload_result = CloudinaryService.upload_avatar(
                file_content, avatar.filename, str(current_user.id)
            )
            avatar_url = upload_result['secure_url']
            current_user.avatar_url = avatar_url
            updates_made.append("avatar")
        except Exception as e:
            logger.error(f"Avatar processing failed: {e}")
            raise HTTPException(status_code=503, detail="Storage service unavailable")

    update_dict = user_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        setattr(current_user, field, value)
        updates_made.append(field)

    if updates_made:
        try:
            current_user.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(current_user)
            logger.success(f"Profile updated: {', '.join(updates_made)}")
        except Exception as e:
            logger.error(f"DB Update failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to save changes")

    return StandardResponse(
        message="Profile updated" if updates_made else "No changes detected",
        data={
            "user": UserResponse.model_validate(current_user),
            "updates_applied": updates_made
        }
    )

@router.post("/me/change-password", response_model=StandardResponse[None])
async def change_password(
    password_change: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Change the authenticated user's password by providing the current and new password.
    Invalidates all existing refresh tokens after change for security.
    """
    logger.info(f"Password change initiated for user: {current_user.email}")
    
    # Verify current password
    if not verify_password(password_change.current_password, current_user.password_hash):
        logger.warning(f"Failed password change attempt - incorrect password for user: {current_user.email}")
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    
    # Update password
    current_user.password_hash = get_password_hash(password_change.new_password)
    
    # Revoke all existing refresh tokens
    await db.execute(
        select(RefreshToken).where(
            RefreshToken.user_id == current_user.id
        ).update({"is_revoked": True})
    )
    
    await db.commit()
    
    logger.success(f"Password changed successfully for user: {current_user.email}")
    
    return StandardResponse(message="Password changed successfully")


@router.get("/me/saved-searches", response_model=StandardResponse[SavedSearchList])
async def get_saved_searches(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve the list of queries the user has saved for future reference or alerts.
    """
    logger.info(f"Fetching saved searches for user: {current_user.email} (page: {page}, limit: {limit})")
    
    offset = (page - 1) * limit
    
    # Get total count
    count_result = await db.execute(
        select(func.count(SavedSearch.id)).where(SavedSearch.user_id == current_user.id)
    )
    total = count_result.scalar()
    
    # Get paginated results
    result = await db.execute(
        select(SavedSearch)
        .where(SavedSearch.user_id == current_user.id)
        .order_by(desc(SavedSearch.created_at))
        .offset(offset)
        .limit(limit)
    )
    searches = result.scalars().all()
    
    logger.success(f"Retrieved {len(searches)} saved searches for user: {current_user.email}")
    
    return StandardResponse(
        message="Saved searches retrieved successfully",
        data=SavedSearchList(
            saved_searches=[SavedSearchResponse.model_validate(s) for s in searches],
            total=total,
            page=page,
            limit=limit
        )
    )


@router.post("/me/saved-searches", response_model=StandardResponse[SavedSearchResponse], status_code=201)
async def save_search(
    search_data: SavedSearchCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Save a search query and its filters for future access or alert notifications.
    Limit: 20 saved searches per user.
    """
    logger.info(f"Attempting to save search for user: {current_user.email} - Query: {search_data.query}")
    
    # Check limit
    count_result = await db.execute(
        select(func.count(SavedSearch.id)).where(SavedSearch.user_id == current_user.id)
    )
    count = count_result.scalar()
    
    if count >= 20:
        logger.warning(f"Maximum saved searches limit reached for user: {current_user.email}")
        raise HTTPException(status_code=400, detail="Maximum saved searches limit (20) reached")
    
    # Create new saved search
    saved_search = SavedSearch(
        user_id=current_user.id,
        query=search_data.query,
        filters=search_data.filters,
        notify_on_match=search_data.notify_on_match
    )
    
    db.add(saved_search)
    await db.commit()
    await db.refresh(saved_search)
    
    logger.success(f"Search saved successfully for user: {current_user.email} - ID: {saved_search.id}")
    
    return StandardResponse(
        message="Search saved successfully",
        data=SavedSearchResponse.model_validate(saved_search),
        meta={"saved_search_id": str(saved_search.id)}
    )


@router.delete("/me/saved-searches/{search_id}", response_model=StandardResponse[None])
async def delete_saved_search(
    search_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Remove a saved search by its ID.
    Verify ownership: saved_search.user_id must match current user.
    """
    logger.info(f"Attempting to delete saved search for user: {current_user.email} - Search ID: {search_id}")
    
    try:
        search_uuid = uuid.UUID(search_id)
    except ValueError:
        logger.warning(f"Invalid search ID format provided: {search_id}")
        raise HTTPException(status_code=400, detail="Invalid search ID format")
    
    # Verify ownership
    result = await db.execute(
        select(SavedSearch).where(
            and_(
                SavedSearch.id == search_uuid,
                SavedSearch.user_id == current_user.id
            )
        )
    )
    saved_search = result.scalars().first()
    
    if not saved_search:
        logger.warning(f"Saved search not found or not owned by user: {current_user.email} - Search ID: {search_id}")
        raise HTTPException(status_code=404, detail="Search not found or not owned by user")
    
    await db.delete(saved_search)
    await db.commit()
    
    logger.success(f"Saved search deleted for user: {current_user.email} - Search ID: {search_id}")
    
    return StandardResponse(message="Saved search deleted successfully")

