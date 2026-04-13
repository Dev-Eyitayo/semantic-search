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


# @router.put("/me", response_model=StandardResponse[dict])
# async def update_profile(
#     user_update: UserUpdate,
#     current_user: User = Depends(get_current_user),
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Update the authenticated user's profile information (name, phone, avatar).
#     """
#     logger.info(f"Profile update initiated for user: {current_user.email}")
    
#     # Update only provided fields
#     if user_update.first_name:
#         current_user.first_name = user_update.first_name
#     if user_update.last_name:
#         current_user.last_name = user_update.last_name
#     if user_update.phone:
#         current_user.phone = user_update.phone
#     if user_update.avatar_url:
#         current_user.avatar_url = user_update.avatar_url
    
#     current_user.updated_at = datetime.now(timezone.utc)
#     await db.commit()
#     await db.refresh(current_user)
    
#     logger.success(f"Profile updated successfully for user: {current_user.email}")
    
#     return StandardResponse(
#         message="Profile updated successfully",
#         data={"user": UserResponse.model_validate(current_user)}
#     )


@router.put("/me", response_model=StandardResponse[dict], status_code=200)
async def update_profile(
    first_name: Optional[str] = Form(None, min_length=2, max_length=100),
    last_name: Optional[str] = Form(None, min_length=2, max_length=100),
    phone: Optional[str] = Form(None, description="Nigerian phone number (08012345678, 2348012345678, or +2348012345678)"),
    avatar: Optional[UploadFile] = File(None, description="Avatar image file (JPEG|PNG|WebP, max 2MB)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Can update:
    - Profile fields: first_name, last_name, phone
    - Avatar media: avatar image file
    All fields are optional - only provided fields will be updated.
    """
    logger.info(f"Profile update initiated for user: {current_user.email}")
    
    updates_made = []
    avatar_url = None
    file_content = None
    
    
    # Validate phone if provided (but don't update yet)
    normalized_phone = None
    if phone:
        try:
            normalized_phone = normalize_nigerian_phone(phone)
            logger.debug(f"Phone number validated for user: {current_user.email}")
        except ValueError as e:
            logger.warning(f"Invalid phone format for user: {current_user.email} - {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    # Validate avatar file if provided (but don't upload yet)
    MAX_AVATAR_SIZE_MB = 2
    if avatar:
        logger.info(f"Avatar file validation initiated for user: {current_user.email}")
        
        try:
            # Read file content
            file_content = await avatar.read()
            
            # Validate file (raises HTTPException if invalid)
            is_valid, error_msg = CloudinaryService.validate_file(
                file_content,
                avatar.filename,
                "avatar",
                MAX_AVATAR_SIZE_MB
            )
            
            if not is_valid:
                logger.warning(f"Invalid avatar file for user: {current_user.email} - {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            logger.debug(f"Avatar file validated successfully for user: {current_user.email}")
            
        except HTTPException:
            # Re-raise validation errors immediately (no file upload happens)
            raise
        except Exception as e:
            logger.error(f"Error validating avatar file for user: {current_user.email} - {e}")
            raise HTTPException(status_code=422, detail="Invalid avatar file")
    

    if file_content is not None:
        logger.info(f"Avatar upload to Cloudinary started for user: {current_user.email}")
        
        try:
            # Upload to Cloudinary (only if validation passed above)
            upload_result = CloudinaryService.upload_avatar(
                file_content,
                avatar.filename,
                str(current_user.id)
            )
            
            avatar_url = upload_result['secure_url']
            logger.success(f"Avatar uploaded to Cloudinary for user: {current_user.email}")
            
        except Exception as e:
            logger.error(f"Cloudinary upload failed for user: {current_user.email} - {e}")
            raise HTTPException(status_code=503, detail="Failed to upload avatar to storage service")
    
    
    # Update profile fields in memory (no DB commit yet)
    if first_name:
        current_user.first_name = first_name
        updates_made.append("first_name")
        logger.debug(f"Updated first_name in memory for user: {current_user.email}")
    
    if last_name:
        current_user.last_name = last_name
        updates_made.append("last_name")
        logger.debug(f"Updated last_name in memory for user: {current_user.email}")
    
    if normalized_phone is not None:
        current_user.phone = normalized_phone
        updates_made.append("phone")
        logger.debug(f"Updated phone in memory for user: {current_user.email}")
    
    # Update avatar URL if upload succeeded
    if avatar_url:
        current_user.avatar_url = avatar_url
        updates_made.append("avatar")
        logger.debug(f"Updated avatar_url in memory for user: {current_user.email}")
    

    if updates_made:
        try:
            current_user.updated_at = datetime.now(timezone.utc)
            await db.commit()
            await db.refresh(current_user)
            logger.success(f"Profile updated in database for user: {current_user.email} - Updates: {', '.join(updates_made)}")
        except Exception as e:
            logger.error(f"Database commit failed for user: {current_user.email} - {e}")
            raise HTTPException(status_code=500, detail="Failed to save profile changes")
    else:
        logger.info(f"No updates provided for user: {current_user.email}")
    
    return StandardResponse(
        message="Profile updated successfully" if updates_made else "No updates provided",
        data={
            "user": UserResponse.model_validate(current_user),
            "updates_applied": updates_made,
            "avatar_url": avatar_url
        },
        meta={"fields_updated": len(updates_made)}
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

