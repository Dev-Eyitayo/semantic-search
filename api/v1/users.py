from fastapi import APIRouter, Depends, HTTPException, Query, Header
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


@router.put("/me", response_model=StandardResponse[dict])
async def update_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update the authenticated user's profile information (name, phone, avatar).
    """
    logger.info(f"Profile update initiated for user: {current_user.email}")
    
    # Update only provided fields
    if user_update.first_name:
        current_user.first_name = user_update.first_name
    if user_update.last_name:
        current_user.last_name = user_update.last_name
    if user_update.phone:
        current_user.phone = user_update.phone
    if user_update.avatar_url:
        current_user.avatar_url = user_update.avatar_url
    
    current_user.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(current_user)
    
    logger.success(f"Profile updated successfully for user: {current_user.email}")
    
    return StandardResponse(
        message="Profile updated successfully",
        data={"user": UserResponse.model_validate(current_user)}
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


# @router.get("/debug/token-test", response_model=StandardResponse[dict], tags=["Debug"])
# async def debug_token_test(
#     current_user: User = Depends(get_current_user),
# ):
#     """
#     Debug endpoint to verify token authentication is working.
#     This endpoint requires valid authentication to access.
#     """
#     return StandardResponse(
#         message="Token authentication working correctly!",
#         data={
#             "user_id": str(current_user.id),
#             "email": current_user.email,
#             "is_verified": current_user.is_verified,
#             "role": current_user.role,
#             "first_name": current_user.first_name,
#             "last_name": current_user.last_name,
#             "phone": current_user.phone,
#             "avatar_url": current_user.avatar_url,
#         }
#     )


# @router.get("/debug/headers", response_model=StandardResponse[dict], tags=["Debug"])
# async def debug_headers(
#     authorization: Optional[str] = Header(None),
# ):
#     """
#     Debug endpoint to check what Authorization headers Swagger is sending.
#     No authentication required.
#     """
#     return StandardResponse(
#         message="Headers received",
#         data={
#             "authorization_header": authorization if authorization else "NOT RECEIVED",
#             "header_present": authorization is not None,
#         }
#     )
