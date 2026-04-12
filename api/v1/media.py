from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import uuid

from api.deps import get_current_user
from db.session import get_db
from db.models.user import User
from db.models.property import Property
from schemas.media import (
    UploadListingImageResponse, UploadAvatarResponse,
    DeleteImageRequest, DeleteImageResponse
)
from schemas.base import StandardResponse
from services.cloudinary_service import CloudinaryService
from loguru import logger

router = APIRouter()

# File size constants
MAX_LISTING_IMAGE_SIZE_MB = 5
MAX_AVATAR_SIZE_MB = 2


@router.post("/upload-listing-image", response_model=StandardResponse[UploadListingImageResponse], status_code=201)
async def upload_listing_image(
    file: UploadFile = File(..., description="Image file (JPEG|PNG|WebP, max 5MB)"),
    listing_id: str = Form(None, description="Optional listing UUID for folder organization"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a single property listing image to Cloudinary.
    Returns the secure URL to include in property creation/update request.
    Only listers and admins can upload listing images.
    """
    logger.info(f"Listing image upload initiated - User: {current_user.email}, File: {file.filename}")
    
    # Check authorization
    if current_user.role not in ["lister", "admin"]:
        logger.warning(f"Unauthorized upload attempt - User: {current_user.email}, Role: {current_user.role}")
        raise HTTPException(status_code=403, detail="Only listers and admins can upload listing images")
    
    # Verify listing ownership if listing_id provided
    if listing_id:
        try:
            listing_uuid = uuid.UUID(listing_id)
        except ValueError:
            logger.warning(f"Invalid listing ID format: {listing_id}")
            raise HTTPException(status_code=400, detail="Invalid listing ID format")
        
        # Check if user owns the listing (if not admin)
        if current_user.role != "admin":
            result = await db.execute(
                select(Property).where(Property.id == listing_uuid)
            )
            listing = result.scalars().first()
            
            if not listing:
                logger.warning(f"Listing not found - ID: {listing_id}")
                raise HTTPException(status_code=404, detail="Listing not found")
            
            if listing.lister_id != current_user.id:
                logger.warning(f"Unauthorized listing image upload - User: {current_user.email}, Listing: {listing_id}")
                raise HTTPException(status_code=403, detail="Not authorized to upload images for this listing")
    else:
        # Use user ID as folder if listing_id not provided
        listing_id = str(current_user.id)
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file
        is_valid, error_msg = CloudinaryService.validate_file(
            file_content,
            file.filename,
            "listing",
            MAX_LISTING_IMAGE_SIZE_MB
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Upload to Cloudinary
        upload_result = CloudinaryService.upload_listing_image(
            file_content,
            file.filename,
            listing_id
        )
        
        logger.success(f"Listing image uploaded successfully - Public ID: {upload_result['public_id']}")
        
        return StandardResponse(
            message="Image uploaded successfully",
            data=UploadListingImageResponse(**upload_result),
            meta={"file_size_mb": round(len(file_content) / (1024 * 1024), 2), "folder": f"listings/{listing_id}", "image_number": 1}
        )
        
    except Exception as e:
        logger.error(f"Failed to upload listing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Cloudinary upload failed")


@router.post("/upload-avatar", response_model=StandardResponse[UploadAvatarResponse], status_code=201)
async def upload_avatar(
    file: UploadFile = File(..., description="Avatar image file (JPEG|PNG|WebP, max 2MB)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a user profile avatar image to Cloudinary with face detection.
    Returns the secure URL to use in PUT /api/v1/users/me.
    Automatically deletes old avatar and applies circular crop transformation.
    """
    logger.info(f"Avatar upload initiated - User: {current_user.email}, File: {file.filename}")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file
        is_valid, error_msg = CloudinaryService.validate_file(
            file_content,
            file.filename,
            "avatar",
            MAX_AVATAR_SIZE_MB
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Upload to Cloudinary
        upload_result = CloudinaryService.upload_avatar(
            file_content,
            str(current_user.id)
        )
        
        logger.success(f"Avatar uploaded successfully for user: {current_user.email}")
        
        return StandardResponse(
            message="Avatar uploaded successfully",
            data=UploadAvatarResponse(**upload_result),
            meta={"file_size_mb": round(len(file_content) / (1024 * 1024), 2), "folder": f"avatars/{current_user.id}", "profile": current_user.email}
        )
        
    except Exception as e:
        logger.error(f"Failed to upload avatar for user {current_user.email}: {str(e)}")
        raise HTTPException(status_code=500, detail="Cloudinary upload failed")


@router.delete("/listing-image", response_model=StandardResponse[DeleteImageResponse])
async def delete_listing_image(
    request: DeleteImageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Remove a specific listing image from Cloudinary by its public_id.
    Only the listing owner (lister) or admin can delete images.
    Also removes the image URL from the listing's images array.
    """
    logger.info(f"Listing image deletion initiated - User: {current_user.email}, Public ID: {request.public_id}")
    
    try:
        # Extract listing ID from public_id (format: listings/{listing_id}/...)
        parts = request.public_id.split("/")
        if len(parts) < 2 or parts[0] != "listings":
            logger.warning(f"Invalid public_id format: {request.public_id}")
            raise HTTPException(status_code=400, detail="Invalid image public_id format")
        
        listing_id_str = parts[1]
        
        try:
            listing_id = uuid.UUID(listing_id_str)
        except ValueError:
            logger.warning(f"Invalid listing ID in public_id: {listing_id_str}")
            raise HTTPException(status_code=400, detail="Invalid listing ID format")
        
        # Verify authorization
        result = await db.execute(
            select(Property).where(Property.id == listing_id)
        )
        listing = result.scalars().first()
        
        if not listing:
            logger.warning(f"Listing not found - ID: {listing_id}")
            raise HTTPException(status_code=404, detail="Listing not found")
        
        # Check authorization (owner or admin)
        if listing.lister_id != current_user.id and current_user.role != "admin":
            logger.warning(f"Unauthorized image deletion attempt - User: {current_user.email}, Listing: {listing_id}")
            raise HTTPException(status_code=403, detail="Not authorized to delete this image")
        
        # Delete from Cloudinary
        success = CloudinaryService.delete_image(request.public_id)
        
        if not success:
            logger.warning(f"Failed to delete image from Cloudinary: {request.public_id}")
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Remove URL from listing.images array if it exists
        if listing.images:
            # Try to find and remove the URL that matches this public_id
            images_to_remove = []
            for img_url in listing.images:
                if request.public_id in img_url:
                    images_to_remove.append(img_url)
            
            for img_url in images_to_remove:
                listing.images.remove(img_url)
            
            if images_to_remove:
                await db.commit()
                logger.info(f"Removed {len(images_to_remove)} image URL(s) from listing {listing_id}")
        
        logger.success(f"Image deleted successfully - Public ID: {request.public_id}, User: {current_user.email}")
        
        return StandardResponse(
            message="Image deleted from Cloudinary",
            data=DeleteImageResponse(public_id=request.public_id),
            meta={"folder": f"listings/{listing_id_str}", "images_remaining": len(listing.images) if listing.images else 0}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete image {request.public_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete image")
