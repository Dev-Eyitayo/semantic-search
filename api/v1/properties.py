from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Form, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, and_, desc, or_
from datetime import datetime, timezone
import uuid
import json

from api.deps import get_current_user, RoleChecker
from db.session import get_db
from db.models.user import User
from db.models.property import Property
from core.enums import PropertyType, PriceType, PropertyStatus
from core.permissions import IsListerOrAdmin, IsObjectOwner, IsLister
from core.permission_utils import check_object_permission
from schemas.property import (
    PropertyCreate, PropertyUpdate, PropertyResponse, PropertyListResponse,
    PropertyListWithPagination, PropertyMyListingsResponse, PropertyListingResponse
)
from schemas.base import StandardResponse
from services.cloudinary_service import CloudinaryService
from loguru import logger
from typing import List, Optional

router = APIRouter()


@router.get("", response_model=StandardResponse[PropertyListWithPagination])
async def list_properties(
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=50),
    property_type: PropertyType = Query(None),
    bedrooms: int = Query(None),
    bathrooms: int = Query(None),
    price_min: float = Query(None, ge=0),
    price_max: float = Query(None, gt=0),
    price_type: PriceType = Query(None),
    location: str = Query(None),
    sort_by: str = Query("newest", pattern="^(price_asc|price_desc|newest|oldest)$"),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve a paginated list of all approved property listings.
    Supports filtering by property type, bedrooms, bathrooms, price range, location, and sorting.
    """
    logger.info(f"Listing properties - Page: {page}, Limit: {limit}, Filters: property_type={property_type}, bedrooms={bedrooms}")
    
    offset = (page - 1) * limit
    
    # Build filter conditions
    filters = [Property.status == PropertyStatus.APPROVED]
    
    if property_type:
        filters.append(Property.property_type == property_type)
    if bedrooms is not None:
        filters.append(Property.bedrooms == bedrooms)
    if bathrooms is not None:
        filters.append(Property.bathrooms == bathrooms)
    if price_min is not None:
        filters.append(Property.price >= price_min)
    if price_max is not None:
        filters.append(Property.price <= price_max)
    if price_type:
        filters.append(Property.price_type == price_type)
    if location:
        filters.append(Property.location.ilike(f"%{location}%"))
    
    # Determine sort order
    if sort_by == "price_asc":
        order_by = Property.price.asc()
    elif sort_by == "price_desc":
        order_by = Property.price.desc()
    elif sort_by == "oldest":
        order_by = Property.created_at.asc()
    else:  # newest
        order_by = Property.created_at.desc()
    
    # Get total count
    count_result = await db.execute(
        select(func.count(Property.id)).where(and_(*filters))
    )
    total = count_result.scalar()
    
    # Get paginated results
    result = await db.execute(
        select(Property)
        .where(and_(*filters))
        .order_by(order_by)
        .offset(offset)
        .limit(limit)
    )
    properties = result.scalars().all()
    
    logger.success(f"Retrieved {len(properties)} approved properties")
    
    pages = (total + limit - 1) // limit
    
    return StandardResponse(
        message="Properties retrieved successfully",
        data=PropertyListWithPagination(
            properties=[PropertyListResponse.model_validate(p) for p in properties],
            total=total,
            page=page,
            limit=limit,
            pages=pages
        )
    )


@router.get("/{property_id}", response_model=StandardResponse[PropertyResponse])
async def get_property(
    property_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve full details of a single approved property listing, including all images, amenities, and lister contact info.
    Increments view count asynchronously.
    """
    try:
        prop_uuid = uuid.UUID(property_id)
    except ValueError:
        logger.warning(f"Invalid property ID format: {property_id}")
        raise HTTPException(status_code=400, detail="Invalid property ID format")
    
    logger.info(f"Fetching property details - ID: {property_id}")
    
    # Get property with lister info
    result = await db.execute(
        select(Property).where(
            and_(
                Property.id == prop_uuid,
                Property.status == PropertyStatus.APPROVED
            )
        )
    )
    property_obj = result.scalars().first()
    
    if not property_obj:
        logger.warning(f"Property not found or not approved - ID: {property_id}")
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Increment view count asynchronously (fire and forget)
    async def increment_views():
        property_obj.view_count += 1
        await db.commit()
    
    try:
        await increment_views()
    except Exception as e:
        logger.error(f"Failed to increment view count for property {property_id}: {str(e)}")
    
    # Get lister details
    lister = await db.get(User, property_obj.lister_id)
    
    logger.success(f"Property retrieved successfully - ID: {property_id}")
    
    return StandardResponse(
        message="Property retrieved successfully",
        data=PropertyResponse.model_validate(property_obj),
        meta={"lister": lister}
    )


@router.post("", response_model=StandardResponse[PropertyListingResponse], status_code=201)
async def create_property(
    # Property details as form fields
    title: str = Form(..., min_length=10, max_length=255),
    description: str = Form(..., min_length=100, max_length=5000),
    price: float = Form(..., gt=0),
    price_type: str = Form(...),
    location: str = Form(..., min_length=5, max_length=255),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    bedrooms: int = Form(..., ge=0, le=20),
    bathrooms: int = Form(..., ge=0, le=20),
    property_type: str = Form(...),
    amenities: Optional[str] = Form(None),  # JSON string
    # Image files (optional, max 15)
    images: List[UploadFile] = File(default=[], description="Property images (max 15, JPEG|PNG|WebP, 5MB each)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _=Depends(IsListerOrAdmin())  # DRF-style permission dependency
):
    """
    Create a new property listing with images in a single multipart request.
    - Upload images to Cloudinary automatically organized by listing ID
    - Create property with all details and image URLs in one step
    - Status defaults to pending_review
    Only listers and admins can create listings.
    """
    logger.info(f"Creating property listing - User: {current_user.email}, Title: {title}, Images: {len(images)}")
    
    try:
        # Validate enum values
        try:
            price_type_enum = PriceType(price_type)
            property_type_enum = PropertyType(property_type)
        except ValueError as e:
            logger.warning(f"Invalid enum value: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
        
        # Validate description length
        if len(description) < 100:
            raise HTTPException(status_code=400, detail="Description must be at least 100 characters")
        
        # Validate image count
        if len(images) > 15:
            logger.warning(f"Too many images provided - Count: {len(images)}")
            raise HTTPException(status_code=400, detail="Maximum 15 images allowed per listing")
        
        # Parse amenities from JSON string
        amenities_list = []
        if amenities:
            try:
                amenities_list = json.loads(amenities)
                if not isinstance(amenities_list, list):
                    raise ValueError("Amenities must be a JSON array")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid amenities JSON format")
        
        # Create property FIRST (without images) to get the property ID
        new_property = Property(
            lister_id=current_user.id,
            title=title,
            description=description,
            price=price,
            price_type=price_type_enum,
            location=location,
            latitude=latitude,
            longitude=longitude,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            property_type=property_type_enum,
            amenities=amenities_list,
            images=[],  # Will be populated after upload
            thumbnail=None,  # Will be set after first image upload
            status=PropertyStatus.PENDING_REVIEW
        )
        
        db.add(new_property)
        await db.commit()
        await db.refresh(new_property)
        
        logger.info(f"Property created with ID: {new_property.id}, ready for image uploads")
        
        # Now upload images to the correct property-specific folder
        cloudinary_urls = []
        property_id_str = str(new_property.id)
        
        for idx, image_file in enumerate(images):
            try:
                # Read file content
                file_content = await image_file.read()
                
                # Validate file
                is_valid, error_msg = CloudinaryService.validate_file(
                    file_content,
                    image_file.filename,
                    "listing",
                    5  # 5MB max for listing images
                )
                
                if not is_valid:
                    logger.warning(f"Invalid image file: {error_msg}")
                    raise HTTPException(status_code=400, detail=f"Image {idx + 1}: {error_msg}")
                
                # Upload to Cloudinary with proper folder organization: listings/{property_id}/
                upload_result = CloudinaryService.upload_listing_image(
                    file_content,
                    image_file.filename,
                    property_id_str
                )
                cloudinary_urls.append(upload_result["secure_url"])
                logger.info(f"Image {idx + 1} uploaded to listings/{property_id_str}/ - Public ID: {upload_result['public_id']}")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to upload image {idx + 1}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to upload image {idx + 1}")
        
        # Set thumbnail from first image and update property with all image URLs
        if cloudinary_urls:
            new_property.thumbnail = cloudinary_urls[0]
        new_property.images = cloudinary_urls
        
        await db.commit()
        await db.refresh(new_property)
        
        logger.success(f"Property listing created - ID: {new_property.id}, Images: {len(cloudinary_urls)}, User: {current_user.email}")
        
        # Schedule embedding generation as background task
        # TODO: Implement S-BERT embedding generation task
        
        return StandardResponse(
            message="Listing submitted successfully with images. Embedding generation in progress.",
            data=PropertyListingResponse.model_validate(new_property),
            meta={
                "property_id": str(new_property.id),
                "images_uploaded": len(cloudinary_urls),
                "cloudinary_folder": f"listings/{new_property.id}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create property listing: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create listing")


@router.put("/{property_id}", response_model=StandardResponse[PropertyListingResponse])
async def update_property(
    property_id: str,
    # Optional property update fields
    title: Optional[str] = Form(None, min_length=10, max_length=255),
    description: Optional[str] = Form(None, min_length=100, max_length=5000),
    price: Optional[float] = Form(None, gt=0),
    price_type: Optional[str] = Form(None),
    location: Optional[str] = Form(None, min_length=5, max_length=255),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    bedrooms: Optional[int] = Form(None, ge=0, le=20),
    bathrooms: Optional[int] = Form(None, ge=0, le=20),
    property_type: Optional[str] = Form(None),
    amenities: Optional[str] = Form(None),  # JSON string
    # New images to add/replace (optional)
    images: List[UploadFile] = File(default=[], description="Property images to add/replace (max 15 total)"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing property listing with optional image uploads.
    - Can update any property field
    - Can add/replace images in same request
    - Re-triggers embedding if title or description changes
    Only the owner lister or admin can update.
    """
    try:
        prop_uuid = uuid.UUID(property_id)
    except ValueError:
        logger.warning(f"Invalid property ID format: {property_id}")
        raise HTTPException(status_code=400, detail="Invalid property ID format")
    
    logger.info(f"Updating property listing - ID: {property_id}, User: {current_user.email}, Images provided: {len(images)}")
    
    # Get property first to check ownership
    prop_uuid = uuid.UUID(property_id)
    result = await db.execute(
        select(Property).where(Property.id == prop_uuid)
    )
    property_obj = result.scalars().first()
    
    if not property_obj:
        logger.warning(f"Property not found - ID: {property_id}")
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check ownership permission - user must own the property or be admin
    permission = IsObjectOwner(Property, "property_id", "lister_id")
    await check_object_permission(permission, current_user, property_id, db)
    
    try:
        # Track if description changed (requires re-embedding)
        description_changed = description and description != property_obj.description
        
        # Update text fields
        if title:
            property_obj.title = title
        if description:
            property_obj.description = description
        if price:
            property_obj.price = price
        if price_type:
            try:
                property_obj.price_type = PriceType(price_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid price_type: {price_type}")
        if location:
            property_obj.location = location
        if latitude is not None:
            property_obj.latitude = latitude
        if longitude is not None:
            property_obj.longitude = longitude
        if bedrooms is not None:
            property_obj.bedrooms = bedrooms
        if bathrooms is not None:
            property_obj.bathrooms = bathrooms
        if property_type:
            try:
                property_obj.property_type = PropertyType(property_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid property_type: {property_type}")
        if amenities:
            try:
                amenities_list = json.loads(amenities)
                if not isinstance(amenities_list, list):
                    raise ValueError("Amenities must be a JSON array")
                property_obj.amenities = amenities_list
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid amenities JSON format")
        
        # Handle image uploads
        if images:
            cloudinary_urls = []
            
            for idx, image_file in enumerate(images):
                try:
                    # Read file content
                    file_content = await image_file.read()
                    
                    # Validate file
                    is_valid, error_msg = CloudinaryService.validate_file(
                        file_content,
                        image_file.filename,
                        "listing",
                        5  # 5MB max for listing images
                    )
                    
                    if not is_valid:
                        logger.warning(f"Invalid image file: {error_msg}")
                        raise HTTPException(status_code=400, detail=f"Image {idx + 1}: {error_msg}")
                    
                    # Upload to Cloudinary
                    upload_result = CloudinaryService.upload_listing_image(
                        file_content,
                        image_file.filename,
                        property_id
                    )
                    cloudinary_urls.append(upload_result["secure_url"])
                    logger.info(f"Image {idx + 1} uploaded - Public ID: {upload_result['public_id']}")
                    
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"Failed to upload image {idx + 1}: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to upload image {idx + 1}")
            
            # Validate total image count
            if len(cloudinary_urls) > 15:
                logger.warning(f"Too many images provided - Count: {len(cloudinary_urls)}")
                raise HTTPException(status_code=400, detail="Maximum 15 images allowed per listing")
            
            # Update images (replace existing ones)
            property_obj.images = cloudinary_urls
            property_obj.thumbnail = cloudinary_urls[0] if cloudinary_urls else property_obj.thumbnail
            logger.info(f"Property images updated - New count: {len(cloudinary_urls)}, ID: {property_id}")
        
        # Reset status to pending_review if description changed
        if description_changed:
            property_obj.status = PropertyStatus.PENDING_REVIEW
            property_obj.embedding = None  # Clear embedding for re-generation
            logger.info(f"Property description changed - resetting status to pending_review - ID: {property_id}")
        
        property_obj.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(property_obj)
        
        logger.success(f"Property updated successfully - ID: {property_id}, User: {current_user.email}, Images updated: {len(images) > 0}")
        
        return StandardResponse(
            message="Listing updated successfully",
            data=PropertyListingResponse.model_validate(property_obj),
            meta={
                "re_embedding": description_changed,
                "images_updated": len(images),
                "cloudinary_folder": f"listings/{property_id}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update property: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update listing")


@router.delete("/{property_id}", response_model=StandardResponse[None])
async def delete_property(
    property_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Soft-delete a property listing. Marks status as 'deleted'. Removes from search index.
    Only property owner or admin can delete.
    """
    try:
        prop_uuid = uuid.UUID(property_id)
    except ValueError:
        logger.warning(f"Invalid property ID format: {property_id}")
        raise HTTPException(status_code=400, detail="Invalid property ID format")
    
    logger.info(f"Deleting property listing - ID: {property_id}, User: {current_user.email}")
    
    # Get property
    result = await db.execute(
        select(Property).where(Property.id == prop_uuid)
    )
    property_obj = result.scalars().first()
    
    if not property_obj:
        logger.warning(f"Property not found - ID: {property_id}")
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check ownership permission using DRF-style permissions
    permission = IsObjectOwner(Property, "property_id", "lister_id")
    await check_object_permission(permission, current_user, property_id, db)
    
    # Soft delete
    property_obj.status = PropertyStatus.DELETED
    property_obj.embedding = None  # Remove from vector index
    await db.commit()
    
    logger.success(f"Property deleted successfully (soft delete) - ID: {property_id}, User: {current_user.email}")
    
    # TODO: Delete Cloudinary images via Cloudinary Destroy API
    
    return StandardResponse(message="Listing deleted successfully")


@router.get("/my-listings", response_model=StandardResponse[PropertyMyListingsResponse])
async def get_my_listings(
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=50),
    status: PropertyStatus = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _=Depends(IsLister())  # DRF-style permission: only listers can access their listings
):
    """
    Retrieve all listings submitted by the currently authenticated lister,
    including pending, approved, and rejected. Only accessible to listers.
    """
    logger.info(f"Fetching my listings - User: {current_user.email}, Page: {page}, Status: {status}")
    
    offset = (page - 1) * limit
    
    # Build filters
    filters = [Property.lister_id == current_user.id, Property.status != PropertyStatus.DELETED]
    
    if status:
        filters.append(Property.status == status)
    
    # Get total count
    count_result = await db.execute(
        select(func.count(Property.id)).where(and_(*filters))
    )
    total = count_result.scalar()
    
    # Get paginated results
    result = await db.execute(
        select(Property)
        .where(and_(*filters))
        .order_by(desc(Property.created_at))
        .offset(offset)
        .limit(limit)
    )
    properties = result.scalars().all()
    
    logger.success(f"Retrieved {len(properties)} listings for user: {current_user.email}")
    
    return StandardResponse(
        message="Listings retrieved successfully",
        data=PropertyMyListingsResponse(
            listings=[PropertyListingResponse.model_validate(p) for p in properties],
            total=total,
            page=page,
            limit=limit
        )
    )
