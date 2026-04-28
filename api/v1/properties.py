"""
Properties endpoint for managing real estate listings.
Includes CRUD operations and property search functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import desc
from datetime import datetime, timezone
import uuid
from typing import Optional

from sqlalchemy import func, desc, or_
from sqlalchemy.orm import selectinload, joinedload
from core.enums import PropertyType, PriceType


from api.deps import get_current_user, get_current_user_optional
from db.session import get_db
from db.models.user import User
from db.models.property import Property
from core.enums import PropertyStatus, UserRole
from schemas.property import (
    PropertyCreate, PropertyUpdate, PropertyResponse, PropertyListWithPagination
)
from schemas.base import StandardResponse
from loguru import logger

router = APIRouter()



@router.get("/meta/filters", tags=["Properties"])
async def get_filter_metadata(db: AsyncSession = Depends(get_db)):
    """
    Returns available locations and types to populate frontend dropdowns.
    Optimized to only return locations that actually have active listings.
    """

    result = await db.execute(
        select(Property.location)
        .where(Property.status == PropertyStatus.APPROVED)
        .distinct()
    )
    locations = result.scalars().all()
    
    return StandardResponse(
        message="Filter metadata retrieved",
        data={
            "property_types": [e.value for e in PropertyType],
            "price_types": [e.value for e in PriceType],
            "locations": locations,
            "price_steps": [500000, 1000000, 5000000, 10000000, 50000000]
        }
    )


@router.get("/", response_model=StandardResponse[PropertyListWithPagination], tags=["Properties"])
async def list_properties(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    property_type: Optional[PropertyType] = Query(None),
    bedrooms: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    # 1. Base query with optimized relationship loading
    query = select(Property).options(selectinload(Property.lister))

    # 2. Permission-based Visibility
    if not current_user or current_user.role == UserRole.SEEKER:
        query = query.where(Property.status == PropertyStatus.APPROVED)
    elif current_user.role == UserRole.LISTER:
        query = query.where(
            or_(
                Property.status == PropertyStatus.APPROVED,
                Property.lister_id == current_user.id
            )
        )

    if status:
        try:
            query = query.where(Property.status == PropertyStatus[status.upper()])
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    if location and location.lower() != "all":
        query = query.where(Property.location.ilike(f"%{location}%"))

    if min_price:
        query = query.where(Property.price >= min_price)
    
    if max_price:
        query = query.where(Property.price <= max_price)
    
    if property_type:
        query = query.where(Property.property_type == property_type)
        
    if bedrooms:
        query = query.where(Property.bedrooms >= bedrooms)

    count_stmt = select(func.count()).select_from(query.subquery())
    total_count = (await db.execute(count_stmt)).scalar_one()

    offset = (page - 1) * limit
    result = await db.execute(
        query.order_by(desc(Property.created_at))
             .offset(offset)
             .limit(limit)
    )
    
    properties = result.unique().scalars().all()
    pages = (total_count + limit - 1) // limit

    return StandardResponse(
        message="Properties retrieved successfully",
        data=PropertyListWithPagination(
            properties=properties,
            total=total_count,
            page=page,
            limit=limit,
            pages=pages
        )
    )

@router.get("/{property_id}", response_model=StandardResponse[PropertyResponse], tags=["Properties"])
async def get_property(
    property_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional)
):
    """
    Get a specific property by ID.
    Only returns APPROVED properties unless user is admin/lister of that property.
    """
    result = await db.execute(
        select(Property)
        .where(Property.id == property_id)
        .options(joinedload(Property.lister))
    )
    property_obj = result.unique().scalars().first()
    
    if not property_obj:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check access
    if property_obj.status != PropertyStatus.APPROVED:
        if not current_user:
            raise HTTPException(status_code=403, detail="Property not accessible")
        if current_user.role == UserRole.SEEKER:
            raise HTTPException(status_code=403, detail="Property not accessible")
        if current_user.role == UserRole.LISTER and property_obj.lister_id != current_user.id:
            raise HTTPException(status_code=403, detail="Property not accessible")
    
    logger.info(f"Retrieved property: {property_id}")
    
    return StandardResponse(
        message="Property retrieved successfully",
        data=property_obj
    )


@router.post("/", response_model=StandardResponse[PropertyResponse], tags=["Properties"])
async def create_property(
    property_data: PropertyCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new property listing.
    Only listers can create properties.
    """
    # Check authorization
    if current_user.role != UserRole.LISTER:
        raise HTTPException(
            status_code=403,
            detail="Only listers can create property listings"
        )
    
    # Create property
    new_property = Property(
        id=uuid.uuid4(),
        lister_id=current_user.id,
        title=property_data.title,
        description=property_data.description,
        price=property_data.price,
        price_type=property_data.price_type,
        location=property_data.location,
        latitude=property_data.latitude,
        longitude=property_data.longitude,
        bedrooms=property_data.bedrooms,
        bathrooms=property_data.bathrooms,
        property_type=property_data.property_type,
        amenities=property_data.amenities or [],
        images=property_data.images or [],
        thumbnail=property_data.thumbnail,
        status=PropertyStatus.PENDING_REVIEW,
        created_at=datetime.now(timezone.utc)
    )
    
    db.add(new_property)
    await db.commit()
    await db.refresh(new_property)
    
    logger.info(f"Created property: {new_property.id} by lister {current_user.id}")
    
    return StandardResponse(
        message="Property created successfully",
        data=new_property
    )


@router.put("/{property_id}", response_model=StandardResponse[PropertyResponse], tags=["Properties"])
async def update_property(
    property_id: uuid.UUID,
    property_data: PropertyUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a property listing.
    Only the lister who created it can update it.
    """
    result = await db.execute(
        select(Property)
        .where(Property.id == property_id)
        .options(joinedload(Property.lister))
    )
    property_obj = result.unique().scalars().first()
    
    if not property_obj:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check authorization
    if property_obj.lister_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="You can only update your own properties"
        )
    
    # Update fields
    update_data = property_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(property_obj, field, value)
    
    property_obj.updated_at = datetime.now(timezone.utc)
    
    db.add(property_obj)
    await db.commit()
    await db.refresh(property_obj)
    
    logger.info(f"Updated property: {property_id}")
    
    return StandardResponse(
        message="Property updated successfully",
        data=property_obj
    )


@router.delete("/{property_id}", tags=["Properties"])
async def delete_property(
    property_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a property listing.
    Only the lister who created it or admins can delete it.
    """
    result = await db.execute(
        select(Property).where(Property.id == property_id)
    )
    property_obj = result.scalars().first()
    
    if not property_obj:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Check authorization
    if property_obj.lister_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=403,
            detail="You can only delete your own properties"
        )
    
    await db.delete(property_obj)
    await db.commit()
    
    logger.info(f"Deleted property: {property_id}")
    
    return StandardResponse(
        message="Property deleted successfully",
        data=None
    )