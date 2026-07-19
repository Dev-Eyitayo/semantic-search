"""Admin service endpoints for listing management and analytics"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select, and_, or_
from sqlalchemy.orm import joinedload
from datetime import datetime, timezone, timedelta
from uuid import UUID
from typing import Optional

from api.deps import get_current_user, RoleChecker
from db.session import get_db
from db.models.user import User
from db.models.property import Property
from db.models.search import SearchLog, AdminAuditLog
from core.enums import PropertyStatus, UserRole, PriceType
from schemas.admin import (
    ListPendingListingsResponse, PendingListingResponse, ListerInfo,
    ApproveListingRequest, ApproveListingResponse,
    RejectListingRequest, RejectListingResponse,
    ListingAuditLogResponse, AdminAuditEntry,
    ListUsersResponse, UserInfo,
    UpdateUserRequest, UpdateUserResponse,
    SearchAnalyticsResponse, SearchAnalyticsQuery, DailySearchEntry
)
from schemas.base import StandardResponse
from services.ai_service import generate_embedding
from services.mail_service import send_listing_approved_email, send_listing_rejected_email
from loguru import logger

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


@router.get(
    "/listings/pending",
    response_model=StandardResponse[ListPendingListingsResponse],
    tags=["Admin/Listings"]
)
async def list_pending_listings(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("oldest", regex="^(oldest|newest)$"),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    """
    Retrieve all pending listings awaiting admin review.
    Includes lister verification status for decision-making.
    """
    logger.info(f"Admin {admin_user.id} retrieving pending listings - Page: {page}, Limit: {limit}")
    
    # Base query for pending listings
    query = select(Property).where(
        Property.status == PropertyStatus.PENDING_REVIEW
    )
    
    # Sort by submitted date
    if sort_by == "oldest":
        query = query.order_by(Property.created_at.asc())
    else:
        query = query.order_by(Property.created_at.desc())
    
    # Get total count
    count_query = select(func.count(Property.id)).where(
        Property.status == PropertyStatus.PENDING_REVIEW
    )
    count_result = await db.execute(count_query)
    total = count_result.scalar()
    
    # Paginate
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    properties = result.scalars().all()
    
    # Fetch lister information for each property
    listings = []
    for prop in properties:
        lister_result = await db.execute(
            select(User).where(User.id == prop.lister_id)
        )
        lister = lister_result.scalars().first()
        
        if lister:
            listings.append(
                PendingListingResponse(
                    id=prop.id,
                    title=prop.title,
                    price=prop.price,
                    price_type=prop.price_type.value,
                    location=prop.location,
                    bedrooms=prop.bedrooms,
                    bathrooms=prop.bathrooms,
                    property_type=prop.property_type.value,
                    lister=ListerInfo(
                        id=lister.id,
                        full_name=f"{lister.first_name} {lister.last_name}",
                        email=lister.email,
                        phone=lister.phone,
                        is_verified=lister.is_verified,
                        created_at=lister.created_at
                    ),
                    submitted_at=prop.created_at
                )
            )
    
    logger.success(f"Retrieved {len(listings)} pending listings for admin review")
    
    return StandardResponse(
        message="Pending listings retrieved successfully",
        data=ListPendingListingsResponse(
            listings=listings,
            total=total,
            page=page,
            limit=limit
        )
    )


@router.post(
    "/listings/{property_id}/approve",
    response_model=StandardResponse[ApproveListingResponse],
    tags=["Admin/Listings"]
)
async def approve_listing(
    property_id: UUID,
    request: ApproveListingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    """
    Approve a pending listing and trigger S-BERT embedding generation.
    Embedding will be indexed into pgvector for semantic search.
    """
    logger.info(f"Admin {admin_user.id} approving listing {property_id}")
    
    # Get property
    result = await db.execute(
        select(Property).where(Property.id == property_id)
    )
    prop = result.scalars().first()
    
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    if prop.status != PropertyStatus.PENDING_REVIEW:
        raise HTTPException(
            status_code=409,
            detail="Only pending listings can be approved"
        )
    
    try:
        # PHASE 1: Generate embedding for the property
        property_text = f"{prop.title}. {prop.description}"
        embedding, processing_time_ms = generate_embedding(
            text=property_text,
            normalize=True
        )
        
        # PHASE 2: Update property status and embedding
        prop.status = PropertyStatus.APPROVED
        prop.embedding = embedding
        prop.updated_at = datetime.now(timezone.utc)
        
        # PHASE 3: Log admin action
        audit_log = AdminAuditLog(
            action="APPROVE_LISTING",
            admin_id=admin_user.id,
            listing_id=property_id,
            details={
                "reason": "Listed approved by admin",
                "notes": request.notes,
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_time_ms": processing_time_ms
            }
        )
        db.add(audit_log)
        
        # PHASE 4: Commit all changes
        await db.commit()
        await db.refresh(prop)
        
        # Send notification email in background
        lister = await db.execute(select(User).where(User.id == prop.lister_id))
        lister_user = lister.scalars().first()
        if lister_user:
            background_tasks.add_task(
                send_listing_approved_email,
                lister_user.email,
                lister_user.first_name,
                prop.title,
                prop.location,
                prop.updated_at.strftime("%B %d, %Y")
            )
        
        logger.success(f"Listing {property_id} approved and indexed for search")
        
        return StandardResponse(
            message="Listing approved and indexed for search",
            data=ApproveListingResponse(
                property_id=property_id,
                message="Listing approved and indexed for search",
                status=PropertyStatus.APPROVED.value,
                embedding_generated=True,
                indexed_at=datetime.now(timezone.utc)
            )
        )
    
    except Exception as e:
        await db.rollback()
        logger.error(f"Approval failed for listing {property_id}: {e}")
        raise HTTPException(status_code=503, detail="Approval processing failed")


@router.post(
    "/listings/{property_id}/reject",
    response_model=StandardResponse[RejectListingResponse],
    tags=["Admin/Listings"]
)
async def reject_listing(
    property_id: UUID,
    request: RejectListingRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    """
    Reject a pending listing with mandatory reason.
    Lister is notified via email about the rejection.
    """
    logger.info(f"Admin {admin_user.id} rejecting listing {property_id}")
    
    # Get property
    result = await db.execute(
        select(Property).where(Property.id == property_id)
    )
    prop = result.scalars().first()
    
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    if prop.status != PropertyStatus.PENDING_REVIEW:
        raise HTTPException(
            status_code=409,
            detail="Only pending listings can be rejected"
        )
    
    try:
        # Update property
        prop.status = PropertyStatus.REJECTED
        prop.rejection_reason = request.reason
        prop.updated_at = datetime.now(timezone.utc)
        
        # Log admin action
        audit_log = AdminAuditLog(
            action="REJECT_LISTING",
            admin_id=admin_user.id,
            listing_id=property_id,
            details={
                "rejection_reason": request.reason,
                "internal_notes": request.notes
            }
        )
        db.add(audit_log)
        
        await db.commit()
        await db.refresh(prop)
        
        # Send rejection email
        lister = await db.execute(select(User).where(User.id == prop.lister_id))
        lister_user = lister.scalars().first()
        if lister_user:
            background_tasks.add_task(
                send_listing_rejected_email,
                lister_user.email,
                lister_user.first_name,
                prop.title,
                prop.location,
                request.reason
            )
        
        logger.success(f"Listing {property_id} rejected - Notified lister")
        
        return StandardResponse(
            message="Listing rejected. Lister has been notified.",
            data=RejectListingResponse(
                property_id=property_id,
                message="Listing rejected. Lister has been notified.",
                status=PropertyStatus.REJECTED.value
            )
        )
    
    except Exception as e:
        await db.rollback()
        logger.error(f"Rejection failed for listing {property_id}: {e}")
        raise HTTPException(status_code=503, detail="Rejection processing failed")


@router.get(
    "/listings/{property_id}/audit",
    response_model=StandardResponse[ListingAuditLogResponse],
    tags=["Admin/Listings"]
)
async def get_listing_audit(
    property_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    """
    Retrieve the full audit trail for a specific listing.
    Includes all admin actions: approvals, rejections, flags, etc.
    """
    logger.info(f"Admin {admin_user.id} retrieving audit log for listing {property_id}")
    
    # Verify property exists
    result = await db.execute(
        select(Property).where(Property.id == property_id)
    )
    prop = result.scalars().first()
    
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    
    # Get all audit logs for this property
    audit_query = select(AdminAuditLog).where(
        AdminAuditLog.target_id == property_id
    ).order_by(AdminAuditLog.timestamp.desc())
    
    audit_result = await db.execute(audit_query)
    audit_logs = audit_result.scalars().all()
    
    # Format audit entries
    entries = []
    for log in audit_logs:
        # Get admin info
        if log.user_id:
            admin_result = await db.execute(
                select(User).where(User.id == log.user_id)
            )
            admin_user_obj = admin_result.scalars().first()
            admin_info = {
                "id": str(admin_user_obj.id),
                "full_name": f"{admin_user_obj.first_name} {admin_user_obj.last_name}"
            } if admin_user_obj else None
        else:
            admin_info = None
        
        entries.append(
            AdminAuditEntry(
                id=log.id,
                action=log.action,
                admin=admin_info,
                notes=log.details.get("rejection_reason") if log.details else None,
                performed_at=log.timestamp
            )
        )
    
    logger.success(f"Retrieved {len(entries)} audit entries for listing {property_id}")
    
    return StandardResponse(
        message="Listing audit trail retrieved successfully",
        data=ListingAuditLogResponse(
            property_id=property_id,
            audit_log=entries
        )
    )


@router.get(
    "/users",
    response_model=StandardResponse[ListUsersResponse],
    tags=["Admin/Users"]
)
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    role: Optional[str] = Query(None),
    is_verified: Optional[bool] = Query(None),
    search: Optional[str] = Query(None, max_length=100),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    """
    List all registered users with optional filters.
    Admin can filter by role, verification status, and search by name/email.
    """
    logger.info(f"Admin {admin_user.id} retrieving users list")
    
    # Build query
    query = select(User)
    
    # Apply filters
    filters = []
    if role:
        filters.append(User.role == role)
    if is_verified is not None:
        filters.append(User.is_verified == is_verified)
    if search:
        filters.append(
            or_(
                User.first_name.ilike(f"%{search}%"),
                User.last_name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%")
            )
        )
    
    if filters:
        query = query.where(and_(*filters))
    
    # Count total
    count_query = query.with_only_columns(func.count(User.id))
    count_result = await db.execute(count_query)
    total = count_result.scalar()
    
    # Paginate
    offset = (page - 1) * limit
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Format response
    user_list = [
        UserInfo(
            id=user.id,
            full_name=f"{user.first_name} {user.last_name}",
            email=user.email,
            phone=user.phone,
            role=user.role.value,
            is_verified=user.is_verified,
            created_at=user.created_at
        )
        for user in users
    ]
    
    logger.success(f"Retrieved {len(users)} users")
    
    return StandardResponse(
        message="Users retrieved successfully",
        data=ListUsersResponse(
            users=user_list,
            total=total,
            page=page,
            limit=limit
        )
    )


@router.patch(
    "/users/{user_id}",
    response_model=StandardResponse[UpdateUserResponse],
    tags=["Admin/Users"]
)
async def update_user(
    user_id: UUID,
    request: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    """
    Update a user's role or verification status.
    Admins cannot demote other admins (prevent accidental lockout).
    """
    logger.info(f"Admin {admin_user.id} updating user {user_id}")
    
    if not request.role and request.is_verified is None:
        raise HTTPException(
            status_code=400,
            detail="At least one field (role or is_verified) must be provided"
        )
    
    # Get user
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent demoting admins
    if request.role and user.role == UserRole.ADMIN and request.role != UserRole.ADMIN.value:
        raise HTTPException(
            status_code=403,
            detail="Cannot demote another admin"
        )
    
    # Update fields
    if request.role:
        user.role = UserRole(request.role)
    if request.is_verified is not None:
        user.is_verified = request.is_verified
    
    user.updated_at = datetime.now(timezone.utc)
    
    # Log action
    audit_log = AdminAuditLog(
        action="UPDATE_USER",
        admin_id=admin_user.id,
        listing_id=None,
        details={
            "user_id": str(user_id),
            "role": request.role,
            "is_verified": request.is_verified
        }
    )
    db.add(audit_log)
    
    await db.commit()
    await db.refresh(user)
    
    logger.success(f"User {user_id} updated successfully")
    
    return StandardResponse(
        message="User updated successfully",
        data=UpdateUserResponse(
            message="User updated successfully",
            user=UserInfo(
                id=user.id,
                full_name=f"{user.first_name} {user.last_name}",
                email=user.email,
                phone=user.phone,
                role=user.role.value,
                is_verified=user.is_verified,
                created_at=user.created_at
            )
        )
    )


@router.get(
    "/analytics/search",
    response_model=StandardResponse[SearchAnalyticsResponse],
    tags=["Admin/Analytics"]
)
async def get_search_analytics(
    period: str = Query("7d", regex="^(today|7d|30d)$"),
    db: AsyncSession = Depends(get_db),
    admin_user: User = Depends(RoleChecker([UserRole.ADMIN]))
):
    """
    Retrieve aggregated search analytics for the specified period.
    Includes top queries, average response times, search volume, and fallback rate.
    """
    logger.info(f"Admin {admin_user.id} retrieving search analytics - Period: {period}")
    
    # Determine date range
    now = datetime.now(timezone.utc)
    if period == "today":
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "7d":
        start_date = now - timedelta(days=7)
    else:  # 30d
        start_date = now - timedelta(days=30)
    
    # Get all searches in period
    search_query = select(SearchLog).where(
        SearchLog.created_at >= start_date
    )
    
    result = await db.execute(search_query)
    searches = result.scalars().all()
    
    if not searches:
        return StandardResponse(
            message="Search analytics retrieved successfully",
            data=SearchAnalyticsResponse(
                period=period,
                total_searches=0,
                avg_response_time_ms=0.0,
                semantic_search_pct=0.0,
                fallback_search_pct=0.0,
                top_queries=[],
                searches_by_day=[]
            )
        )
    
    # Calculate metrics
    total_searches = len(searches)
    avg_response_time = sum(s.processing_time_ms for s in searches) / total_searches if searches else 0
    
    semantic_count = sum(1 for s in searches if s.search_type == "semantic")
    fallback_count = sum(1 for s in searches if s.search_type == "fallback")
    
    semantic_pct = (semantic_count / total_searches * 100) if total_searches > 0 else 0
    fallback_pct = (fallback_count / total_searches * 100) if total_searches > 0 else 0
    
    # Get top queries
    query_counts = {}
    for search in searches:
        if search.query and search.query.strip():
            query_counts[search.query] = query_counts.get(search.query, 0) + 1
    
    top_queries = sorted(
        query_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    top_queries_list = [
        SearchAnalyticsQuery(query=q, count=c)
        for q, c in top_queries
    ]
    
    # Calculate searches by day
    day_counts = {}
    for search in searches:
        day_key = search.created_at.strftime("%Y-%m-%d")
        day_counts[day_key] = day_counts.get(day_key, 0) + 1
    
    searches_by_day = [
        DailySearchEntry(date=day, count=count)
        for day, count in sorted(day_counts.items())
    ]
    
    logger.success(f"Retrieved search analytics - Total searches: {total_searches}")
    
    return StandardResponse(
        message="Search analytics retrieved successfully",
        data=SearchAnalyticsResponse(
            period=period,
            total_searches=total_searches,
            avg_response_time_ms=round(avg_response_time, 2),
            semantic_search_pct=round(semantic_pct, 1),
            fallback_search_pct=round(fallback_pct, 1),
            top_queries=top_queries_list,
            searches_by_day=searches_by_day
        )
    )
