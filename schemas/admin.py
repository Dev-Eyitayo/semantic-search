"""Admin service schemas for listing and user management"""

from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from core.enums import PropertyStatus, UserRole, PriceType


class ListerInfo(BaseModel):
    """Basic lister information for admin review"""
    id: UUID
    full_name: str
    email: str
    phone: Optional[str] = None
    is_verified: bool
    created_at: datetime


class PendingListingResponse(BaseModel):
    """A pending listing for admin review"""
    id: UUID
    title: str
    price: float
    price_type: str
    location: str
    bedrooms: int
    bathrooms: int
    property_type: str
    lister: ListerInfo
    submitted_at: datetime


class ListPendingListingsResponse(BaseModel):
    """Response for listing pending properties"""
    listings: List[PendingListingResponse]
    total: int
    page: int
    limit: int


class ApproveListingRequest(BaseModel):
    """Request to approve a pending listing"""
    notes: Optional[str] = Field(None, max_length=500)


class ApproveListingResponse(BaseModel):
    """Response after approving a listing"""
    message: str
    property_id: UUID
    status: str
    embedding_generated: bool
    indexed_at: datetime


class RejectListingRequest(BaseModel):
    """Request to reject a pending listing"""
    reason: str = Field(..., min_length=10, max_length=300)
    notes: Optional[str] = Field(None, max_length=500)


class RejectListingResponse(BaseModel):
    """Response after rejecting a listing"""
    message: str
    property_id: UUID
    status: str


class AdminAuditEntry(BaseModel):
    """Single audit log entry"""
    id: UUID
    action: str
    admin: Optional[dict]
    notes: Optional[str]
    performed_at: datetime


class ListingAuditLogResponse(BaseModel):
    """Response containing full audit trail for a listing"""
    property_id: UUID
    audit_log: List[AdminAuditEntry]


class UserInfo(BaseModel):
    """User information for admin view"""
    id: UUID
    full_name: str
    email: str
    phone: Optional[str] = None
    role: str
    is_verified: bool
    created_at: datetime


class ListUsersResponse(BaseModel):
    """Response for listing users"""
    users: List[UserInfo]
    total: int
    page: int
    limit: int


class UpdateUserRequest(BaseModel):
    """Request to update user role or status"""
    role: Optional[str] = None
    is_verified: Optional[bool] = None


class UpdateUserResponse(BaseModel):
    """Response after updating user"""
    message: str
    user: UserInfo


class SearchAnalyticsQuery(BaseModel):
    """Single top search query entry"""
    query: str
    count: int


class DailySearchEntry(BaseModel):
    """Daily search volume entry"""
    date: str
    count: int


class SearchAnalyticsResponse(BaseModel):
    """Response containing search analytics"""
    period: str
    total_searches: int
    avg_response_time_ms: float
    semantic_search_pct: float
    fallback_search_pct: float
    top_queries: List[SearchAnalyticsQuery]
    searches_by_day: List[DailySearchEntry]
