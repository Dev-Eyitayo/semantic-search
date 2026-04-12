from enum import Enum as PyEnum


class UserRole(str, PyEnum):
    """User roles in the system"""
    SEEKER = "seeker"
    LISTER = "lister"
    ADMIN = "admin"


class PriceType(str, PyEnum):
    """Property price types"""
    RENT = "rent"
    SALE = "sale"


class PropertyType(str, PyEnum):
    """Property types"""
    APARTMENT = "apartment"
    HOUSE = "house"
    DUPLEX = "duplex"
    STUDIO = "studio"
    COMMERCIAL = "commercial"


class PropertyStatus(str, PyEnum):
    """Property listing status"""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    DELETED = "deleted"
