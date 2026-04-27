from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from core.enums import PropertyType, PriceType, PropertyStatus


class PropertyCreate(BaseModel):
    title: str = Field(..., min_length=10, max_length=255)
    description: str = Field(..., min_length=100, max_length=5000)
    price: float = Field(..., gt=0)
    price_type: PriceType
    location: str = Field(..., min_length=5, max_length=255)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bedrooms: int = Field(..., ge=0, le=20)
    bathrooms: int = Field(..., ge=0, le=20)
    property_type: PropertyType
    amenities: Optional[List[str]] = None
    images: Optional[List[str]] = Field(None, max_length=15)

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        if v and len(v) > 15:
            raise ValueError("Maximum 15 images allowed per listing")
        if v:
            for img in v:
                if not img.startswith("https://res.cloudinary.com/"):
                    raise ValueError("Images must be Cloudinary URLs")
        return v


class PropertyUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=10, max_length=255)
    description: Optional[str] = Field(None, min_length=100, max_length=5000)
    price: Optional[float] = Field(None, gt=0)
    price_type: Optional[PriceType] = None
    location: Optional[str] = Field(None, min_length=5, max_length=255)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bedrooms: Optional[int] = Field(None, ge=0, le=20)
    bathrooms: Optional[int] = Field(None, ge=0, le=20)
    property_type: Optional[PropertyType] = None
    amenities: Optional[List[str]] = None
    images: Optional[List[str]] = Field(None, max_length=15)

    @field_validator("images")
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        if v and len(v) > 15:
            raise ValueError("Maximum 15 images allowed per listing")
        if v:
            for img in v:
                if not img.startswith("https://res.cloudinary.com/"):
                    raise ValueError("Images must be Cloudinary URLs")
        return v


class ListerInfo(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    phone: Optional[str] = None
    avatar_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PropertyResponse(BaseModel):
    id: UUID
    title: str
    description: str
    price: float
    price_type: PriceType
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    bedrooms: int
    bathrooms: int
    property_type: PropertyType
    amenities: Optional[List[str]] = None
    images: List[str]
    thumbnail: Optional[str] = None
    status: PropertyStatus
    view_count: int
    created_at: datetime
    lister: Optional[ListerInfo] = None
    
    model_config = ConfigDict(from_attributes=True)


class PropertyListResponse(BaseModel):
    id: UUID
    title: str
    price: float
    price_type: PriceType
    location: str
    bedrooms: int
    bathrooms: int
    property_type: PropertyType
    lister: Optional[ListerInfo] = None
    thumbnail: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PropertyListWithPagination(BaseModel):
    properties: List[PropertyListResponse]
    total: int
    page: int
    limit: int
    pages: int


class PropertyListingResponse(BaseModel):
    id: UUID
    title: str
    status: PropertyStatus
    price: float
    created_at: datetime
    thumbnail: Optional[str] = None
    lister: Optional[ListerInfo] = None
    rejection_reason: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PropertyMyListingsResponse(BaseModel):
    listings: List[PropertyListingResponse]
    total: int
    page: int
    limit: int
