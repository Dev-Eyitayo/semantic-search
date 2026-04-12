from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class UploadListingImageResponse(BaseModel):
    """Response from listing image upload"""
    secure_url: str
    public_id: str
    width: int
    height: int
    format: str
    resource_type: str
    url: Optional[str] = None


class UploadAvatarResponse(BaseModel):
    """Response from avatar upload"""
    secure_url: str
    public_id: str
    width: int
    height: int
    url: Optional[str] = None


class DeleteImageRequest(BaseModel):
    """Request to delete an image"""
    public_id: str = Field(..., description="Cloudinary public ID of the image")


class DeleteImageResponse(BaseModel):
    """Response from image deletion"""
    message: str = "Image deleted from Cloudinary"
    public_id: str
