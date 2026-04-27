import re
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict
from fastapi import Form
from typing import Optional
from uuid import UUID
from datetime import datetime
from core.enums import UserRole
from core.phone_utils import normalize_nigerian_phone


class UserResponse(BaseModel):
    id: UUID
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    role: UserRole
    is_verified: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)



class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=2, max_length=100)
    last_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = Field(None)

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return normalize_nigerian_phone(v)
        return v

    @classmethod
    def as_form(
        cls,
        first_name: Optional[str] = Form(None),
        last_name: Optional[str] = Form(None),
        phone: Optional[str] = Form(None),
    ):
        return cls(
            first_name=first_name, 
            last_name=last_name, 
            phone=phone
        )

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        if not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', v):
            raise ValueError("Password must contain at least 1 uppercase letter, 1 number, and 1 special character")
        return v


class SavedSearchCreate(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    filters: Optional[dict] = None
    notify_on_match: bool = False


class SavedSearchResponse(BaseModel):
    id: UUID
    query: str
    filters: Optional[dict] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SavedSearchList(BaseModel):
    saved_searches: list[SavedSearchResponse]
    total: int
    page: int
    limit: int
