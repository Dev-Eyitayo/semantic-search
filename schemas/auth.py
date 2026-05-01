import re
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict
from typing import Optional
from uuid import UUID
from core.enums import UserRole
from core.phone_utils import normalize_nigerian_phone

class UserCreate(BaseModel):
    email: EmailStr
    first_name: str = Field(..., min_length=2, max_length=100)
    last_name: str = Field(..., min_length=2, max_length=100)
    phone: Optional[str] = Field(None, description="Nigerian phone number (08012345678, 2348012345678, or +2348012345678)")
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.SEEKER

    @field_validator("email")
    @classmethod
    def email_to_lower(cls, v: str) -> str:
        return v.lower()

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        if v:
            return normalize_nigerian_phone(v)
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        # Requirement: 1 uppercase, 1 number, 1 special char 
        if not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', v):
            raise ValueError("Password must contain at least 1 uppercase letter, 1 number, and 1 special character")
        return v
    
class VerifyEmailRequest(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)

class ResendOTPRequest(BaseModel):
    email: EmailStr

class Token(BaseModel):
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours 
    role: UserRole
    user_id: UUID

class UserResponse(BaseModel):
    message: str = "Registration successful. Check your email."
    user_id: UUID
    role: UserRole

    model_config = ConfigDict(from_attributes=True)

class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=72)

    @field_validator("email")
    @classmethod
    def email_to_lower(cls, v: str) -> str:
        return v.lower()


class TokenRefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="The valid refresh token issued at login")


class PasswordResetRequest(BaseModel):
    email: EmailStr

    @field_validator("email")
    @classmethod
    def email_to_lower(cls, v: str) -> str:
        return v.lower()


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours


class PasswordResetConfirm(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        # Requirement: 1 uppercase, 1 number, 1 special char 
        if not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', v):
            raise ValueError("Password must contain at least 1 uppercase letter, 1 number, and 1 special character")
        return v