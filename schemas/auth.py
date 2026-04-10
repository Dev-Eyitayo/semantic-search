import re
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict
from typing import Optional
from uuid import UUID

class UserCreate(BaseModel):
    email: EmailStr
    first_name: str = Field(..., min_length=2, max_length=100)
    last_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8)
    role: str = "seeker"

    @field_validator("email")
    @classmethod
    def email_to_lower(cls, v: str) -> str:
        return v.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        # Requirement: 1 uppercase, 1 number, 1 special char 
        if not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', v):
            raise ValueError("Password must contain at least 1 uppercase letter, 1 number, and 1 special character")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ["seeker", "lister"]:
            raise ValueError("Role must be one of: seeker, lister")
        return v
    

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours 
    role: str
    user_id: UUID

class UserResponse(BaseModel):
    message: str = "Registration successful. Check your email."
    user_id: UUID
    role: str

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

class PasswordResetConfirm(BaseModel):
    token: str 
    new_password: str = Field(..., min_length=8)

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not re.match(r'^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$', v):
            raise ValueError("Password must contain at least 1 uppercase, 1 number, and 1 special char")
        return v