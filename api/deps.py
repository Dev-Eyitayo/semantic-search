from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from db.models.user import User
from core.config import settings
from services.redis_service import is_token_blacklisted
from db.session import get_db
from core.permissions import IsAdmin, IsLister, IsAuthenticated, IsListerOrAdmin
from loguru import logger
import uuid

# Use HTTPBearer instead of OAuth2PasswordBearer for simple Bearer token input in Swagger
security_scheme = HTTPBearer(description="Bearer token - paste your JWT access token here")

async def get_current_user(
    credentials = Depends(security_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Extract and validate JWT token from Authorization header.
    Returns authenticated User object.
    """
    try:
        # Extract token from credentials
        token = credentials.credentials
        
        # Decode JWT
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        jti: str = payload.get("jti")
        
        # Check if token is blacklisted
        if await is_token_blacklisted(jti):
            logger.warning(f"Attempt to use revoked token with JTI: {jti}")
            raise HTTPException(status_code=401, detail="Token has been revoked")
            
        if user_id is None:
            logger.warning("Invalid token: missing 'sub' claim")
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except JWTError as e:
        logger.warning(f"JWT validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Could not validate token")

    # Fetch user from database
    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalars().first()
    
    if not user:
        logger.warning(f"User not found for ID: {user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user.is_verified:
        logger.warning(f"Unverified user attempted access: {user.email}")
        raise HTTPException(status_code=403, detail="Please verify your email to enable access")
        
    return user


class RoleChecker:
    """
    Role-based access control checker.
    
    DEPRECATED: Use core.permissions classes instead for cleaner, DRF-style permissions.
    
    Example with new system:
        @router.post("/listing")
        async def create_listing(
            current_user: User = Depends(get_current_user),
            _=Depends(IsListerOrAdmin())
        ):
            ...
    """
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, user: User = Depends(get_current_user)):
        if user.role not in self.allowed_roles:
            logger.warning(f"Role-based access denied for user {user.email} with role {user.role}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail=f"Role {user.role} is not authorized to access this resource"
            )
        return user


# Pre-configured permission dependencies for backward compatibility
is_admin = RoleChecker(["admin"])
is_lister = RoleChecker(["lister", "admin"])