from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from db.models.user import User
from core.config import settings
from services.redis_service import is_token_blacklisted
from db.session import get_db
from fastapi import Request
from core.permissions import IsAdmin, IsLister, IsAuthenticated, IsListerOrAdmin
from loguru import logger
import uuid


security_scheme = HTTPBearer(description="Bearer token - paste your JWT access token here", auto_error=False)


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> User:
    logger.info(f"Cookies detected: {request.cookies}")
    logger.info(f"Authorization Header detected: {request.headers.get('authorization')}")

    token = request.cookies.get("access_token")

    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ")[1]
    if not token:
        logger.warning("No token provided in cookie or header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Not authenticated"
        )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        jti: str = payload.get("jti")
        
        if await is_token_blacklisted(jti):
            raise HTTPException(status_code=401, detail="Token has been revoked")
            
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
            
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalars().first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.is_verified:
            raise HTTPException(status_code=403, detail="Please verify your email")
            
        return user

    except JWTError as e:
        logger.error(f"JWT Validation Error: {str(e)}")
        raise HTTPException(status_code=401, detail="Could not validate token")
    

async def get_current_user_optional(
    request: Request,
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    try:
        return await get_current_user(request, db)
    except HTTPException:
        return None
class RoleChecker:
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