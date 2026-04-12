"""
Provides helper functions and decorators for applying permissions to endpoints.
"""

from fastapi import HTTPException, status
from db.models.user import User
from sqlalchemy.ext.asyncio import AsyncSession
from core.permissions import BasePermission
from loguru import logger
from typing import Callable, Awaitable


async def check_object_permission(
    permission: BasePermission,
    user: User,
    obj_id: str,
    db: AsyncSession
) -> bool:
    """
    Check object-level permission.
    
    Args:
        permission: Permission instance to check
        user: Current user
        obj_id: ID of the object to check
        db: Database session
    
    Returns:
        True if permission granted, False otherwise
        
    Raises:
        HTTPException: If permission denied
    """
    has_access = await permission.has_object_permission(user, obj_id, db)
    
    if not has_access:
        logger.warning(f"Object permission denied for user {user.email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized. {permission.__class__.__name__} permission required."
        )
    
    return True


async def check_permissions(
    user: User,
    *permissions: BasePermission
) -> bool:
    """
    Check multiple permissions for a user.
    
    Args:
        user: Current user
        *permissions: Permission instances to check
    
    Returns:
        True if all permissions granted
        
    Raises:
        HTTPException: If any permission denied
    """
    for permission in permissions:
        if not permission.has_permission(user):
            logger.warning(f"Permission denied for user {user.email}: {permission.__class__.__name__}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized. {permission.__class__.__name__} permission required."
            )
    
    return True


def permission_required(*permissions: BasePermission):
    """
    Decorator for checking permissions on an endpoint.
    
    Usage:
        @permission_required(IsAuthenticated(), IsLister())
        async def my_endpoint(current_user: User = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs
            user = kwargs.get("current_user")
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            await check_permissions(user, *permissions)
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator
