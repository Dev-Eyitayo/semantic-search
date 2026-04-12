"""
Provides reusable permission classes for role-based access control (RBAC).
Each permission class implements:
- has_permission(): Check if user can access the resource
- has_object_permission(): Check if user can access/modify a specific object

Usage in routers:
    @router.get("/endpoint")
    async def get_endpoint(
        current_user: User = Depends(get_current_user),
        _=Depends(IsAuthenticated())
    ):
        ...
        
    @router.put("/resource/{id}")
    async def update_resource(
        resource_id: str,
        current_user: User = Depends(get_current_user),
        db: AsyncSession = Depends(get_db),
        _=Depends(IsObjectOwner(Property, "id", "lister_id"))
    ):
        ...
"""

from abc import ABC, abstractmethod
from fastapi import Depends, HTTPException, status
from typing import Optional, Type, Any
from db.models.user import User
from db.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from loguru import logger
import uuid


class BasePermission(ABC):
    """
    Base permission class. All permission classes should inherit from this.
    """
    
    @abstractmethod
    def has_permission(self, user: User) -> bool:
        """
        Check if user has permission to access the resource.
        Return True if permission is granted, False otherwise.
        """
        raise NotImplementedError
    
    async def has_object_permission(
        self, user: User, obj: Any, db: Optional[AsyncSession] = None
    ) -> bool:
        """
        Check if user has permission to access/modify a specific object.
        Default returns True - ovveride for object-level checks.
        """
        return True
    
    def __call__(self, user: User = Depends(lambda: None)):
        """
        For use as a dependency in FastAPI endpoints.
        """
        if not self.has_permission(user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. {self.__class__.__name__} permission required."
            )
        return user



# BASIC PERMISSIONS


class IsAuthenticated(BasePermission):
    """Allows access only to authenticated users."""
    
    def has_permission(self, user: User) -> bool:
        return user is not None


class IsAdmin(BasePermission):
    """Allows access only to admin users."""
    
    def has_permission(self, user: User) -> bool:
        if user is None:
            return False
        is_allowed = user.role == "admin"
        if not is_allowed:
            logger.warning(f"Admin access denied for user {user.email} with role {user.role}")
        return is_allowed


class IsLister(BasePermission):
    """Allows access only to lister users."""
    
    def has_permission(self, user: User) -> bool:
        if user is None:
            return False
        is_allowed = user.role == "lister"
        if not is_allowed:
            logger.warning(f"Lister access denied for user {user.email} with role {user.role}")
        return is_allowed


class IsSeeker(BasePermission):
    """Allows access only to seeker users."""
    
    def has_permission(self, user: User) -> bool:
        if user is None:
            return False
        is_allowed = user.role == "seeker"
        if not is_allowed:
            logger.warning(f"Seeker access denied for user {user.email} with role {user.role}")
        return is_allowed


class IsAdminOrReadOnly(BasePermission):
    """
    Allows admin users to perform any action.
    Non-admin users can only perform safe (read) methods.
    """
    
    def has_permission(self, user: User, method: str = "GET") -> bool:
        if user is None:
            return method == "GET"
        
        if user.role == "admin":
            return True
        
        # Non-admin can only read
        return method == "GET"


class IsListerOrAdmin(BasePermission):
    """Allows access to lister and admin users."""
    
    def has_permission(self, user: User) -> bool:
        if user is None:
            return False
        is_allowed = user.role in ["lister", "admin"]
        if not is_allowed:
            logger.warning(f"Lister/Admin access denied for user {user.email} with role {user.role}")
        return is_allowed


class IsSeekerOrAdmin(BasePermission):
    """Allows access to seeker and admin users."""
    
    def has_permission(self, user: User) -> bool:
        if user is None:
            return False
        is_allowed = user.role in ["seeker", "admin"]
        if not is_allowed:
            logger.warning(f"Seeker/Admin access denied for user {user.email} with role {user.role}")
        return is_allowed



# OBJECT-LEVEL PERMISSIONS


class IsObjectOwner(BasePermission):
    """
    Check if user is the owner of an object.
    
    Args:
        model: SQLAlchemy model class
        object_id_param: Name of the parameter containing the object ID (e.g., "property_id")
        owner_field: Name of the field in the model that contains the owner ID (e.g., "lister_id")
    
    Usage:
        @router.put("/{property_id}")
        async def update_property(
            property_id: str,
            current_user: User = Depends(get_current_user),
            db: AsyncSession = Depends(get_db),
            _=Depends(IsObjectOwner(Property, "property_id", "lister_id"))
        ):
            ...
    """
    
    def __init__(self, model: Type, object_id_param: str, owner_field: str):
        self.model = model
        self.object_id_param = object_id_param
        self.owner_field = owner_field
    
    def has_permission(self, user: User) -> bool:
        # First check if user is authenticated
        return user is not None
    
    async def has_object_permission(
        self, user: User, obj_id: str, db: AsyncSession
    ) -> bool:
        """
        Check if user owns the object or is admin.
        """
        try:
            # Convert to UUID if needed
            obj_uuid = uuid.UUID(obj_id) if isinstance(obj_id, str) else obj_id
        except ValueError:
            logger.warning(f"Invalid object ID format: {obj_id}")
            return False
        
        # Admins can access anything
        if user.role == "admin":
            return True
        
        # Get the object from database
        result = await db.execute(
            select(self.model).where(self.model.id == obj_uuid)
        )
        obj = result.scalars().first()
        
        if not obj:
            logger.warning(f"Object not found: {obj_id}")
            return False
        
        # Check if user is the owner
        owner_id = getattr(obj, self.owner_field, None)
        is_owner = owner_id == user.id
        
        if not is_owner:
            logger.warning(f"User {user.email} is not owner of {self.model.__name__} {obj_id}")
        
        return is_owner


class IsObjectOwnerOrAdmin(BasePermission):
    """
    Check if user is the owner of an object or is an admin.
    Combines IsObjectOwner and IsAdmin for object-level access.
    """
    
    def __init__(self, model: Type, owner_field: str = "user_id"):
        self.model = model
        self.owner_field = owner_field
    
    def has_permission(self, user: User) -> bool:
        return user is not None
    
    async def has_object_permission(
        self, user: User, obj: Any
    ) -> bool:
        """Check if user is owner or admin."""
        if user.role == "admin":
            return True
        
        owner_id = getattr(obj, self.owner_field, None)
        return owner_id == user.id


class IsVerified(BasePermission):
    """Allows access only to verified users."""
    
    def has_permission(self, user: User) -> bool:
        if user is None:
            return False
        is_verified = user.is_verified
        if not is_verified:
            logger.warning(f"Unverified user {user.email} attempted access")
        return is_verified


class IsVerifiedAndAuthenticated(BasePermission):
    """Allows access only to authenticated and verified users."""
    
    def has_permission(self, user: User) -> bool:
        if user is None:
            return False
        is_allowed = user is not None and user.is_verified
        if not is_allowed:
            logger.warning(f"Access denied for user {user.email if user else 'unknown'}")
        return is_allowed



# PERMISSION DEPENDENCY FACTORIES


def get_permission_dependency(*permissions: BasePermission):
    """
    Create a dependency that checks multiple permissions.
    
    Usage:
        check_permissions = get_permission_dependency(IsAuthenticated(), IsLister())
        
        @router.post("/listing")
        async def create_listing(
            current_user: User = Depends(get_current_user),
            _=Depends(check_permissions)
        ):
            ...
    """
    async def check(user: User = Depends(lambda: None)) -> User:
        for permission in permissions:
            if not permission.has_permission(user):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. {permission.__class__.__name__} permission required."
                )
        return user
    
    return check



# SHORTHAND INSTANCES FOR COMMON USE CASES


# These can be used directly as dependencies
is_authenticated = Depends(IsAuthenticated())
is_admin = Depends(IsAdmin())
is_lister = Depends(IsLister())
is_seeker = Depends(IsSeeker())
is_lister_or_admin = Depends(IsListerOrAdmin())
is_seeker_or_admin = Depends(IsSeekerOrAdmin())
is_verified = Depends(IsVerified())
is_verified_and_authenticated = Depends(IsVerifiedAndAuthenticated())
