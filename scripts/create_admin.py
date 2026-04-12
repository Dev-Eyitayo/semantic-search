"""
Create Admin Account Script

Creates a new admin account interactively.
Run this to create an admin user for the application.

Usage:
    python scripts/create_admin.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from db.models.user import User
from core.config import settings
from core.security import get_password_hash
from core.enums import UserRole
from uuid import uuid4
from loguru import logger
import re

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_password(password: str) -> tuple[bool, str]:
    """Validate password strength"""
    errors = []
    
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search(r'[0-9]', password):
        errors.append("Password must contain at least one digit")
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    
    if errors:
        return False, "\n  ".join(errors)
    return True, ""


def validate_phone(phone: str) -> bool:
    """Validate phone number - E.164 format"""
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone.replace(" ", "").replace("-", "")) is not None


async def create_admin():
    """Create admin account interactively"""
    
    print("\n" + "=" * 70)
    print("🔑 CREATE ADMIN ACCOUNT")
    print("=" * 70 + "\n")
    
    # Create async engine
    engine = create_async_engine(settings.DATABASE_ASYNC_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    try:
        # Input email
        while True:
            email = input("📧 Admin Email: ").strip().lower()
            
            if not validate_email(email):
                logger.error("❌ Invalid email format")
                continue
            
            # Check if email already exists
            async with async_session() as session:
                result = await session.execute(
                    select(User).where(User.email == email)
                )
                existing = result.scalars().first()
                
                if existing:
                    logger.error(f"❌ Email already exists: {email}")
                    continue
            
            break
        
        # Input first name
        while True:
            first_name = input("👤 First Name: ").strip()
            if not first_name or len(first_name) < 2:
                logger.error("❌ First name must be at least 2 characters")
                continue
            break
        
        # Input last name
        while True:
            last_name = input("👤 Last Name: ").strip()
            if not last_name or len(last_name) < 2:
                logger.error("❌ Last name must be at least 2 characters")
                continue
            break
        
        # Input phone
        while True:
            phone = input("☎️  Phone Number (E.164 format, e.g., +1234567890): ").strip()
            if not validate_phone(phone):
                logger.error("❌ Invalid phone number format")
                continue
            break
        
        # Input password
        while True:
            password = input("🔐 Password: ").strip()
            
            is_valid, error_msg = validate_password(password)
            if not is_valid:
                logger.error(f"❌ Password validation failed:\n  {error_msg}")
                continue
            
            password_confirm = input("🔐 Confirm Password: ").strip()
            if password != password_confirm:
                logger.error("❌ Passwords do not match")
                continue
            
            break
        
        # Summary
        print("\n" + "=" * 70)
        print("📋 ADMIN ACCOUNT SUMMARY")
        print("=" * 70)
        print(f"Email:      {email}")
        print(f"Name:       {first_name} {last_name}")
        print(f"Phone:      {phone}")
        print(f"Role:       ADMIN")
        print(f"Verified:   Yes")
        print("=" * 70)
        
        # Confirmation
        confirm = input("\n✅ Create this admin account? (yes/no): ").strip().lower()
        if confirm not in ["yes", "y"]:
            logger.warning("⚠️  Account creation cancelled")
            return
        
        # Create admin account
        async with async_session() as session:
            admin_user = User(
                id=uuid4(),
                email=email,
                first_name=first_name,
                last_name=last_name,
                phone=phone,
                password_hash=get_password_hash(password),
                role=UserRole.ADMIN,
                is_verified=True
            )
            
            session.add(admin_user)
            await session.commit()
            
            logger.success(f"✅ Admin account created successfully!")
            print("\n" + "=" * 70)
            print("🎉 Admin Account Details:")
            print("=" * 70)
            print(f"Email:    {email}")
            print(f"Password: {password}")
            print("=" * 70)
            print("\n⚠️  Keep these credentials safe!")
            print("   You can now log in and create other accounts.\n")
    
    except Exception as e:
        logger.error(f"❌ Error creating admin account: {str(e)}")
        return
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_admin())
