import secrets
import re
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, select, update
from datetime import datetime, timezone, timedelta
from jose import jwt, JWTError

from core.security import (
    get_password_hash, verify_password, 
    create_access_token, create_refresh_token,
)
from core.config import settings
from db.session import get_db
from db.models.user import User, RefreshToken
from schemas.auth import (
    UserCreate, VerifyEmailRequest, UserResponse, UserLogin, Token, TokenRefreshRequest,
    PasswordResetRequest, AccessTokenResponse, PasswordResetConfirm
)
from schemas.base import StandardResponse
from services.redis_service import (
    store_otp, verify_otp_code, blacklist_token
)
from services.mail_service import send_verification_email, send_password_reset_otp
from loguru import logger   

router = APIRouter()

@router.post("/register", response_model=StandardResponse[UserResponse], status_code=201)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Registers a new user and triggers a background OTP email via Celery.
    """
    email_lower = user_in.email.lower()
    logger.info(f"Attempting to register user with email: {email_lower}")
    
    result = await db.execute(
        select(User).where(func.lower(User.email) == email_lower)
    )
    if result.scalars().first():
        raise HTTPException(status_code=409, detail="Email already registered")
    
    new_user = User(
        email=email_lower,
        password_hash=get_password_hash(user_in.password),
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        phone=user_in.phone,
        role=user_in.role
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    otp_code = "".join([str(secrets.randbelow(10)) for _ in range(6)])
    await store_otp(email_lower, otp_code, expire_seconds=600)
    
    logger.success(f"User {email_lower} created successfully. Sending OTP.")
    send_verification_email.delay(new_user.email, new_user.first_name, otp_code)
    
    return StandardResponse(
        message="Registration successful. Please check your email for the OTP.",
        data=UserResponse(
            user_id=new_user.id,
            role=new_user.role
        )
    )


@router.post("/verify-email", response_model=StandardResponse[None])
async def verify_email(
    payload: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Validates OTP against Redis and activates the account.
    """
    email_lower = payload.email.lower()
    is_valid = await verify_otp_code(email_lower, payload.otp)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")
    
    result = await db.execute(
        select(User).where(func.lower(User.email) == email_lower)
    )
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    user.is_verified = True
    await db.commit()
    
    return StandardResponse(message="Email verified successfully. You can now login.")


@router.post("/login", response_model=StandardResponse[Token])
async def login(user_in: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    Authenticates user and issues 24h Access + 7d Refresh tokens.
    """
    email_lower = user_in.email.lower()
    logger.info(f"Login attempt for email: {email_lower}")

    result = await db.execute(
        select(User).where(func.lower(User.email) == email_lower)
    )
    user = result.scalars().first()
    
    if not user or not verify_password(user_in.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="Account not verified")

    access_token = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id)
    
    db_refresh = RefreshToken(
        user_id=user.id,
        token=refresh_token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7)
    )
    db.add(db_refresh)
    await db.commit()
    
    return StandardResponse(
        message="Login successful",
        data=Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            role=user.role,
            user_id=user.id
        )
    )

@router.post("/logout", response_model=StandardResponse[None])
async def logout(token_data: TokenRefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Invalidates session by blacklisting the JTI in Redis.
    """
    try:
        payload = jwt.decode(
            token_data.refresh_token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        jti = payload.get("jti")
        exp = payload.get("exp")
        ttl = int(exp - datetime.now(timezone.utc).timestamp())
        
        if ttl > 0:
            await blacklist_token(jti, ttl)
            
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token == token_data.refresh_token)
        )
        db_token = result.scalars().first()
        if db_token:
            db_token.is_revoked = True
            await db.commit()
            
        return StandardResponse(message="Logged out successfully")
            
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    

@router.post("/refresh", response_model=StandardResponse[AccessTokenResponse], status_code=200)
async def refresh_token(token_data: TokenRefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Exchange a valid refresh token for a new access token without re-authentication.
    Implements token rotation - invalidates old refresh token and issues a new one.
    """
    try:
        payload = jwt.decode(
            token_data.refresh_token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        user_id = payload.get("sub")
        jti = payload.get("jti")
        token_type = payload.get("type")
        
        if token_type != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Check if refresh token is revoked in DB
        result = await db.execute(
            select(RefreshToken).where(
                RefreshToken.token == token_data.refresh_token,
                RefreshToken.is_revoked == False
            )
        )
        db_token = result.scalars().first()
        
        if not db_token or db_token.is_revoked:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
        
        # Verify token hasn't expired
        if db_token.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status_code=401, detail="Refresh token expired")
        
        # Get user to check they still exist
        user = await db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Revoke old refresh token (token rotation)
        db_token.is_revoked = True
        
        # Create new tokens
        new_access_token = create_access_token(user.id, user.role)
        new_refresh_token = create_refresh_token(user.id)
        
        # Store new refresh token
        new_db_token = RefreshToken(
            user_id=user.id,
            token=new_refresh_token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        db.add(new_db_token)
        await db.commit()
        
        return StandardResponse(
            message="Token refreshed successfully",
            data=AccessTokenResponse(
                access_token=new_access_token,
                token_type="bearer",
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
        )
        
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")


@router.post("/forgot-password", response_model=StandardResponse[None], status_code=200)
async def forgot_password(request: PasswordResetRequest, db: AsyncSession = Depends(get_db)):
    """
    Send an OTP to the user's email for password reset.
    Always returns 200 to prevent email enumeration attacks.
    """
    email_lower = request.email.lower()
    logger.info(f"Password reset requested for email: {email_lower}")
    
    result = await db.execute(
        select(User).where(func.lower(User.email) == email_lower)
    )
    user = result.scalars().first()
    
    if user:
        # Generate 6-digit OTP
        otp_code = "".join([str(secrets.randbelow(10)) for _ in range(6)])
        
        # Store OTP in Redis with 15 minute expiry
        await store_otp(f"reset_otp:{email_lower}", otp_code, expire_seconds=900)
        
        logger.success(f"Password reset OTP generated for email: {email_lower}")
        send_password_reset_otp.delay(user.email, user.first_name, otp_code)
    else:
        logger.warning(f"Password reset requested for non-existent email: {email_lower}")
    
    return StandardResponse(
        message="If email exists, OTP has been sent to your email"
    )



@router.post("/reset-password", response_model=StandardResponse[None])
async def reset_password(
    payload: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """
    Finalize password reset using OTP stored in Redis.
    Validation for password strength is handled by the Pydantic schema.
    """
    email_lower = payload.email.lower()
    logger.info(f"Processing password reset for: {email_lower}")

    is_valid = await verify_otp_code(f"reset_otp:{email_lower}", payload.otp)
    
    if not is_valid:
        logger.warning(f"Invalid or expired reset OTP for: {email_lower}")
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

    result = await db.execute(
        select(User).where(func.lower(User.email) == email_lower)
    )
    user = result.scalars().first()
    
    if not user:
        logger.warning(f"User search failed during reset for: {email_lower}")
        raise HTTPException(status_code=404, detail="User not found")
    
    user.password_hash = get_password_hash(payload.new_password)


    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user.id)
        .values(is_revoked=True)
    )

    await db.commit()
    
    logger.success(f"Password reset successful for: {email_lower}")

    return StandardResponse(
        message="Password reset successful. You can now log in.",
        data=None
    )