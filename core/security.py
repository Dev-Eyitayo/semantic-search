import uuid
import bcrypt # Add this
from datetime import datetime, timedelta, timezone
from typing import Any, Union
from jose import jwt
from core.config import settings


def create_access_token(subject: Union[str, Any], role: str) -> str:
    """Generates a 24h Access Token with JTI for Redis blacklisting."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {
        "exp": expire, 
        "sub": str(subject), 
        "role": role,
        "jti": str(uuid.uuid4()), 
        "type": "access"
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(subject: Union[str, Any]) -> str:
    """Generates a 7-day Refresh Token for rotation."""
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.REFRESH_TOKEN_EXPIRE_HOURS)
    to_encode = {
        "exp": expire, 
        "sub": str(subject), 
        "jti": str(uuid.uuid4()), 
        "type": "refresh"
    }
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def get_password_hash(password: str) -> str:
    """Hashes a password using bcrypt (standard cost factor is 12)."""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed_password = bcrypt.hashpw(pwd_bytes, salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed one."""
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'), 
            hashed_password.encode('utf-8')
        )
    except Exception:
        return False