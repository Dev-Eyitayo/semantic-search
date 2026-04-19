import redis.asyncio as redis
from core.config import settings

use_ssl = settings.REDIS_URL.startswith("rediss://")

# 2. Build your connection arguments dynamically
redis_kwargs = {
    "decode_responses": True,
}
if use_ssl:
    redis_kwargs["ssl_cert_reqs"] = None 

redis_client = redis.from_url(settings.REDIS_URL, **redis_kwargs)


async def blacklist_token(jti: str, ttl: int):
    """
    Blacklists a JWT ID (jti) for the remainder of its lifespan.
    High-performance check used during the Logout flow.
    """
    await redis_client.setex(f"blacklist:{jti}", ttl, "true")

async def is_token_blacklisted(jti: str) -> bool:
    """Checks Redis to see if the token was manually invalidated."""
    return await redis_client.exists(f"blacklist:{jti}") > 0


async def store_otp(email: str, otp: str, expire_seconds: int = 600):
    """Stores a 6-digit OTP in Redis keyed by email[cite: 129]."""
    await redis_client.setex(f"otp:{email}", expire_seconds, otp)

async def verify_otp_code(email: str, input_otp: str) -> bool:
    """Checks if the provided OTP matches the one in Redis."""
    stored_otp = await redis_client.get(f"otp:{email}")
    return stored_otp == input_otp