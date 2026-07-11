from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from core.config import settings

import os


redis_url = settings.REDIS_URL

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url,
    strategy="fixed-window",
    # Fail open if Redis is unavailable: an outage must not lock users out
    swallow_errors=True
)