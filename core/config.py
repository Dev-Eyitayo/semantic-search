import os
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, computed_field
from pydantic_core import MultiHostUrl
from typing import Optional

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_ignore_empty=True, 
        extra="ignore"
    )

    PROD: bool = False
    INCLUDE_TOKENS_IN_JSON: bool = True
    UNSPLASH_ACCESS_KEY: str

    POSTGRES_USER: Optional[str] = None
    POSTGRES_PASSWORD: Optional[str] = None
    POSTGRES_DB: Optional[str] = None
    POSTGRES_HOST: Optional[str] = None
    POSTGRES_PORT: int = 5432

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 5
    REFRESH_TOKEN_EXPIRE_HOURS: int = 24


    PROJECT_NAME: str = "Sheltly"
    VERSION: str = "1.0.0"

    REDIS_URL: str
    
    MAIL_USERNAME: Optional[str] = None
    MAIL_PASSWORD: Optional[str] = None
    MAIL_FROM: Optional[str] = None
    MAIL_PORT: int = 587
    MAIL_SERVER: Optional[str] = None
    

    CLOUDINARY_CLOUD_NAME: str = "your-cloud-name"
    CLOUDINARY_API_KEY: str = "your-api-key"
    CLOUDINARY_API_SECRET: str = "your-api-secret"

    FRONTEND_URL: str = "http://localhost:3000"

    EMBEDDING_BACKEND: str = "local"
    HF_API_TOKEN: Optional[str] = None
    HF_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Geocoding: 'nominatim' enables the OpenStreetMap fallback for location
    # strings not resolvable from our own listings; 'none' disables it.
    GEOCODER_PROVIDER: str = "nominatim"
    NOMINATIM_BASE_URL: str = "https://nominatim.openstreetmap.org"
    GEOCODER_COUNTRY_CODES: str = "ng"
    GEOCODE_CACHE_TTL_SECONDS: int = 30 * 24 * 3600

    def _build_database_url(self, driver: str) -> str:
        if not all([self.POSTGRES_USER, self.POSTGRES_PASSWORD, self.POSTGRES_DB, self.POSTGRES_HOST]):
            raise ValueError("POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, and POSTGRES_HOST must be set when DATABASE_* URLs are not provided")

        url = f"postgresql+{driver}://{quote(self.POSTGRES_USER)}:{quote(self.POSTGRES_PASSWORD)}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        return self._ensure_ssl_mode(url)

    def _ensure_ssl_mode(self, url: str) -> str:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if hostname.endswith("supabase.co") or "supabase" in hostname:
            query = dict(parse_qsl(parsed.query, keep_blank_values=True))
            if "sslmode" not in query:
                query["sslmode"] = "require"
                return urlunparse(parsed._replace(query=urlencode(query)))
        return url

    @property
    def DATABASE_ASYNC_URL(self) -> str:
        # Prefer explicit DATABASE_ASYNC_URL; if DATABASE_URL is provided, derive asyncpg URL from it.
        explicit_async = os.getenv("DATABASE_ASYNC_URL")
        explicit_generic = os.getenv("DATABASE_URL")
        if explicit_async:
            return self._ensure_ssl_mode(explicit_async)
        if explicit_generic:
            # Convert generic URL if it uses postgresql:// to postgresql+asyncpg://
            parsed = urlparse(explicit_generic)
            scheme = parsed.scheme
            rest = explicit_generic[len(scheme) + 3 :]
            if scheme.startswith("postgresql"):
                return self._ensure_ssl_mode("postgresql+asyncpg://" + rest)
            return self._ensure_ssl_mode(explicit_generic)
        return self._build_database_url("asyncpg")

    @property
    def DATABASE_SYNC_URL(self) -> str:
        # Prefer explicit DATABASE_SYNC_URL; if DATABASE_URL is provided, derive sync (psycopg2) URL from it.
        explicit_sync = os.getenv("DATABASE_SYNC_URL")
        explicit_generic = os.getenv("DATABASE_URL")
        if explicit_sync:
            return self._ensure_ssl_mode(explicit_sync)
        if explicit_generic:
            parsed = urlparse(explicit_generic)
            scheme = parsed.scheme
            rest = explicit_generic[len(scheme) + 3 :]
            if scheme.startswith("postgresql"):
                return self._ensure_ssl_mode("postgresql+psycopg2://" + rest)
            return self._ensure_ssl_mode(explicit_generic)
        return self._build_database_url("psycopg2")


settings = Settings()