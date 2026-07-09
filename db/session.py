from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from core.config import settings
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse


def _prepare_async_url_and_connect_args(url: str):
    """Remove unsupported query params (like sslmode and channel_binding) for asyncpg and
    return a tuple of (clean_url, connect_args).
    """
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    connect_args = {}

    # If sslmode is present (added for libpq), asyncpg's connect() doesn't
    # accept sslmode keyword — use connect_args instead and remove sslmode.
    if "sslmode" in query:
        # Use basic ssl flag; for advanced needs create an SSLContext.
        connect_args["ssl"] = True
        query.pop("sslmode", None)

    if "channel_binding" in query:
        query.pop("channel_binding", None)

    clean_query = urlencode(query)
    clean_url = urlunparse(parsed._replace(query=clean_query))
    return clean_url, connect_args


clean_async_url, async_connect_args = _prepare_async_url_and_connect_args(settings.DATABASE_ASYNC_URL)

engine = create_async_engine(
    clean_async_url,
    echo=True,
    future=True,
    connect_args=async_connect_args if async_connect_args else None,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session