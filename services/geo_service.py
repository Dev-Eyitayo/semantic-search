"""
Geospatial utilities and location resolution for search.

Distance math uses the haversine formula (exact for a spherical Earth,
error < 0.5% vs the ellipsoid) instead of degree approximations.
Radius queries are served with a two-step strategy that works on any
Postgres (no PostGIS required):

    1. SQL bounding-box prefilter on the indexed (latitude, longitude) pair
    2. exact haversine refinement in Python on the candidate set

Free-text locations ("lekki", "challenge") are resolved to coordinates by a
layered geocoder:

    Redis cache -> centroid of existing approved listings -> Nominatim API

The Nominatim fallback is optional (GEOCODER_PROVIDER setting) and results
are cached aggressively, including negative results, so the external
service is only hit on genuinely new location strings.
"""

import json
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from db.models.property import Property
from core.enums import PropertyStatus
from services.redis_service import redis_client

EARTH_RADIUS_KM = 6371.0088
KM_PER_DEGREE_LAT = 111.32

# Radius search guardrails
MIN_RADIUS_KM = 0.1
MAX_RADIUS_KM = 500.0

GEOCODE_CACHE_PREFIX = "geocode:"
GEOCODE_NEGATIVE_TTL_SECONDS = 24 * 3600  # retry unknown locations daily
GEOCODE_NEGATIVE_SENTINEL = "__miss__"

# Common shorthands users type that don't appear verbatim in listing locations.
# Keys must already be normalized (lowercase, single spaces).
LOCATION_ALIASES = {
    "vi": "victoria island",
    "v.i": "victoria island",
    "v.i.": "victoria island",
    "ph": "port harcourt",
    "portharcourt": "port harcourt",
    "abj": "abuja",
    "gra ibadan": "bodija",
    "leadcity": "lead city",
    "leadcity": "leadcity University",
}


@dataclass(frozen=True)
class GeoPoint:
    """A resolved coordinate pair with the source it came from."""
    lat: float
    lng: float
    source: str = "unknown"  # cache | listings | nominatim


def validate_coordinates(lat: float, lng: float) -> None:
    """Raise ValueError if a coordinate pair is out of range."""
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude must be between -90 and 90, got {lat}")
    if not (-180.0 <= lng <= 180.0):
        raise ValueError(f"Longitude must be between -180 and 180, got {lng}")


def validate_radius(radius_km: float) -> None:
    """Raise ValueError if a search radius is outside the supported range."""
    if not (MIN_RADIUS_KM <= radius_km <= MAX_RADIUS_KM):
        raise ValueError(
            f"radius_km must be between {MIN_RADIUS_KM} and {MAX_RADIUS_KM}, got {radius_km}"
        )


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Great-circle distance between two coordinate pairs in kilometres.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bounding_box(lat: float, lng: float, radius_km: float) -> Tuple[float, float, float, float]:
    """
    Bounding box (lat_min, lat_max, lng_min, lng_max) that fully contains the
    circle of `radius_km` around a point. Slightly over-selects by design;
    callers must refine with haversine_km. Longitude bounds are clamped to
    [-180, 180] rather than wrapped (fine for our market, far from the
    antimeridian).
    """
    validate_coordinates(lat, lng)
    validate_radius(radius_km)

    dlat = radius_km / KM_PER_DEGREE_LAT
    # Longitude degrees shrink with latitude; guard cos() near the poles
    dlng = radius_km / (KM_PER_DEGREE_LAT * max(math.cos(math.radians(lat)), 0.01))

    return (
        max(lat - dlat, -90.0),
        min(lat + dlat, 90.0),
        max(lng - dlng, -180.0),
        min(lng + dlng, 180.0),
    )


def distance_score(distance_km: float) -> float:
    """
    Map a distance to a 0.5-0.95 relevance score with smooth exponential
    decay, matching the bands the ranking formula already used
    (~0.9 within 5km, ~0.75 within 15km, ~0.6 within 50km).
    """
    return 0.5 + 0.45 * math.exp(-max(distance_km, 0.0) / 15.0)


def normalize_location(name: str) -> str:
    """
    Normalize a user-provided location string for cache keys, alias lookup,
    and ILIKE matching: lowercase, trimmed, single-spaced, common
    abbreviations expanded.
    """
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return LOCATION_ALIASES.get(normalized, normalized)


def parse_geo_filters(filters: dict) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    """
    Extract and validate geo keys (lat, lng, radius_km) from a search filters
    dict. Returns (center, radius_km); either may be None.

    Raises ValueError with a user-presentable message on invalid input.
    """
    lat = filters.get("lat")
    lng = filters.get("lng")
    radius_km = filters.get("radius_km")

    center = None
    if lat is not None or lng is not None:
        if lat is None or lng is None:
            raise ValueError("Both 'lat' and 'lng' must be provided together")
        try:
            lat, lng = float(lat), float(lng)
        except (TypeError, ValueError):
            raise ValueError("'lat' and 'lng' must be numbers")
        validate_coordinates(lat, lng)
        center = (lat, lng)

    if radius_km is not None:
        try:
            radius_km = float(radius_km)
        except (TypeError, ValueError):
            raise ValueError("'radius_km' must be a number")
        validate_radius(radius_km)
        if center is None and not filters.get("location"):
            raise ValueError(
                "'radius_km' requires either 'lat'/'lng' or a 'location' to search around"
            )

    return center, radius_km


async def _geocode_from_listings(db: AsyncSession, normalized_name: str) -> Optional[GeoPoint]:
    """
    Resolve a location from the platform's own data: the centroid of approved
    listings whose location matches. Self-consistent with what users can
    actually find, and free.
    """
    result = await db.execute(
        select(
            func.avg(Property.latitude),
            func.avg(Property.longitude),
            func.count(Property.id),
        ).where(
            Property.status == PropertyStatus.APPROVED,
            Property.location.ilike(f"%{normalized_name}%"),
            Property.latitude.isnot(None),
            Property.longitude.isnot(None),
        )
    )
    avg_lat, avg_lng, count = result.one()

    if count and avg_lat is not None and avg_lng is not None:
        return GeoPoint(lat=float(avg_lat), lng=float(avg_lng), source="listings")
    return None


async def _geocode_from_nominatim(normalized_name: str) -> Optional[GeoPoint]:
    """
    Resolve a location via the Nominatim (OpenStreetMap) API. Only called on
    cache and listing-centroid misses; results are cached by the caller.
    """
    params = {
        "q": normalized_name,
        "format": "json",
        "limit": 1,
        "countrycodes": settings.GEOCODER_COUNTRY_CODES,
    }
    # Nominatim usage policy requires an identifying User-Agent
    headers = {"User-Agent": f"{settings.PROJECT_NAME}/{settings.VERSION} ({settings.FRONTEND_URL})"}

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{settings.NOMINATIM_BASE_URL}/search", params=params, headers=headers
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.warning(f"Nominatim geocoding failed for '{normalized_name}': {e}")
        return None

    if not data:
        return None

    try:
        return GeoPoint(lat=float(data[0]["lat"]), lng=float(data[0]["lon"]), source="nominatim")
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"Unexpected Nominatim response for '{normalized_name}': {e}")
        return None


async def resolve_location(db: AsyncSession, name: str) -> Optional[GeoPoint]:
    """
    Resolve a free-text location name to coordinates.

    Resolution order: Redis cache -> centroid of approved listings ->
    Nominatim API (if enabled). Failures at any layer degrade gracefully;
    negative results are cached to avoid re-querying unknown strings.

    Returns None when the location cannot be resolved.
    """
    if not name or len(name.strip()) < 2:
        return None

    normalized = normalize_location(name)
    cache_key = f"{GEOCODE_CACHE_PREFIX}{normalized}"

    try:
        cached = await redis_client.get(cache_key)
        if cached == GEOCODE_NEGATIVE_SENTINEL:
            return None
        if cached:
            payload = json.loads(cached)
            return GeoPoint(lat=payload["lat"], lng=payload["lng"], source="cache")
    except Exception as e:
        logger.warning(f"Geocode cache read failed: {e}")

    point = await _geocode_from_listings(db, normalized)

    if point is None and settings.GEOCODER_PROVIDER == "nominatim":
        point = await _geocode_from_nominatim(normalized)

    try:
        if point is not None:
            await redis_client.setex(
                cache_key,
                settings.GEOCODE_CACHE_TTL_SECONDS,
                json.dumps({"lat": point.lat, "lng": point.lng, "source": point.source}),
            )
        else:
            await redis_client.setex(cache_key, GEOCODE_NEGATIVE_TTL_SECONDS, GEOCODE_NEGATIVE_SENTINEL)
    except Exception as e:
        logger.warning(f"Geocode cache write failed: {e}")

    if point:
        logger.debug(f"Resolved location '{name}' -> ({point.lat:.4f}, {point.lng:.4f}) via {point.source}")
    else:
        logger.debug(f"Could not resolve location '{name}'")

    return point
