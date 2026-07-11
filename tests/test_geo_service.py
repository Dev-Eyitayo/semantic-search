"""
Unit tests for services.geo_service: distance math, bounding boxes,
distance scoring, location normalization, filter parsing, and the layered
location resolver (with Redis and DB faked).
"""

import json
import math

import pytest

from services import geo_service
from services.geo_service import (
    GeoPoint,
    GEOCODE_NEGATIVE_SENTINEL,
    bounding_box,
    distance_score,
    haversine_km,
    normalize_location,
    parse_geo_filters,
    resolve_location,
)

# Reference coordinates from seed data
YABA = (6.5244, 3.3792)
VICTORIA_ISLAND = (6.4281, 3.4192)
BODIJA_IBADAN = (7.4200, 3.9100)


class TestHaversine:
    def test_zero_distance_for_same_point(self):
        assert haversine_km(*YABA, *YABA) == 0.0

    def test_known_distance_yaba_to_victoria_island(self):
        # ~11.6 km straight line
        d = haversine_km(*YABA, *VICTORIA_ISLAND)
        assert 11.0 < d < 12.5

    def test_known_distance_lagos_to_ibadan(self):
        # ~115 km straight line
        d = haversine_km(*YABA, *BODIJA_IBADAN)
        assert 105.0 < d < 125.0

    def test_symmetry(self):
        assert haversine_km(*YABA, *VICTORIA_ISLAND) == pytest.approx(
            haversine_km(*VICTORIA_ISLAND, *YABA)
        )


class TestBoundingBox:
    def test_contains_points_at_radius_in_cardinal_directions(self):
        lat, lng, radius = 6.5, 3.4, 10.0
        lat_min, lat_max, lng_min, lng_max = bounding_box(lat, lng, radius)

        dlat = radius / 111.32
        dlng = radius / (111.32 * math.cos(math.radians(lat)))
        for plat, plng in [
            (lat + dlat * 0.999, lng),
            (lat - dlat * 0.999, lng),
            (lat, lng + dlng * 0.999),
            (lat, lng - dlng * 0.999),
        ]:
            assert lat_min <= plat <= lat_max
            assert lng_min <= plng <= lng_max
            assert haversine_km(lat, lng, plat, plng) <= radius * 1.01

    def test_box_over_selects_but_never_under_selects(self):
        # A point just outside the circle diagonally can still be in the box
        # (that's why we refine with haversine), but a point inside the
        # circle must always be inside the box.
        lat, lng, radius = 6.5, 3.4, 5.0
        lat_min, lat_max, lng_min, lng_max = bounding_box(lat, lng, radius)
        inside = (lat + 0.03, lng + 0.02)  # ~4 km away
        assert haversine_km(lat, lng, *inside) < radius
        assert lat_min <= inside[0] <= lat_max
        assert lng_min <= inside[1] <= lng_max

    def test_latitude_bounds_clamped(self):
        lat_min, lat_max, _, _ = bounding_box(89.9, 0.0, 100.0)
        assert lat_max == 90.0

    def test_invalid_coordinates_rejected(self):
        with pytest.raises(ValueError):
            bounding_box(91.0, 3.4, 5.0)
        with pytest.raises(ValueError):
            bounding_box(6.5, 181.0, 5.0)

    def test_invalid_radius_rejected(self):
        with pytest.raises(ValueError):
            bounding_box(6.5, 3.4, 0.0)
        with pytest.raises(ValueError):
            bounding_box(6.5, 3.4, 501.0)


class TestDistanceScore:
    def test_monotonically_decreasing(self):
        scores = [distance_score(d) for d in [0, 1, 5, 15, 50, 200]]
        assert scores == sorted(scores, reverse=True)

    def test_bounds(self):
        assert distance_score(0) == pytest.approx(0.95)
        assert 0.5 <= distance_score(10_000) <= 0.51

    def test_matches_legacy_bands(self):
        # Old implementation: 0.9 within 5km, 0.75 within 15km, 0.6 within 50km
        assert distance_score(4) > 0.8
        assert distance_score(14) > 0.65
        assert distance_score(45) > 0.5


class TestNormalizeLocation:
    def test_lowercases_and_trims(self):
        assert normalize_location("  Lekki Phase 1,   Lagos ") == "lekki phase 1, lagos"

    def test_expands_aliases(self):
        assert normalize_location("VI") == "victoria island"
        assert normalize_location("ph") == "port harcourt"

    def test_leaves_unknown_strings_alone(self):
        assert normalize_location("Bodija") == "bodija"


class TestParseGeoFilters:
    def test_no_geo_keys(self):
        assert parse_geo_filters({"price_max": 100}) == (None, None)

    def test_valid_center_and_radius(self):
        center, radius = parse_geo_filters({"lat": 6.5, "lng": 3.4, "radius_km": 10})
        assert center == (6.5, 3.4)
        assert radius == 10.0

    def test_lat_without_lng_rejected(self):
        with pytest.raises(ValueError, match="together"):
            parse_geo_filters({"lat": 6.5})

    def test_out_of_range_coordinates_rejected(self):
        with pytest.raises(ValueError):
            parse_geo_filters({"lat": 95.0, "lng": 3.4})

    def test_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="numbers"):
            parse_geo_filters({"lat": "abc", "lng": 3.4})

    def test_radius_without_center_or_location_rejected(self):
        with pytest.raises(ValueError, match="radius_km"):
            parse_geo_filters({"radius_km": 5})

    def test_radius_with_location_string_allowed(self):
        center, radius = parse_geo_filters({"radius_km": 5, "location": "yaba"})
        assert center is None
        assert radius == 5.0


class FakeRedis:
    def __init__(self, initial=None):
        self.store = dict(initial or {})
        self.writes = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        self.store[key] = value
        self.writes[key] = (ttl, value)


class FakeCentroidResult:
    def __init__(self, row):
        self._row = row

    def one(self):
        return self._row


class FakeDb:
    def __init__(self, centroid_row):
        self.centroid_row = centroid_row

    async def execute(self, query):
        return FakeCentroidResult(self.centroid_row)


class TestResolveLocation:
    async def test_resolves_from_listing_centroid_and_caches(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr(geo_service, "redis_client", fake_redis)
        db = FakeDb(centroid_row=(6.45, 3.40, 12))

        point = await resolve_location(db, "Yaba")

        assert point == GeoPoint(lat=6.45, lng=3.40, source="listings")
        cached = json.loads(fake_redis.store["geocode:yaba"])
        assert cached["lat"] == 6.45

    async def test_cache_hit_skips_db(self, monkeypatch):
        payload = json.dumps({"lat": 6.43, "lng": 3.42, "source": "listings"})
        fake_redis = FakeRedis({"geocode:victoria island": payload})
        monkeypatch.setattr(geo_service, "redis_client", fake_redis)

        class ExplodingDb:
            async def execute(self, query):
                raise AssertionError("DB should not be queried on cache hit")

        point = await resolve_location(ExplodingDb(), "VI")  # alias -> victoria island
        assert point.lat == 6.43
        assert point.source == "cache"

    async def test_unresolvable_location_negative_cached(self, monkeypatch):
        fake_redis = FakeRedis()
        monkeypatch.setattr(geo_service, "redis_client", fake_redis)
        db = FakeDb(centroid_row=(None, None, 0))

        point = await resolve_location(db, "nowhere-ville")

        assert point is None
        assert fake_redis.store["geocode:nowhere-ville"] == GEOCODE_NEGATIVE_SENTINEL

    async def test_negative_cache_short_circuits(self, monkeypatch):
        fake_redis = FakeRedis({"geocode:nowhere-ville": GEOCODE_NEGATIVE_SENTINEL})
        monkeypatch.setattr(geo_service, "redis_client", fake_redis)

        class ExplodingDb:
            async def execute(self, query):
                raise AssertionError("DB should not be queried on negative cache hit")

        assert await resolve_location(ExplodingDb(), "nowhere-ville") is None

    async def test_blank_input_returns_none(self, monkeypatch):
        monkeypatch.setattr(geo_service, "redis_client", FakeRedis())
        assert await resolve_location(FakeDb((None, None, 0)), "  ") is None

    async def test_redis_failure_degrades_gracefully(self, monkeypatch):
        class BrokenRedis:
            async def get(self, key):
                raise ConnectionError("redis down")

            async def setex(self, key, ttl, value):
                raise ConnectionError("redis down")

        monkeypatch.setattr(geo_service, "redis_client", BrokenRedis())
        db = FakeDb(centroid_row=(6.45, 3.40, 3))

        point = await resolve_location(db, "yaba")
        assert point is not None
        assert point.source == "listings"
