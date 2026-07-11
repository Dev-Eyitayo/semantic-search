"""
Integration tests for the geo-aware search endpoints, with the database
dependency overridden by an in-memory fake and the embedding service stubbed.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v1 import search as search_module
from core.enums import PriceType, PropertyStatus, PropertyType
from db.models.property import Property
from db.session import get_db
from services import geo_service

YABA = (6.5244, 3.3792)
VICTORIA_ISLAND = (6.4281, 3.4192)
BODIJA_IBADAN = (7.4200, 3.9100)


def make_property(title, location, lat, lng, **overrides) -> Property:
    defaults = dict(
        id=uuid.uuid4(),
        lister_id=uuid.uuid4(),
        title=title,
        description=f"{title} with modern amenities",
        price=1_000_000.0,
        price_type=PriceType.RENT,
        location=location,
        latitude=lat,
        longitude=lng,
        bedrooms=2,
        bathrooms=2,
        property_type=PropertyType.APARTMENT,
        amenities=["parking"],
        thumbnail=None,
        status=PropertyStatus.APPROVED,
        created_at=datetime.now(timezone.utc) - timedelta(days=3),
    )
    defaults.update(overrides)
    return Property(**defaults)


class FakeResult:
    """Mimics the slice of SQLAlchemy's Result API the endpoints use."""

    def __init__(self, items=None, row=None):
        self._items = items or []
        self._row = row

    def scalars(self):
        return self

    def all(self):
        return self._items

    def first(self):
        return self._items[0] if self._items else None

    def one(self):
        return self._row


class FakeSession:
    """Returns queued FakeResults in the order queries are executed."""

    def __init__(self, results):
        self.results = list(results)

    async def execute(self, query):
        return self.results.pop(0)

    def add(self, obj):
        pass

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(search_module.router, prefix="/api/v1/search")
    return app


def client_with_db(app, results) -> TestClient:
    session = FakeSession(results)

    async def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


class FakeRedis:
    async def get(self, key):
        return None

    async def setex(self, key, ttl, value):
        pass


@pytest.fixture(autouse=True)
def fake_geo_redis(monkeypatch):
    monkeypatch.setattr(geo_service, "redis_client", FakeRedis())


class TestNearbyEndpoint:
    def test_returns_properties_sorted_by_distance_within_radius(self, app):
        near = make_property("Flat near Yaba", "Yaba, Lagos", YABA[0] + 0.005, YABA[1] + 0.005)
        nearer = make_property("Flat in Yaba", "Yaba, Lagos", YABA[0] + 0.001, YABA[1])
        # Inside the bounding box corner but outside the 5km circle
        corner = make_property("Corner flat", "Somewhere, Lagos", YABA[0] + 0.044, YABA[1] + 0.044)

        client = client_with_db(app, [FakeResult(items=[near, nearer, corner])])
        response = client.get(
            "/api/v1/search/nearby",
            params={"lat": YABA[0], "lng": YABA[1], "radius_km": 5.0},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        titles = [r["title"] for r in data["results"]]
        assert titles == ["Flat in Yaba", "Flat near Yaba"]
        assert data["total_results"] == 2
        assert data["results"][0]["distance_km"] <= data["results"][1]["distance_km"]
        assert all(r["distance_km"] <= 5.0 for r in data["results"])

    def test_resolves_location_string_to_center(self, app):
        prop = make_property("Yaba studio", "Yaba, Lagos", *YABA)
        # First execute: geocode centroid query; second: property fetch
        client = client_with_db(app, [
            FakeResult(row=(YABA[0], YABA[1], 10)),
            FakeResult(items=[prop]),
        ])

        response = client.get(
            "/api/v1/search/nearby",
            params={"location": "yaba", "radius_km": 5.0},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["resolved_from"] == "yaba"
        assert data["center_lat"] == pytest.approx(YABA[0])
        assert data["total_results"] == 1

    def test_lat_without_lng_rejected(self, app):
        client = client_with_db(app, [])
        response = client.get("/api/v1/search/nearby", params={"lat": 6.5})
        assert response.status_code == 400

    def test_missing_center_rejected(self, app):
        client = client_with_db(app, [])
        response = client.get("/api/v1/search/nearby", params={"radius_km": 5})
        assert response.status_code == 400

    def test_unresolvable_location_returns_404(self, app):
        client = client_with_db(app, [FakeResult(row=(None, None, 0))])
        response = client.get("/api/v1/search/nearby", params={"location": "atlantis"})
        assert response.status_code == 404

    def test_out_of_range_latitude_rejected(self, app):
        client = client_with_db(app, [])
        response = client.get("/api/v1/search/nearby", params={"lat": 95, "lng": 3.4})
        assert response.status_code == 422


class TestSemanticSearchGeoFilters:
    def stub_similarity(self, monkeypatch):
        def fake_batch_similarity_search(query, candidates, top_k=None, normalize=True):
            return [(idx, text, 0.8) for idx, text in enumerate(candidates)]

        monkeypatch.setattr(search_module, "batch_similarity_search", fake_batch_similarity_search)

    def test_radius_filter_excludes_far_properties_and_reports_distance(self, app, monkeypatch):
        self.stub_similarity(monkeypatch)
        near = make_property("Yaba apartment", "Yaba, Lagos", YABA[0] + 0.005, YABA[1])
        # In the bbox corner, outside the exact radius: must be refined away
        corner = make_property("Corner apartment", "Edge, Lagos", YABA[0] + 0.044, YABA[1] + 0.044)

        client = client_with_db(app, [
            FakeResult(items=[]),               # RankingConfig -> defaults
            FakeResult(items=[near, corner]),   # property fetch (bbox already applied in SQL)
        ])
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "apartment with parking",
                "filters": {"lat": YABA[0], "lng": YABA[1], "radius_km": 5.0},
                "explain": True,
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_results"] == 1
        result = data["results"][0]
        assert result["title"] == "Yaba apartment"
        assert result["distance_km"] is not None
        assert result["distance_km"] < 5.0

    def test_invalid_geo_filters_return_400(self, app, monkeypatch):
        self.stub_similarity(monkeypatch)
        client = client_with_db(app, [])
        response = client.post(
            "/api/v1/search/semantic",
            json={"query": "apartment in lagos", "filters": {"lat": 6.5}},
        )
        assert response.status_code == 400
        # The test app has no custom exception handlers, so the body uses
        # FastAPI's default 'detail' key (main.py rewraps this as 'message')
        assert "lng" in response.json()["detail"]

    def test_radius_with_unresolvable_location_returns_400(self, app, monkeypatch):
        self.stub_similarity(monkeypatch)
        client = client_with_db(app, [
            FakeResult(row=(None, None, 0)),  # geocode centroid miss
        ])
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "apartment with parking",
                "filters": {"location": "atlantis", "radius_km": 5},
            },
        )
        assert response.status_code == 400

    def test_zero_results_inside_radius_returns_empty_envelope(self, app, monkeypatch):
        self.stub_similarity(monkeypatch)
        corner = make_property("Corner apartment", "Edge, Lagos", YABA[0] + 0.044, YABA[1] + 0.044)
        client = client_with_db(app, [
            FakeResult(items=[]),             # RankingConfig -> defaults
            FakeResult(items=[corner]),       # only a corner property, refined away
        ])
        response = client.post(
            "/api/v1/search/semantic",
            json={
                "query": "apartment with parking",
                "filters": {"lat": YABA[0], "lng": YABA[1], "radius_km": 5.0},
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_results"] == 0
        assert data["results"] == []

    def test_semantic_search_returns_clean_503_when_embedding_backend_fails(self, app, monkeypatch):
        def fail_similarity(query, candidates, top_k=None, normalize=True):
            raise RuntimeError("embedding backend unavailable")

        monkeypatch.setattr(search_module, "batch_similarity_search", fail_similarity)
        client = client_with_db(app, [
            FakeResult(row=(YABA[0], YABA[1], 1)),
            FakeResult(items=[]),
            FakeResult(items=[make_property("Luxury apartment", "Yaba, Lagos", *YABA)]),
        ])

        response = client.post(
            "/api/v1/search/semantic",
            json={"query": "apartment in yaba", "filters": {"location": "yaba"}},
        )

        assert response.status_code == 503
        # The test app has no custom exception handlers, so the body uses
        # FastAPI's default 'detail' key (main.py rewraps this as 'message')
        assert "temporarily unavailable" in response.json()["detail"].lower()
