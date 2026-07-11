# Sheltly

Backend server for an AI-driven semantic search and explainable ranking system for a real estate platform.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17%20(pgvector)-336791)
![License](https://img.shields.io/badge/license-MIT-green)

Built as part of my final-year research on semantic search for real estate platforms. Traditional property search relies on keyword matching, which fails the moment a user's words don't literally appear in a listing — "cheap flat near leadcity" won't match a listing titled "Affordable 2-bedroom apartment in Challenge". Sheltly uses sentence embeddings to match on *meaning* instead, and ranks results with a weighted, explainable scoring model so users (and researchers) can see exactly why a listing ranked where it did.

The project doubles as a comparison testbed: it exposes both a semantic search endpoint and a plain keyword (Postgres full-text) endpoint, plus evaluation scripts (`eval_ndcg_precision.py`, `eval_ablation_study.py`) for measuring nDCG/precision and running ablations across ranking features.

## Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [API](#-api)
- [Configuration](#-configuration)
- [Example Searches](#-example-searches)
- [Design Decisions](#-design-decisions)
- [Future Improvements](#-future-improvements)
- [What I Learned](#-what-i-learned)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Semantic search** — natural-language queries matched against listings using S-BERT sentence embeddings (`all-MiniLM-L6-v2`, 384 dimensions) and cosine similarity.
- **Smart query parsing** — bedrooms ("2-bedroom", "3br"), property type, amenities, budget ("2.5M", "budget 3 million") and locations are extracted from the query text and applied as hard filters and score bonuses.
- **Explainable ranking** — every result carries per-feature scores (semantic, price, location, recency) with human-readable labels. A separate `/ai/explain` endpoint computes SHAP feature attributions and runs basic bias checks (e.g. flagging over-weighted location or price features).
- **Tunable ranking weights** — feature weights live in the database (`RankingConfig`) and can be adjusted at runtime via the admin API, with query-aware boosts (a query mentioning a location increases the location weight).
- **Geospatial search** — exact haversine distance math, radius search (`GET /search/nearby` or `lat`/`lng`/`radius_km` filters on semantic search), distance-aware ranking, and a layered geocoder that resolves free-text areas ("lekki", "vi") via Redis cache → listing centroids → optional Nominatim.
- **Keyword search baseline** — Postgres `tsvector` full-text search, used both as a fallback and as the control condition for the research comparison.
- **Similar properties, suggestions, and search feedback** — content-based "more like this", autocomplete from popular past queries (Redis-cached), and a feedback endpoint that logs relevance judgments for offline evaluation.
- **Full platform backend** — JWT auth with email verification (OTP), role-based permissions, property CRUD with an admin approval workflow, Cloudinary media uploads, background email delivery, and rate limiting.

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI (async), Uvicorn, Pydantic v2 |
| Database | PostgreSQL 17 (`pgvector/pgvector` image), SQLAlchemy 2.0 (asyncpg), Alembic migrations |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`), local inference or Hugging Face Inference API |
| Explainability | SHAP feature attribution |
| Caching / background work | Redis (suggestion + geocode cache, OTPs, token blacklist, rate limits); emails and search logging via FastAPI BackgroundTasks — no worker process required |
| Auth | JWT (python-jose), bcrypt, OTP email verification |
| Media | Cloudinary |
| Mail | fastapi-mail (MailHog in dev) |
| Tooling | uv, Docker Compose, Loguru, slowapi |

## ⚙️ How It Works

The pipeline, end to end:

1. **Ingestion** — listers create properties through the API (`POST /api/v1/properties`). Listings hold a title, free-text description, price, location (with optional coordinates), bedrooms/bathrooms, type, and amenities. Admins approve listings before they become searchable. A seed script (`scripts/seed_properties.py`) populates the database for development and experiments.

2. **Embedding generation** — each listing's title and description are concatenated and encoded into a 384-dimensional vector by `all-MiniLM-L6-v2`. An embedding is just a list of numbers positioned so that texts with similar meaning end up close together — "flat" and "apartment" land near each other even though they share no letters. The model runs locally by default; setting `EMBEDDING_BACKEND=hf_inference_api` offloads encoding to the Hugging Face Inference API instead (useful on small servers where loading torch is too heavy).

3. **Storage** — embeddings are L2-normalized and stored on the property row in Postgres, alongside an in-memory LRU-style cache (10k entries) so repeated encodes are free within a process. The `/api/v1/ai/reindex/{id}` endpoint regenerates a listing's embedding after edits.

4. **Query encoding** — at search time the user's raw query ("cheap 2 bedroom apartment in challenge") is encoded with the same model, so query and listings live in the same vector space. In parallel, lightweight parsers extract structured attributes (bedrooms, type, budget, amenities, locations) from the text.

5. **Similarity search** — candidate listings (pre-filtered by status and any extracted/explicit filters) are compared against the query vector using cosine similarity, computed as a single batched NumPy dot product since vectors are normalized. This returns a 0–1 semantic score per listing.

6. **Ranking and explanation** — the final score is a weighted combination:

   ```
   final = 0.50·semantic + 0.20·price + 0.20·location + 0.10·recency  (+ attribute-match bonuses)
   ```

   Weights come from the `RankingConfig` table and shift dynamically (e.g. location weight increases when the query names an area). The location score is distance-aware: when a search center is known (explicit coordinates, a geocoded filter location, or an area mentioned in the query), each listing's exact haversine distance feeds a smooth decay curve and is returned as `distance_km`. Listings matching extracted attributes (right type, right bedroom count, requested amenities) get small capped bonuses. When `explain: true` is passed, each result includes the per-feature breakdown and a plain-language summary; the `/ai/explain` endpoint goes further with SHAP attributions per feature.

Every search is logged asynchronously (query, filters, latency, result count) to `SearchLog`, which powers autocomplete suggestions and the offline evaluation scripts.

## 📁 Project Structure

```
semantic-search/
├── main.py                  # FastAPI app, CORS, routers, exception handlers
├── api/
│   ├── deps.py              # Auth dependencies, role checks
│   └── v1/                  # Route modules: auth, users, properties, media,
│                            #   search, ai (ranking/explainability), admin
├── core/                    # Settings, security/JWT, permissions, celery app,
│                            #   rate limiting, logging
├── db/
│   ├── models/              # SQLAlchemy models (Property, User, SearchLog,
│   │                        #   SearchFeedback, RankingConfig, ...)
│   └── session.py           # Async engine/session
├── schemas/                 # Pydantic request/response models
├── services/
│   ├── embedding_service.py # Model singleton, batch encoding, similarity, cache
│   ├── ai_service.py        # SHAP explanations, reranking
│   ├── geo_service.py       # Haversine, radius search, layered geocoding
│   └── ...                  # Redis, mail, Cloudinary, Celery tasks
├── scripts/                 # seed_properties.py, create_admin.py
├── migrations/              # Alembic migrations
├── tests/                   # Unit + endpoint tests (uv run pytest)
├── eval_ndcg_precision.py   # nDCG / precision@k evaluation
├── eval_ablation_study.py   # Ranking-feature ablation study
└── docker-compose*.yml      # Dev and prod stacks
```

## 🚀 Installation

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker.

```bash
# 1. Clone
git clone https://github.com/dev-eyitayo/semantic-search.git
cd semantic-search

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Environment variables
cp .env.example .env
# then edit .env — at minimum set SECRET_KEY (openssl rand -hex 32),
# REDIS_URL, and the Postgres credentials

# 4. Start Postgres, Redis, and MailHog for development
docker-compose -f docker-compose.dev.yml up -d

# 5. Run migrations
uv run alembic upgrade head

# 6. (Optional) seed sample properties and create an admin
uv run python -m scripts.seed_properties
uv run python -m scripts.create_admin

# 7. Start the API (docs served at http://localhost:8000/)
uv run fastapi dev main.py
```

Emails (OTP verification, password reset) are sent via FastAPI background tasks in the API
process — no separate worker is needed, which keeps the app deployable on single-process
free tiers (e.g. Render).

Alternatively, run the whole stack (API + Postgres + Redis + MailHog) with:

```bash
docker-compose up --build
```

The first local run downloads the embedding model (~90 MB). To skip that entirely, set `EMBEDDING_BACKEND=hf_inference_api` and provide `HF_API_TOKEN`.

## 📡 API

Interactive Swagger docs are served at `/` and ReDoc at `/redoc`. All responses share the envelope `{ "status", "message", "data" }`. Highlights below — auth, users, properties, media, and admin routers are documented in Swagger.

### `POST /api/v1/search/semantic`

Semantic search with query parsing, weighted ranking, and optional explanations.

```json
{
  "query": "2 bedroom apartment in challenge under 2.5M",
  "filters": { "price_max": 2500000 },
  "page": 1,
  "limit": 10,
  "explain": true
}
```

```json
{
  "status": "success",
  "message": "Search results retrieved successfully",
  "data": {
    "query": "2 bedroom apartment in challenge under 2.5M",
    "total_results": 14,
    "page": 1,
    "limit": 10,
    "processing_time_ms": 182,
    "results": [
      {
        "id": "7f9b6c2e-...",
        "title": "Modern 2-Bedroom Apartment in Challenge",
        "price": 2200000,
        "location": "Challenge, Ibadan",
        "bedrooms": 2,
        "ranking_score": 0.87,
        "semantic_score": 0.81,
        "price_score": 0.78,
        "location_score": 0.85,
        "recency_score": 0.95,
        "explanations": [
          {
            "feature": "semantic_score",
            "label": "Strongly matches your search intent",
            "weight": 0.81,
            "direction": "positive"
          }
        ],
        "explanation_summary": "Matches: Type: apartment + Bedrooms: 2. Strongly relevant!"
      }
    ]
  }
}
```

### `GET /api/v1/search/keyword`

Postgres full-text (`tsvector`) baseline search. Query params: `q`, `page`, `limit`, `location`, `bedrooms`, `price_max`.

```
GET /api/v1/search/keyword?q=apartment%20challenge&limit=10
```

### `GET /api/v1/search/nearby`

Radius search sorted by distance. Center from `lat`/`lng`, or from a free-text `location` that gets geocoded.

```
GET /api/v1/search/nearby?location=yaba&radius_km=5&limit=10
GET /api/v1/search/nearby?lat=6.5244&lng=3.3792&radius_km=5
```

Results include `distance_km` per property. The same geo filters work inside semantic search — pass `"filters": { "lat": 6.5244, "lng": 3.3792, "radius_km": 5 }` (or `"location": "yaba", "radius_km": 5`) to `POST /search/semantic` to constrain results to a radius while keeping semantic ranking; each result then carries its exact `distance_km`.

### `GET /api/v1/search/similar/{property_id}`

Content-based similar listings (bedrooms, geographic distance, price band, amenity overlap).

### `GET /api/v1/search/suggestions?q=2 bed`

Autocomplete from popular logged queries, cached in Redis for an hour.

### `POST /api/v1/search/feedback`

Log a relevance judgment (`clicked`, `contacted`, etc.) against a query/listing pair — the raw material for the evaluation scripts.

### `POST /api/v1/ai/explain`

SHAP feature attribution for a specific query + property pair, including bias flags (e.g. over-weighted location features).

```json
{ "query": "cheap apartment near university", "property_id": "7f9b6c2e-..." }
```

### Other AI/ranking endpoints

| Method | Route | Description |
|---|---|---|
| `POST` | `/api/v1/ai/embed` | Generate an embedding for arbitrary text |
| `POST` | `/api/v1/ai/reindex/{property_id}` | Regenerate a listing's stored embedding |
| `GET` / `PATCH` | `/api/v1/ai/ranking-config` | Read / update ranking weights (admin) |
| `GET` | `/api/v1/ai/audit` | Audit log of explanation requests |

Auth uses JWT bearer tokens: `POST /api/v1/auth/register` → `POST /api/v1/auth/verify-email` → `POST /api/v1/auth/login` → `Authorization: Bearer <token>`.

## 🔧 Configuration

All settings load from `.env` (see `.env.example`).

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | ✅ | JWT signing key — generate with `openssl rand -hex 32` |
| `ALGORITHM` | | JWT algorithm (default `HS256`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | | Access token lifetime |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_HOST` / `POSTGRES_PORT` | ✅* | Database credentials |
| `DATABASE_ASYNC_URL` / `DATABASE_SYNC_URL` | ✅* | Full connection strings; take precedence over the parts above. `sslmode=require` is added automatically for Supabase hosts |
| `REDIS_URL` | ✅ | Redis connection (cache + Celery broker) |
| `EMBEDDING_BACKEND` | | `local` (default) or `hf_inference_api` |
| `HF_API_TOKEN` | | Hugging Face token (required for `hf_inference_api`) |
| `HF_EMBEDDING_MODEL` | | Default `sentence-transformers/all-MiniLM-L6-v2` |
| `GEOCODER_PROVIDER` | | `nominatim` (default) enables the OpenStreetMap fallback for unknown locations; `none` disables external lookups |
| `NOMINATIM_BASE_URL` / `GEOCODER_COUNTRY_CODES` | | Nominatim endpoint and country bias (defaults: public OSM instance, `ng`) |
| `GEOCODE_CACHE_TTL_SECONDS` | | Geocode result cache lifetime (default 30 days) |
| `CLOUDINARY_CLOUD_NAME` / `CLOUDINARY_API_KEY` / `CLOUDINARY_API_SECRET` | | Media uploads |
| `MAIL_SERVER` / `MAIL_PORT` / `MAIL_FROM` / `MAIL_USERNAME` / `MAIL_PASSWORD` | | SMTP (MailHog defaults in dev) |
| `UNSPLASH_ACCESS_KEY` | ✅ | Used by the property seeding script for listing images |
| `FRONTEND_URL` | | Frontend origin for email links (default `http://localhost:3000`) |
| `VERSION` | | API version string |

\* Provide either the `DATABASE_*_URL` pair or the individual `POSTGRES_*` parts.

## 🔍 Example Searches

**"cheap apartment near leadcity"** — keyword search finds nothing unless a listing literally contains "cheap" or "leadcity". The embedding model places this query near listings describing "affordable self-contained flat close to Leadcity campus", because it has learned that cheap/affordable and university/campus/leadcity occupy the same semantic neighborhood.

**"family house with large compound"** — no listing says "family house". Semantic search surfaces "spacious 4-bedroom bungalow with big fenced yard, ideal for families" — *compound*, *yard*, and *fenced grounds* are near-synonyms in vector space, and the bedroom count reinforces the family intent.

**"modern duplex in lekki under 5M"** — this shows the hybrid layer on top of pure similarity: "duplex" is extracted and applied as a type filter, "5M" becomes a `price ≤ 5,000,000` filter, "lekki" boosts the location weight in the ranking formula, and the remaining text ("modern") is matched semantically against descriptions. Keyword search would treat "5M" and "under" as literal tokens and match nothing useful.

## 🧠 Design Decisions

**Embeddings over keywords.** Property seekers describe intent ("quiet area, close to work, within budget"), while listers write marketing copy. Keyword search only works when both sides happen to choose the same words. Embeddings close that vocabulary gap. The keyword endpoint was kept deliberately — the research question is *how much better* semantic search is, and you can't measure that without a controlled baseline on the same data.

**`all-MiniLM-L6-v2` over larger models.** Bigger models (e.g. `all-mpnet-base-v2`, 768-dim) score a few points higher on retrieval benchmarks, but MiniLM is ~5x faster, small enough to run on a CPU-only container, and 384-dim vectors halve storage and similarity-computation cost. For short property descriptions the quality gap was not worth the latency. The pluggable HF Inference API backend exists for the same reason: the deployment target couldn't afford to hold torch + model weights in memory.

**Explainability as a first-class feature.** A single opaque relevance score is hard to trust and impossible to debug. Decomposing the ranking into weighted, named features (semantic/price/location/recency) means every result can say *why* it ranked where it did, weights can be tuned from data rather than guesswork, and the SHAP layer can flag when a feature (like location) is dominating in ways that might encode bias.

**In-application similarity, for now.** Candidates are pre-filtered in SQL (status, price, bedrooms, type), then scored in a single batched NumPy dot product. At the thousands-of-listings scale of this research project, that's fast (the batch path is capped at 1,000 candidates) and keeps the ranking logic in one debuggable place. It does not scale to millions of listings — that's what the pgvector migration below is for; the Docker stack already runs the `pgvector/pgvector` Postgres image in anticipation.

**Bounding box + haversine over PostGIS.** Radius search runs as an indexed SQL bounding-box prefilter followed by exact haversine refinement in Python. PostGIS would be the textbook answer, but it isn't in the local `pgvector` image, and adding it means a custom database build for what is point-radius search over thousands of rows — the two-step approach is exact, uses a plain btree index on `(latitude, longitude)`, and works identically on local Docker and Supabase. Location strings are geocoded in layers: Redis cache first, then the centroid of our own matching listings (free and self-consistent with what's actually searchable), and only then the Nominatim API — with negative caching so unknown strings don't hammer an external service on every search.

**Trade-offs accepted:** embedding inference adds latency versus a pure SQL query (mitigated by the in-memory embedding cache and batch encoding); the linear weighted ranking is less powerful than a learned ranker but is transparent and cheap to explain; the regex-based query parsers are simple and fail on unusual phrasings, but they're deterministic and free compared to an LLM parsing step.

## 🗺 Future Improvements

- **Native pgvector similarity** — move cosine similarity into Postgres with an HNSW/IVFFlat index so retrieval scales past the in-app candidate cap.
- **Hybrid search** — fuse semantic and BM25/full-text scores (e.g. reciprocal rank fusion) instead of treating keyword search as a fallback.
- **Cross-encoder reranking** — rerank the top-k candidates with a cross-encoder for higher precision at the top of the list.
- **Learned ranking weights** — fit the feature weights from the collected `SearchFeedback` data instead of hand-tuning them.
- **PostGIS** — if listing volume or query patterns outgrow the bounding-box + haversine approach, move distance computation into the database with a proper spatial index.
- **Personalization** — bias ranking using a user's search and feedback history.
- **Multilingual queries** — swap in a multilingual embedding model (e.g. `paraphrase-multilingual-MiniLM`) to support Nigerian Pidgin and other languages.

## 📚 What I Learned

The biggest lesson: the embedding model is the easy part. `sentence-transformers` gives you semantic similarity in five lines — the actual work is everything around it. Real queries mix semantic intent with hard constraints ("2 bedroom", "under 2.5M", "in yaba"), and a pure vector search cheerfully returns a beautiful 3-bedroom duplex for a "2 bedroom" query because they're semantically close. Separating *constraints* (filter) from *intent* (rank) mattered more to result quality than any model choice.

Operationally, embedding models are heavier than typical web dependencies — a naive deployment pulled in CUDA torch and blew up the Docker image, which forced the CPU-only build and eventually the remote-inference backend. Batch encoding was the single biggest performance win: encoding 1,000 candidate descriptions one-by-one is unusable; as one batched call it's milliseconds. And normalizing embeddings up front turns cosine similarity into a plain dot product, which simplifies everything downstream.

Finally, evaluation is genuinely hard. "The results look better" isn't a finding — building the feedback logging, relevance judgments, and nDCG/ablation scripts took real effort but is what turns a demo into a defensible comparison between semantic and keyword retrieval.

## 🤝 Contributing

Contributions are welcome, especially around the future-improvements list.

1. Fork the repo and create a branch: `git checkout -b feat/my-change`
2. Install dependencies with `uv sync` and copy `.env.example` to `.env`
3. Make your changes; keep the code style consistent with the existing modules (typed SQLAlchemy models, Pydantic schemas per router, Loguru for logging)
4. If you change models, generate a migration: `uv run alembic revision --autogenerate -m "describe change"`
5. Run the tests (`uv run pytest`), and the evaluation scripts if your change affects search or ranking behavior
6. Open a PR with a clear description of what changed and why

For bugs and ideas, open an issue first so we can discuss the approach.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
