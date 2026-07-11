"""
Shared test configuration.

Sets the environment variables Settings requires BEFORE any project module
is imported, so tests run without a .env file (CI) and never hit external
services (geocoder disabled, remote embedding backend so no model download).
"""

import os

os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("UNSPLASH_ACCESS_KEY", "test-unsplash-key")
os.environ["GEOCODER_PROVIDER"] = "none"
os.environ.setdefault("EMBEDDING_BACKEND", "hf_inference_api")
