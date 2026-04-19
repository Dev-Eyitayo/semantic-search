#!/bin/bash

# Start Celery Worker with ONLY 1 concurrency (saves RAM)
# The worker will be the only one loading the heavy Reranker model
uv run celery -A core.celery worker --loglevel=info --concurrency=1 &

# Start FastAPI
uv run fastapi run main.py --host 0.0.0.0 --port $PORT