#!/bin/bash
set -e

export PYTHONPATH=.

if [ "$PROCESS_TYPE" = "api" ]; then
    uv run alembic upgrade head
    uv run python -m scripts.seed_properties
    exec uv run uvicorn main:app --host 0.0.0.0 --port 8000
elif [ "$PROCESS_TYPE" = "worker" ]; then
    exec uv run celery -A core.celery worker --loglevel=info --concurrency=1
else
    exec "$@"
fi