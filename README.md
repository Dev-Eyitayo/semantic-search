commands to start up my set up

docker-compose -f docker-compose.dev.yml up

# Run the FastAPI server with auto-reload
uv run fastapi dev main.py

# Start the worker
uv run celery -A core.celery_app worker --loglevel=info -P solo