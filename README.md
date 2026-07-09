commands to start up locally

# Start the database and redis service for dev
docker-compose -f docker-compose.dev.yml up

# Run the FastAPI server with auto-reload
uv run fastapi dev main.py

# Start the worker
uv run celery -A core.celery_app worker --loglevel=info -P solo

# Run the seed script
Set-Location 'C:\Users\user\Documents\semantic-search'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe .\scripts\seed_properties.py
