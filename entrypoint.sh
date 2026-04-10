#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

export PATH="/app/.venv/bin:$PATH"

# Define the database host and port
DB_HOST=${POSTGRES_HOST:-sheltly-postgres-db}
DB_PORT=5432

echo "----------------------------------------------------"
echo "🚀 Sheltly Entrypoint Script Initializing..."
echo "----------------------------------------------------"

# Function to wait for Postgres to be ready
wait_for_db() {
  echo "⏳ Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
  
  # Use netcat (nc) to check if the port is open
  while ! nc -z "$DB_HOST" "$DB_PORT"; do
    sleep 0.5
  done
  
  echo "✅ PostgreSQL is up and responding!"
}

# 1. Always wait for the database regardless of the process type
wait_for_db

# 2. Run migrations ONLY if this is the API container
# This prevents multiple containers from trying to migrate at the same time
if [ "$PROCESS_TYPE" = "api" ]; then
    echo "🛠️ Detected API process. Checking for pending migrations..."
    # 'upgrade head' will create tables if they don't exist 
    # (once you have migration files in /migrations/versions)
    alembic upgrade head
    echo "✅ Database is synchronized."
fi

echo "----------------------------------------------------"
echo "🎬 Starting Process: $@"
echo "----------------------------------------------------"

# Execute the CMD passed from docker-compose or Dockerfile
exec "$@"