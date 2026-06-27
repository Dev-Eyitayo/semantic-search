FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Force uv to pick the lightweight CPU-only wheels for PyTorch
ENV UV_TORCH_BACKEND=cpu

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the code
COPY . .
RUN chmod +x start.sh

# Run the bundle script
CMD ["./start.sh"]