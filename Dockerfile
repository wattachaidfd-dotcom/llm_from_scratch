# ─────────────────────────────────────────────
# LLM-from-Scratch — Production Docker Image
# ─────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# Copy project
COPY . .

# Make sure all part_* dirs are importable
ENV PYTHONPATH="/app:/app/part_3:/app/part_4:/app/part_6"

# Default: run API server
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
