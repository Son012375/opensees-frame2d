# ═══════════════════════════════════════════════════════════════════════════
# OpenSees Structural Analysis Platform - Docker Image
# Build context: project root (opensees-MCP/)
# Python 3.12 + opensees + ifcopenshell
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

WORKDIR /app

# System dependencies for OpenSees + numerical libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY webapp/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy MCP server (analysis engine)
COPY mcp-server /app/mcp-server

# Copy webapp backend
COPY webapp/backend /app/backend

# Copy data directory (occupancy mapping, section DB, etc.)
COPY data /app/data

# Python path: backend + mcp-server
ENV PYTHONPATH=/app/backend:/app/mcp-server

WORKDIR /app/backend

# Ephemeral jobs directory
RUN mkdir -p /app/backend/jobs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# main_simple: synchronous mode, no Redis/Celery
CMD ["uvicorn", "app.main_simple:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
