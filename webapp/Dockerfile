# ═══════════════════════════════════════════════════════════════════════════
# Frame2D Analysis Web App - Docker Image
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend /app/backend

# Set Python path
ENV PYTHONPATH=/app/backend

# Working directory for the app
WORKDIR /app/backend

# Create jobs directory
RUN mkdir -p /app/backend/jobs

# Default command (overridden in docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
