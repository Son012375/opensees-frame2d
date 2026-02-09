"""
Application configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
JOBS_DIR = BASE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# MCP server path (existing analysis code)
MCP_SERVER_PATH = Path(os.getenv("MCP_SERVER_PATH", "d:/son/opensees-MCP/mcp-server"))

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Worker settings
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", "300"))  # 5 minutes
