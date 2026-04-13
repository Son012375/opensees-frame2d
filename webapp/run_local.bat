@echo off
echo ============================================
echo Frame2D Analysis Web App - Local Runner
echo ============================================
echo.

REM Check if Redis is running
echo [1/3] Checking Redis...
redis-cli ping >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Redis is not running!
    echo.
    echo Please install and start Redis first:
    echo   Option A: Install Memurai (Redis for Windows)
    echo            https://www.memurai.com/
    echo   Option B: Use WSL2 with Redis
    echo            wsl -d Ubuntu
    echo            sudo apt install redis-server
    echo            redis-server
    echo.
    pause
    exit /b 1
)
echo Redis OK!
echo.

REM Set environment variables
set REDIS_URL=redis://localhost:6379/0
set CELERY_BROKER_URL=redis://localhost:6379/0
set CELERY_RESULT_BACKEND=redis://localhost:6379/0
set MCP_SERVER_PATH=d:/son/opensees-MCP/mcp-server
set PYTHONPATH=%~dp0backend

echo [2/3] Starting Celery Worker in background...
start "Celery Worker" cmd /k "cd /d %~dp0backend && celery -A app.tasks.celery_app worker --loglevel=info --pool=solo"

timeout /t 3 /nobreak >nul

echo [3/3] Starting API Server...
echo.
echo ============================================
echo Server running at: http://localhost:8000
echo Press Ctrl+C to stop
echo ============================================
echo.

cd /d %~dp0backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
