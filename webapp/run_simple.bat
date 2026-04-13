@echo off
echo ============================================
echo Frame2D Analysis - Simple Mode (No Redis)
echo ============================================
echo.
echo This mode runs analysis synchronously.
echo No Redis or Celery required!
echo.

set MCP_SERVER_PATH=d:/son/opensees-MCP/mcp-server
set PYTHONPATH=%~dp0backend

cd /d %~dp0backend

echo Starting server...
echo.
echo ============================================
echo Server running at: http://localhost:8000
echo Press Ctrl+C to stop
echo ============================================
echo.

python -m uvicorn app.main_simple:app --host 0.0.0.0 --port 8000 --reload
