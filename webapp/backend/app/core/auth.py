"""
Simple token-based authentication for demo deployment.

Usage:
  - Set DEMO_AUTH_TOKEN env var to enable authentication
  - Access via: https://your-app.azurecontainerapps.io/?token=YOUR_TOKEN
  - Or via header: Authorization: Bearer YOUR_TOKEN
  - If DEMO_AUTH_TOKEN is not set, auth is disabled (local dev)
"""
import os
from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware


DEMO_AUTH_TOKEN = os.getenv("DEMO_AUTH_TOKEN", "")

# Paths that skip authentication
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json"}


class DemoAuthMiddleware(BaseHTTPMiddleware):
    """Simple token auth middleware for demo sharing."""

    async def dispatch(self, request: Request, call_next):
        # Skip if auth not configured
        if not DEMO_AUTH_TOKEN:
            return await call_next(request)

        # Skip public endpoints
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Check token from query param, cookie, or header
        token = (
            request.query_params.get("token")
            or request.cookies.get("demo_token")
            or _extract_bearer(request)
        )

        if token == DEMO_AUTH_TOKEN:
            # Set cookie so user doesn't need token in every request
            response = await call_next(request)
            if request.query_params.get("token"):
                response.set_cookie(
                    "demo_token", DEMO_AUTH_TOKEN,
                    httponly=True, samesite="lax", max_age=86400 * 7,  # 7 days
                )
            return response

        # Unauthorized - show simple login page
        return HTMLResponse(
            content=_login_page(),
            status_code=401,
        )


def _extract_bearer(request: Request) -> str:
    """Extract token from Authorization: Bearer <token> header."""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


def _login_page() -> str:
    return """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>OpenSees Demo - Access Required</title>
    <style>
        body { font-family: -apple-system, sans-serif; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; background: #f5f5f5; }
        .card { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); text-align: center; max-width: 400px; }
        h2 { margin-top: 0; }
        input { padding: 0.6rem 1rem; border: 1px solid #ddd; border-radius: 6px; font-size: 1rem; width: 200px; }
        button { padding: 0.6rem 1.5rem; background: #2563eb; color: white; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; margin-left: 0.5rem; }
        button:hover { background: #1d4ed8; }
        .hint { color: #888; font-size: 0.85rem; margin-top: 1rem; }
    </style>
</head>
<body>
    <div class="card">
        <h2>OpenSees Demo</h2>
        <p>Access token required</p>
        <form onsubmit="go(event)">
            <input type="text" id="tok" placeholder="Enter token" autofocus />
            <button type="submit">Enter</button>
        </form>
        <p class="hint">Contact the admin for access.</p>
    </div>
    <script>
        function go(e) {
            e.preventDefault();
            const t = document.getElementById('tok').value.trim();
            if (t) window.location.href = '/?token=' + encodeURIComponent(t);
        }
    </script>
</body>
</html>"""
