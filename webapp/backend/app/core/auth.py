"""
Simple token-based authentication for demo deployment.

Usage:
  - Set DEMO_AUTH_TOKEN env var to enable authentication
  - Access via: https://your-url/?token=YOUR_TOKEN
  - If DEMO_AUTH_TOKEN is not set, auth is disabled (local dev)
"""
import os
from fastapi import Request, Response
from fastapi.responses import HTMLResponse


def _get_token():
    return os.getenv("DEMO_AUTH_TOKEN", "")


async def check_demo_auth(request: Request):
    """Dependency that checks demo auth token.
    Returns None if auth passes, or HTMLResponse with login page if not."""
    auth_token = _get_token()

    # Skip if auth not configured
    if not auth_token:
        return None

    # Skip static and public paths
    path = request.url.path
    if path in {"/health", "/docs", "/openapi.json"} or path.startswith("/static/"):
        return None

    # Check token from query param, cookie, or header
    token = (
        request.query_params.get("token")
        or request.cookies.get("demo_token")
        or _extract_bearer(request)
    )

    if token == auth_token:
        return None

    # Not authenticated
    return "unauthorized"


def is_operator_token(request: Request) -> bool:
    """Whether the request carries operator (full-access) privileges.

    Operator privilege is the trusted-context gate for the KDS audit endpoint.
    Semantics mirror :func:`check_demo_auth`:

      - ``DEMO_AUTH_TOKEN`` **unset** -> ``True``. This is dev / trusted-operator
        mode (effectively open), consistent with the rest of the app being open
        when no token is configured. **A shared deployment MUST set
        ``DEMO_AUTH_TOKEN``** so this returns ``True`` only for the operator.
      - set -> ``True`` only if the request supplies the matching token
        (query ``token`` / cookie ``demo_token`` / ``Authorization: Bearer``).
    """
    auth_token = _get_token()
    if not auth_token:
        return True
    token = (
        request.query_params.get("token")
        or request.cookies.get("demo_token")
        or _extract_bearer(request)
    )
    return token == auth_token


def make_auth_response(request: Request) -> HTMLResponse:
    """Create the login page response."""
    return HTMLResponse(content=_login_page(), status_code=401)


def set_auth_cookie(response: Response, request: Request):
    """Set auth cookie if token was passed as query param."""
    auth_token = _get_token()
    if auth_token and request.query_params.get("token"):
        response.set_cookie(
            "demo_token", auth_token,
            httponly=True, samesite="lax", max_age=86400 * 7,
        )


def _extract_bearer(request: Request) -> str:
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
