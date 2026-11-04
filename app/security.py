"""Security building blocks: headers middleware, CSRF, safe redirects, rate limiter."""

from __future__ import annotations

import secrets
import time
from collections.abc import MutableMapping
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from slowapi import Limiter
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CSRF_SESSION_KEY = "csrf"
CSRF_HEADER = "x-csrf-token"
CSRF_FORM_FIELD = "csrf_token"

CSP = "; ".join(
    [
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data:",
        "connect-src 'self'",
        "font-src 'self'",
        "manifest-src 'self'",
        "form-action 'self'",
        "base-uri 'none'",
        "frame-ancestors 'none'",
    ]
)


def security_headers(*, hsts: bool) -> dict[bytes, bytes]:
    headers = {
        b"content-security-policy": CSP,
        b"x-content-type-options": "nosniff",
        b"referrer-policy": "no-referrer",
        b"x-frame-options": "DENY",
        b"permissions-policy": "camera=(), microphone=(), geolocation=()",
        b"cross-origin-opener-policy": "same-origin",
        b"cross-origin-resource-policy": "same-origin",
    }
    if hsts:
        headers[b"strict-transport-security"] = "max-age=63072000; includeSubDomains"
    return {k: v.encode("latin-1") for k, v in headers.items()}


class SecurityHeadersMiddleware:
    """Pure ASGI middleware adding strict security headers to every HTTP response."""

    def __init__(self, app: ASGIApp, *, hsts: bool) -> None:
        self.app = app
        self.headers = security_headers(hsts=hsts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_static = scope["path"].startswith("/static/")

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw = [
                    (k, v) for k, v in message.get("headers", []) if k.lower() not in self.headers
                ]
                raw.extend(self.headers.items())
                if not is_static and not any(k.lower() == b"cache-control" for k, _ in raw):
                    raw.append((b"cache-control", b"no-store"))
                message["headers"] = raw
            await send(message)

        await self.app(scope, receive, send_wrapper)


# --- CSRF -----------------------------------------------------------------------------
def ensure_csrf_token(session: MutableMapping[str, Any]) -> str:
    token = session.get(CSRF_SESSION_KEY)
    if not isinstance(token, str) or len(token) < 32:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


async def verify_csrf(request: Request) -> None:
    """Dependency for every state-changing request (header for fetch, field for forms)."""
    expected = request.session.get(CSRF_SESSION_KEY)
    supplied = request.headers.get(CSRF_HEADER)
    if supplied is None and "application/x-www-form-urlencoded" in request.headers.get(
        "content-type", ""
    ):
        form = await request.form()
        value = form.get(CSRF_FORM_FIELD)
        supplied = value if isinstance(value, str) else None
    if (
        not isinstance(expected, str)
        or not supplied
        or not secrets.compare_digest(expected.encode(), supplied.encode())
    ):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


# --- Redirect safety --------------------------------------------------------------------
def safe_next_path(candidate: str | None, default: str = "/label") -> str:
    """Only allow same-site absolute paths (blocks open redirects like //evil.com or https://)."""
    if not candidate or not candidate.startswith("/") or candidate.startswith("//"):
        return default
    if "\\" in candidate or any(ord(c) < 32 for c in candidate):
        return default
    parts = urlsplit(candidate)
    if parts.scheme or parts.netloc:
        return default
    return candidate


# --- Rate limiting ----------------------------------------------------------------------
def rate_limit_key(request: Request) -> str:
    """Per authenticated labeler when available, else per client IP."""
    session = request.scope.get("session") or {}
    labeler_id = session.get("labeler_id")
    if labeler_id is not None:
        return f"labeler:{labeler_id}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


limiter = Limiter(key_func=rate_limit_key, default_limits=[])


class RateLimits:
    """Mutable holder so limit strings come from Settings at app creation (read lazily)."""

    login = "10/minute"
    submit = "120/minute"
    read = "240/minute"


rate_limits = RateLimits()


def now() -> float:
    return time.time()
