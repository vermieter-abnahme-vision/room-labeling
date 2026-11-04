"""Application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.staticfiles import StaticFiles

from app import __version__
from app.config import Settings, get_settings
from app.db import Database, get_database
from app.deps import NotAuthenticatedError
from app.labelschemes import SchemeRegistry
from app.logging_config import configure_logging
from app.routers import admin, auth, health, images, labeling
from app.security import (
    SecurityHeadersMiddleware,
    limiter,
    rate_limits,
    safe_next_path,
    security_headers,
)
from app.web.rendering import WEB_DIR, render

logger = logging.getLogger(__name__)

SESSION_COOKIE = "rl_session"


def _wants_json(request: Request) -> bool:
    return request.url.path.startswith("/api/") or "text/html" not in request.headers.get(
        "accept", ""
    )


def create_app(settings: Settings | None = None, database: Database | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.require_web_config()
    configure_logging()

    rate_limits.login = settings.rate_limit_login
    rate_limits.submit = settings.rate_limit_submit
    rate_limits.read = settings.rate_limit_read
    limiter.enabled = settings.rate_limit_enabled

    db = database or get_database(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        logger.info("room-labeling %s started (%s)", __version__, settings.environment)
        yield

    dev = settings.environment == "dev"
    app = FastAPI(
        title="Room Labeling",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if dev else None,
        redoc_url=None,
        openapi_url="/openapi.json" if dev else None,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.registry = SchemeRegistry.from_settings(settings)
    app.state.limiter = limiter
    app.state.oauth = auth.build_oauth(settings)

    # Middleware order: last added = outermost.
    assert settings.secret_key
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie=SESSION_COOKIE,
        max_age=settings.session_ttl_seconds,
        same_site="lax",
        https_only=settings.cookie_secure,
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.hsts_enabled)

    # --- Error handling: generic messages, no stack traces to the client -----------------
    @app.exception_handler(NotAuthenticatedError)
    async def _not_authenticated(request: Request, _exc: NotAuthenticatedError) -> Response:
        if _wants_json(request):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        target = safe_next_path(
            request.url.path + (f"?{request.url.query}" if request.url.query else "")
        )
        return RedirectResponse(f"/auth/login?next={target}", status_code=303)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        messages = {
            403: "Kein Zugriff.",
            404: "Nicht gefunden.",
            429: "Zu viele Anfragen.",
        }
        if _wants_json(request):
            return JSONResponse(
                {"detail": exc.detail if exc.status_code < 500 else "Error"}, exc.status_code
            )
        return render(
            request,
            "error.html",
            {"message": messages.get(exc.status_code, "Fehler.")},
            exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, _exc: RequestValidationError) -> Response:
        return JSONResponse({"detail": "Invalid request"}, status_code=422)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(_request: Request, _exc: RateLimitExceeded) -> Response:
        return JSONResponse(
            {"detail": "Too many requests"}, status_code=429, headers={"Retry-After": "30"}
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> Response:
        logger.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        # ServerErrorMiddleware sits outside our middleware: add the headers here.
        headers = {
            k.decode(): v.decode() for k, v in security_headers(hsts=settings.hsts_enabled).items()
        }
        return JSONResponse({"detail": "Internal server error"}, status_code=500, headers=headers)

    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")
    for router in (health.router, auth.router, labeling.router, images.router, admin.router):
        app.include_router(router)
    return app


def run_factory() -> FastAPI:  # pragma: no cover - uvicorn entry (`--factory`)
    return create_app()
