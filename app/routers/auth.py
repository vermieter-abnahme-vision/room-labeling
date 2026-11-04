"""OIDC authorization-code flow with PKCE (S256), state + nonce, id_token validation."""

from __future__ import annotations

import logging
import secrets

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, Request
from starlette.responses import RedirectResponse, Response

from app.config import Settings
from app.db import Database
from app.deps import get_database, get_settings_dep
from app.security import (
    ensure_csrf_token,
    limiter,
    now,
    rate_limits,
    safe_next_path,
    verify_csrf,
)
from app.services.identity import upsert_labeler
from app.web.rendering import render

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

NEXT_KEY = "post_login_next"


def build_oauth(settings: Settings) -> OAuth:
    """Register the (provider-agnostic) OIDC client from ENV configuration."""
    assert settings.oidc_issuer and settings.oidc_client_id
    oauth = OAuth()
    oauth.register(
        name="oidc",
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret,
        server_metadata_url=settings.oidc_issuer.rstrip("/") + "/.well-known/openid-configuration",
        client_kwargs={"scope": settings.oidc_scopes, "code_challenge_method": "S256"},
    )
    return oauth


@router.get("/login", include_in_schema=False)
@limiter.limit(lambda: rate_limits.login)
async def login(
    request: Request, settings: Settings = Depends(get_settings_dep), next: str | None = None
) -> Response:
    request.session.clear()  # never reuse pre-login session data
    request.session[NEXT_KEY] = safe_next_path(next)
    oauth: OAuth = request.app.state.oauth
    # redirect_uri comes from configuration only (allowlist enforced at startup), never the request
    return await oauth.oidc.authorize_redirect(request, settings.oidc_redirect_uri)  # type: ignore[no-any-return]


@router.get("/callback", include_in_schema=False)
@limiter.limit(lambda: rate_limits.login)
async def callback(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    database: Database = Depends(get_database),
) -> Response:
    oauth: OAuth = request.app.state.oauth
    next_path = safe_next_path(request.session.get(NEXT_KEY))
    try:
        # Validates state, PKCE verifier, nonce, id_token signature, iss, aud and exp.
        token = await oauth.oidc.authorize_access_token(request)
    except OAuthError as exc:
        logger.warning("OIDC callback rejected: %s", exc.error)
        request.session.clear()
        return render(request, "error.html", {"message": "Anmeldung fehlgeschlagen."}, 400)
    claims = token.get("userinfo")
    if not claims:
        logger.warning("OIDC callback without verifiable id_token")
        request.session.clear()
        return render(request, "error.html", {"message": "Anmeldung fehlgeschlagen."}, 400)

    with database.write_session() as db:
        labeler = upsert_labeler(db, settings, dict(claims))
    # Session rotation: drop everything from the login phase, issue fresh identifiers.
    request.session.clear()
    request.session.update(
        {
            "labeler_id": labeler.id,
            "sub": labeler.oidc_sub,
            "email_verified": claims.get("email_verified") is True,
            "iat": int(now()),
            "sid": secrets.token_urlsafe(16),
        }
    )
    ensure_csrf_token(request.session)
    logger.info("login ok labeler_id=%s", labeler.id)
    return RedirectResponse(next_path, status_code=303)


@router.post("/logout", include_in_schema=False, dependencies=[Depends(verify_csrf)])
async def logout(request: Request) -> Response:
    request.session.clear()
    return RedirectResponse("/auth/login", status_code=303)
