"""FastAPI dependencies: settings, DB sessions, authentication/authorisation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from app.config import Settings
from app.db import Database
from app.labelschemes import SchemeRegistry
from app.models import Labeler
from app.security import now
from app.services.identity import is_admin_identity

logger = logging.getLogger(__name__)


class NotAuthenticatedError(Exception):
    """Raised when no valid session exists; handled in main (redirect or 401)."""


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_registry(request: Request) -> SchemeRegistry:
    registry: SchemeRegistry = request.app.state.registry
    return registry


def get_database(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


@dataclass(frozen=True)
class CurrentUser:
    labeler: Labeler
    is_admin: bool


def current_user(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    database: Database = Depends(get_database),
) -> CurrentUser:
    """The authenticated labeler for this request (session cookie), or NotAuthenticatedError."""
    data = request.session
    labeler_id = data.get("labeler_id")
    sub = data.get("sub")
    issued = data.get("iat")
    if not isinstance(labeler_id, int) or not isinstance(sub, str) or not isinstance(issued, int):
        raise NotAuthenticatedError
    if now() - issued > settings.session_ttl_seconds:
        data.clear()
        raise NotAuthenticatedError
    with database.read_session() as db:
        labeler = db.get(Labeler, labeler_id)
    if labeler is None or labeler.oidc_sub != sub or labeler.is_library:
        data.clear()
        raise NotAuthenticatedError
    admin = is_admin_identity(
        settings, labeler.oidc_sub, labeler.email, data.get("email_verified") is True
    )
    return CurrentUser(labeler=labeler, is_admin=admin)


def require_admin(user: CurrentUser = Depends(current_user)) -> CurrentUser:
    """Server-side admin check on every /admin route (live ENV allowlist, not a stored flag)."""
    if not user.is_admin:
        logger.warning("admin access denied for labeler id=%s", user.labeler.id)
        raise HTTPException(status_code=403, detail="Forbidden")
    return user
