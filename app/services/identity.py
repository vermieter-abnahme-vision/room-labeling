"""Labeler identity: upsert from OIDC claims and admin allowlist evaluation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Labeler

logger = logging.getLogger(__name__)


def is_admin_identity(
    settings: Settings, sub: str, email: str | None, email_verified: bool
) -> bool:
    """Admin if the OIDC ``sub`` is allowlisted, or the (verified) email is allowlisted."""
    if sub in settings.admin_subs:
        return True
    if not email:
        return False
    if settings.oidc_require_email_verified and not email_verified:
        return False
    return email.lower() in {e.lower() for e in settings.admin_emails}


def _display_name(claims: Mapping[str, Any]) -> str:
    for key in ("name", "preferred_username"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:255]
    email = claims.get("email")
    if isinstance(email, str) and "@" in email:
        return email.split("@", 1)[0][:255]
    return "labeler"


def upsert_labeler(session: Session, settings: Settings, claims: Mapping[str, Any]) -> Labeler:
    """Create or update the labeler for validated ID-token claims."""
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise ValueError("ID token has no 'sub' claim")
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    verified = claims.get("email_verified") is True
    labeler = session.scalar(select(Labeler).where(Labeler.oidc_sub == sub))
    admin = is_admin_identity(settings, sub, email, verified)
    if labeler is None:
        labeler = Labeler(
            oidc_sub=sub, email=email, display_name=_display_name(claims), is_admin=admin
        )
        session.add(labeler)
        logger.info("new labeler registered")
    else:
        labeler.email = email
        labeler.display_name = _display_name(claims)
        labeler.is_admin = admin
    session.flush()
    return labeler
