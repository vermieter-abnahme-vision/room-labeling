from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from authlib.integrations.base_client.errors import OAuthError
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import Database
from app.main import create_app
from app.models import Labeler
from app.services.identity import is_admin_identity, upsert_labeler


@pytest.fixture
def client_and_app(settings: Settings, db: Database) -> Iterator[tuple[TestClient, Any]]:
    app = create_app(settings, db)
    with TestClient(app, follow_redirects=False) as client:
        yield client, app


def _patch_token(
    app: Any, monkeypatch: pytest.MonkeyPatch, token: dict[str, Any] | Exception
) -> None:
    async def fake(_request: Any, **_kw: Any) -> dict[str, Any]:
        if isinstance(token, Exception):
            raise token
        return token

    monkeypatch.setattr(app.state.oauth.oidc, "authorize_access_token", fake)


def test_callback_success_creates_labeler_and_rotates_session(
    client_and_app: tuple[TestClient, Any], db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app = client_and_app
    claims = {
        "sub": "new-user",
        "email": "new@example.test",
        "email_verified": True,
        "name": "Nina",
    }
    _patch_token(app, monkeypatch, {"userinfo": claims})
    r = client.get("/auth/callback?code=c&state=s")
    assert r.status_code == 303 and r.headers["location"] == "/label"
    with db.read_session() as s:
        labeler = s.scalar(select(Labeler).where(Labeler.oidc_sub == "new-user"))
        assert labeler is not None and labeler.display_name == "Nina" and not labeler.is_admin
    # the freshly issued session grants access
    assert client.get("/api/next").status_code == 200


def test_callback_session_rotation_drops_login_state(
    client_and_app: tuple[TestClient, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app = client_and_app
    _patch_token(app, monkeypatch, {"userinfo": {"sub": "u1"}})
    first = client.get("/auth/callback?code=c&state=s")
    assert first.status_code == 303
    cookie_a = client.cookies.get("rl_session")
    client.cookies.clear()
    _patch_token(app, monkeypatch, {"userinfo": {"sub": "u1"}})
    client.get("/auth/callback?code=c&state=s")
    assert client.cookies.get("rl_session") != cookie_a  # new sid/csrf on every login


def test_callback_rejects_token_without_verified_userinfo(
    client_and_app: tuple[TestClient, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app = client_and_app
    _patch_token(app, monkeypatch, {"access_token": "x"})  # no id_token/userinfo
    r = client.get("/auth/callback?code=c&state=s", headers={"accept": "text/html"})
    assert r.status_code == 400 and client.get("/api/next").status_code == 401


def test_callback_oauth_error_is_generic(
    client_and_app: tuple[TestClient, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, app = client_and_app
    _patch_token(app, monkeypatch, OAuthError(error="invalid_grant", description="secret detail"))
    r = client.get("/auth/callback?code=c&state=s", headers={"accept": "text/html"})
    assert r.status_code == 400 and "secret detail" not in r.text


def test_admin_allowlist_rules(settings: Settings) -> None:
    assert is_admin_identity(settings, "admin-sub", None, False)
    assert is_admin_identity(settings, "x", "Boss@Example.test", True)
    assert not is_admin_identity(settings, "x", "boss@example.test", False)  # unverified e-mail
    assert not is_admin_identity(settings, "x", None, True)
    assert not is_admin_identity(settings, "x", "other@example.test", True)
    lax = settings.model_copy(update={"oidc_require_email_verified": False})
    assert is_admin_identity(lax, "x", "boss@example.test", False)


def test_upsert_labeler_updates_profile_and_admin_flag(
    session: Session, settings: Settings
) -> None:
    a = upsert_labeler(
        session, settings, {"sub": "s1", "email": "a@example.test", "preferred_username": "anna"}
    )
    assert a.display_name == "anna" and not a.is_admin
    b = upsert_labeler(
        session,
        settings,
        {"sub": "s1", "email": "boss@example.test", "email_verified": True, "name": "Anna B"},
    )
    assert b.id == a.id and b.is_admin and b.display_name == "Anna B"
    c = upsert_labeler(session, settings, {"sub": "s2", "email": "zed@example.test"})
    assert c.display_name == "zed"
    with pytest.raises(ValueError):
        upsert_labeler(session, settings, {"email": "nosub@example.test"})
