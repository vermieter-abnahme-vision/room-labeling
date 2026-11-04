from __future__ import annotations

from collections.abc import Iterator
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import Database
from app.main import SESSION_COOKIE, create_app
from app.models import Consensus, ConsensusStatus, ImageSource, Labeler
from app.security import safe_next_path
from app.services.annotations import submit_label
from tests.conftest import (
    SECRET,
    login_session,
    make_image,
    make_labeler,
    session_cookie,
    write_jpeg,
)

CSRF = "c" * 43


@pytest.fixture
def app_client(settings: Settings, db: Database) -> Iterator[TestClient]:
    app = create_app(settings, db)
    with TestClient(app, base_url="http://testserver", follow_redirects=False) as client:
        yield client


def login(client: TestClient, db: Database, labeler: Labeler, **kw: object) -> None:
    client.cookies.set(SESSION_COOKIE, session_cookie(login_session(labeler, CSRF, **kw)))  # type: ignore[arg-type]


@pytest.fixture
def world(db: Database, settings: Settings):  # type: ignore[no-untyped-def]
    with db.write_session() as s:
        alice = make_labeler(s, "alice", sub="alice-sub", email="alice@example.test")
        admin = make_labeler(s, "boss", sub="admin-sub", email="boss@example.test")
        images = [make_image(s, i, group=f"g{i}") for i in range(1, 4)]
    for img in images:
        write_jpeg(settings.images_dir / img.path, (img.id * 40, 20, 20))
    return alice, admin, images


# --- authentication --------------------------------------------------------------------
def test_unauthenticated_html_redirects_to_login(app_client: TestClient) -> None:
    r = app_client.get("/label", headers={"accept": "text/html"})
    assert r.status_code == 303 and r.headers["location"].startswith("/auth/login?next=/label")


def test_unauthenticated_api_and_images_are_401(app_client: TestClient, world) -> None:  # type: ignore[no-untyped-def]
    _, _, images = world
    assert app_client.get("/api/next").status_code == 401
    assert app_client.get(f"/img/{images[0].sha256}").status_code == 401
    assert app_client.post("/api/label", json={}).status_code in (401, 403)


def test_forged_or_tampered_cookie_is_rejected(app_client: TestClient, world) -> None:  # type: ignore[no-untyped-def]
    alice, _, _ = world
    good = session_cookie(login_session(alice, CSRF))
    app_client.cookies.set(SESSION_COOKIE, good[:-3] + "AAA")
    assert app_client.get("/api/next").status_code == 401


def test_expired_session_rejected(
    app_client: TestClient, db: Database, world, settings: Settings
) -> None:  # type: ignore[no-untyped-def]
    alice, _, _ = world
    data = login_session(alice, CSRF)
    data["iat"] = int(data["iat"]) - settings.session_ttl_seconds - 10  # type: ignore[call-overload]
    app_client.cookies.set(SESSION_COOKIE, session_cookie(data))
    assert app_client.get("/api/next").status_code == 401


# --- labeling flow + CSRF ------------------------------------------------------------------
def test_label_flow(app_client: TestClient, db: Database, world) -> None:  # type: ignore[no-untyped-def]
    alice, _, images = world
    login(app_client, db, alice)
    data = app_client.get("/api/next").json()
    assert data["image"] and data["progress"] == {"done": 0, "total": 3}
    assert [l["key"] for l in data["labels"]][:2] == ["bad", "kueche"]
    sha = data["image"]["sha256"]
    assert app_client.get(data["image"]["url"]).headers["content-type"] == "image/jpeg"

    r = app_client.post(
        "/api/label", json={"sha256": sha, "label": "kueche"}, headers={"X-CSRF-Token": CSRF}
    )
    assert r.status_code == 200
    nxt = r.json()["next"]
    assert nxt["progress"] == {"done": 1, "total": 3}
    assert nxt["image"]["sha256"] != sha
    assert app_client.get(f"/api/my-label/{sha}").json() == {"label": "kueche"}

    # idempotent re-submit / correction does not create a second vote
    r = app_client.post(
        "/api/label", json={"sha256": sha, "label": "bad"}, headers={"X-CSRF-Token": CSRF}
    )
    assert r.status_code == 200 and r.json()["next"]["progress"]["done"] == 1
    assert app_client.get(f"/api/my-label/{sha}").json() == {"label": "bad"}


def test_label_post_requires_csrf(app_client: TestClient, db: Database, world) -> None:  # type: ignore[no-untyped-def]
    alice, _, images = world
    login(app_client, db, alice)
    body = {"sha256": images[0].sha256, "label": "bad"}
    assert app_client.post("/api/label", json=body).status_code == 403
    assert (
        app_client.post("/api/label", json=body, headers={"X-CSRF-Token": "wrong"}).status_code
        == 403
    )
    with db.read_session() as s:
        assert s.get(Consensus, (images[0].id, "room")).status is ConsensusStatus.OPEN  # type: ignore[union-attr]


def test_label_input_validation(app_client: TestClient, db: Database, world) -> None:  # type: ignore[no-untyped-def]
    alice, _, images = world
    login(app_client, db, alice)
    h = {"X-CSRF-Token": CSRF}
    assert (
        app_client.post(
            "/api/label", json={"sha256": images[0].sha256, "label": "garage"}, headers=h
        ).status_code
        == 422
    )
    assert (
        app_client.post(
            "/api/label", json={"sha256": "../../etc/passwd", "label": "bad"}, headers=h
        ).status_code
        == 422
    )
    assert (
        app_client.post(
            "/api/label", json={"sha256": "f" * 64, "label": "bad"}, headers=h
        ).status_code
        == 404
    )
    assert (
        app_client.post(
            "/api/label",
            json={"sha256": images[0].sha256, "label": "bad", "labeler_id": 99},
            headers=h,
        ).status_code
        == 422
    )
    assert app_client.get("/api/next?scheme=windows").status_code == 404
    body = app_client.post("/api/label", json={"label": 5}, headers=h).json()
    assert body == {"detail": "Invalid request"}  # no input echoed back


def test_annotation_bound_to_session_labeler(app_client: TestClient, db: Database, world) -> None:  # type: ignore[no-untyped-def]
    from sqlalchemy import select

    from app.models import Annotation

    alice, admin, images = world
    login(app_client, db, alice)
    app_client.post(
        "/api/label",
        json={"sha256": images[0].sha256, "label": "bad"},
        headers={"X-CSRF-Token": CSRF},
    )
    with db.read_session() as s:
        assert [a.labeler_id for a in s.scalars(select(Annotation))] == [alice.id]
    _ = admin


# --- images ---------------------------------------------------------------------------
def test_image_delivery_only_by_known_sha(
    app_client: TestClient, db: Database, world, settings: Settings
) -> None:  # type: ignore[no-untyped-def]
    alice, _, images = world
    login(app_client, db, alice)
    for bad in ("..%2f..%2fetc%2fpasswd", "not-a-sha", "A" * 64, "f" * 64):
        assert app_client.get(f"/img/{bad}").status_code in (404, 422)
    ok = app_client.get(f"/img/{images[0].sha256}")
    assert ok.status_code == 200 and ok.headers["cache-control"].startswith("private")


def test_image_path_escape_is_blocked(
    app_client: TestClient, db: Database, world, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    alice, _, images = world
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"top secret")
    with db.write_session() as s:
        from app.models import Image

        s.get(Image, images[0].id).path = "../secret.jpg"  # type: ignore[union-attr]
    login(app_client, db, alice)
    assert app_client.get(f"/img/{images[0].sha256}").status_code == 404


# --- admin -----------------------------------------------------------------------------
def test_admin_requires_admin(app_client: TestClient, db: Database, world) -> None:  # type: ignore[no-untyped-def]
    alice, admin, _ = world
    login(app_client, db, alice)
    for path in ("/admin", "/admin/conflicts", "/admin/export"):
        assert app_client.get(path, headers={"accept": "text/html"}).status_code == 403
    assert app_client.post("/admin/export", headers={"X-CSRF-Token": CSRF}).status_code == 403
    login(app_client, db, admin)
    for path in ("/admin", "/admin/conflicts", "/admin/export"):
        r = app_client.get(path)
        assert r.status_code == 200, path


def test_admin_by_verified_email_only(app_client: TestClient, db: Database) -> None:
    with db.write_session() as s:
        boss2 = make_labeler(s, "boss2", sub="other-sub", email="boss@example.test")
    login(app_client, db, boss2, email_verified=False)
    assert app_client.get("/admin").status_code == 403
    login(app_client, db, boss2, email_verified=True)
    assert app_client.get("/admin").status_code == 200


def test_conflict_resolution_flow(app_client: TestClient, db: Database, world, registry) -> None:  # type: ignore[no-untyped-def]
    alice, admin, images = world
    with db.write_session() as s:
        voters = [make_labeler(s, n) for n in ("v1", "v2", "v3")]
        for who, label in zip(voters, ("bad", "flur", "kueche"), strict=True):
            submit_label(
                s,
                labeler=who,
                sha256=images[0].sha256,
                scheme="room",
                label=label,
                registry=registry,
            )
    login(app_client, db, alice)
    form = {"csrf_token": CSRF, "sha256": images[0].sha256, "label": "flur"}
    assert app_client.post("/admin/conflicts/resolve", data=form).status_code == 403  # not admin
    login(app_client, db, admin)
    page = app_client.get("/admin/conflicts")
    assert images[0].sha256 in page.text and "v1" in page.text
    assert (
        app_client.post("/admin/conflicts/resolve", data={**form, "csrf_token": "bad"}).status_code
        == 403
    )
    assert app_client.post("/admin/conflicts/resolve", data=form).status_code == 303
    with db.read_session() as s:
        row = s.get(Consensus, (images[0].id, "room"))
        assert row is not None and (row.status, row.final_label) == (ConsensusStatus.FINAL, "flur")


def test_export_via_admin(
    app_client: TestClient, db: Database, world, registry, settings: Settings
) -> None:  # type: ignore[no-untyped-def]
    alice, admin, images = world
    with db.write_session() as s:
        a, b = make_labeler(s, "x"), make_labeler(s, "y")
        for who in (a, b):
            submit_label(
                s,
                labeler=who,
                sha256=images[0].sha256,
                scheme="room",
                label="bad",
                registry=registry,
            )
    login(app_client, db, admin)
    assert (
        app_client.post("/admin/export", data={"csrf_token": CSRF, "seed": "1"}).status_code == 303
    )
    page = app_client.get("/admin/export").text
    import re

    name = re.search(r"/admin/export/(\d{8}T\d{6}Z)/manifest\.jsonl", page).group(1)  # type: ignore[union-attr]
    r = app_client.get(f"/admin/export/{name}/manifest.jsonl")
    assert r.status_code == 200 and images[0].sha256 in r.text and "alice" not in r.text
    assert app_client.get(f"/admin/export/{name}/..%2f..%2fetc%2fpasswd").status_code in (404, 422)
    assert app_client.get("/admin/export/..%2f..%2fx/meta.json").status_code in (404, 422)


# --- headers / cookies / misc ---------------------------------------------------------------
def test_security_headers(app_client: TestClient) -> None:
    r = app_client.get("/healthz")
    h = r.headers
    assert r.json() == {"status": "ok"}
    assert "script-src 'self'" in h["content-security-policy"]
    assert "frame-ancestors 'none'" in h["content-security-policy"]
    assert "unsafe-inline" not in h["content-security-policy"]
    assert h["x-content-type-options"] == "nosniff"
    assert h["referrer-policy"] == "no-referrer"
    assert h["cache-control"] == "no-store"


def test_hsts_when_enabled(settings: Settings, db: Database) -> None:
    app = create_app(settings.model_copy(update={"hsts_enabled": True}), db)
    with TestClient(app) as client:
        assert "max-age" in client.get("/healthz").headers["strict-transport-security"]


def _stub_idp_metadata(app, settings: Settings) -> None:  # type: ignore[no-untyped-def]
    app.state.oauth.oidc.server_metadata.update(
        {
            "issuer": settings.oidc_issuer,
            "authorization_endpoint": "https://idp.example.test/authorize",
            "token_endpoint": "https://idp.example.test/token",
            "jwks_uri": "https://idp.example.test/jwks",
            "_loaded_at": 1,
        }
    )


def test_session_cookie_flags(settings: Settings, db: Database) -> None:
    app = create_app(settings.model_copy(update={"cookie_secure": True}), db)
    _stub_idp_metadata(app, settings)
    with TestClient(app, base_url="https://testserver", follow_redirects=False) as client:
        r = client.get("/auth/login")
        cookie = r.headers["set-cookie"].lower()
        assert cookie.startswith(SESSION_COOKIE)
        assert "httponly" in cookie and "secure" in cookie and "samesite=lax" in cookie
        assert "max-age=" in cookie


def test_no_stack_trace_leak(settings: Settings, db: Database, world, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    alice, _, _ = world
    app = create_app(settings, db)

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("secret internals")

    monkeypatch.setattr("app.routers.labeling.next_item", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.cookies.set(SESSION_COOKIE, session_cookie(login_session(alice, CSRF)))
        r = client.get("/api/next")
        assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
        assert "content-security-policy" in r.headers


def test_openapi_disabled_in_prod(settings: Settings, db: Database) -> None:
    app = create_app(
        settings.model_copy(
            update={"environment": "prod", "oidc_issuer": "https://idp.example.test"}
        ),
        db,
    )
    with TestClient(app) as client:
        assert (
            client.get("/docs").status_code == 404
            and client.get("/openapi.json").status_code == 404
        )


def test_rate_limit(settings: Settings, db: Database, world) -> None:  # type: ignore[no-untyped-def]
    alice, _, _ = world
    app = create_app(
        settings.model_copy(update={"rate_limit_enabled": True, "rate_limit_read": "3/minute"}), db
    )
    try:
        with TestClient(app) as client:
            client.cookies.set(SESSION_COOKIE, session_cookie(login_session(alice, CSRF)))
            codes = [client.get("/api/next").status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    finally:
        from app.security import limiter

        limiter.enabled = False
        limiter.reset()


def test_trusted_host(settings: Settings, db: Database) -> None:
    app = create_app(settings.model_copy(update={"allowed_hosts": ["labels.example.test"]}), db)
    with TestClient(app, base_url="http://evil.test") as client:
        assert client.get("/healthz").status_code == 400


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/label", "/label"),
        ("/admin/export?x=1", "/admin/export?x=1"),
        ("//evil.test", "/label"),
        ("https://evil.test", "/label"),
        ("/\\evil.test", "/label"),
        ("javascript:alert(1)", "/label"),
        ("/ok\r\nSet-Cookie: x=1", "/label"),
        (None, "/label"),
    ],
)
def test_safe_next_path(raw: str | None, expected: str) -> None:
    assert safe_next_path(raw) == expected


# --- OIDC redirect (PKCE, state, nonce) -------------------------------------------------------
def test_login_redirect_uses_pkce_state_nonce(settings: Settings, db: Database) -> None:
    app = create_app(settings, db)
    _stub_idp_metadata(app, settings)
    with TestClient(app, follow_redirects=False) as client:
        r = client.get("/auth/login?next=//evil.test")
    assert r.status_code == 302
    url = urlsplit(r.headers["location"])
    q = parse_qs(url.query)
    assert url.netloc == "idp.example.test"
    assert q["response_type"] == ["code"] and q["code_challenge_method"] == ["S256"]
    assert q["code_challenge"][0] and q["state"][0] and q["nonce"][0]
    assert q["redirect_uri"] == [settings.oidc_redirect_uri]
    assert q["scope"] == ["openid email profile"]


def test_callback_without_state_fails_generic(app_client: TestClient) -> None:
    r = app_client.get("/auth/callback?code=abc&state=forged", headers={"accept": "text/html"})
    assert r.status_code == 400 and "Anmeldung fehlgeschlagen" in r.text


def test_logout_requires_csrf(app_client: TestClient, db: Database, world) -> None:  # type: ignore[no-untyped-def]
    alice, _, _ = world
    login(app_client, db, alice)
    assert app_client.post("/auth/logout").status_code == 403
    assert app_client.post("/auth/logout", data={"csrf_token": CSRF}).status_code == 303


def test_settings_validation() -> None:
    with pytest.raises(ValueError, match="at least 32"):
        Settings(_env_file=None, secret_key="short")  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="Missing required"):
        Settings(_env_file=None).require_web_config()  # type: ignore[call-arg]
    _ = SECRET, ImageSource, Session


def test_empty_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_KEY", "")
    monkeypatch.setenv("SIGLIP_REVISION", "")
    monkeypatch.setenv("ADMIN_SUBS", "")
    monkeypatch.setenv("ROOM_LABELS", "bad, kueche ,flur")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.secret_key is None and s.siglip_revision is None and s.admin_subs == []
    assert s.room_labels == ["bad", "kueche", "flur"]
