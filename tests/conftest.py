from __future__ import annotations

import base64
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from itsdangerous import TimestampSigner
from PIL import Image as PILImage
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import Database
from app.labelschemes import SchemeRegistry
from app.models import Base, Image, ImageSource, Labeler
from app.services.consensus import ensure_consensus_rows

SECRET = "unit-test-secret-" + "x" * 32  # not a real secret


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        environment="dev",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        images_dir=tmp_path / "images",
        images_in_dir=tmp_path / "images_in",
        videos_in_dir=tmp_path / "videos_in",
        embed_cache_dir=tmp_path / "embed",
        models_dir=tmp_path / "models",
        export_dir=tmp_path / "exports",
        secret_key=SECRET,
        cookie_secure=False,
        hsts_enabled=False,
        rate_limit_enabled=False,
        oidc_issuer="https://idp.example.test/application/o/rooms/",
        oidc_client_id="client-id",
        oidc_client_secret="client-secret",
        oidc_redirect_uri="http://testserver/auth/callback",
        admin_subs=["admin-sub"],
        admin_emails=["boss@example.test"],
    )


@pytest.fixture
def registry(settings: Settings) -> SchemeRegistry:
    return SchemeRegistry.from_settings(settings)


@pytest.fixture
def db(settings: Settings) -> Iterator[Database]:
    database = Database(settings.database_url)
    Base.metadata.create_all(database.engine)
    yield database
    database.dispose()


@pytest.fixture
def session(db: Database) -> Iterator[Session]:
    with db.write_session() as s:
        yield s


def make_labeler(
    session: Session, name: str, *, sub: str | None = None, email: str | None = None
) -> Labeler:
    labeler = Labeler(oidc_sub=sub or f"sub-{name}", email=email, display_name=name, is_admin=False)
    session.add(labeler)
    session.flush()
    return labeler


def make_image(
    session: Session,
    n: int,
    *,
    group: str = "g1",
    source: ImageSource = ImageSource.UPLOAD_VIDEO,
    with_consensus: bool = True,
) -> Image:
    sha = f"{n:064x}"
    image = Image(sha256=sha, path=f"{sha[:2]}/{sha}.jpg", source=source, group_id=group)
    session.add(image)
    session.flush()
    if with_consensus:
        ensure_consensus_rows(session, [image.id], "room")
    return image


def write_jpeg(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = PILImage.new("RGB", size, color)
    # Add a deterministic pattern so different colours -> different bytes
    img.putpixel((0, 0), (color[1], color[2], color[0]))
    img.save(path, "JPEG", quality=90)
    return path


def session_cookie(data: dict[str, object]) -> str:
    """Forge a valid signed Starlette session cookie (tests only; signs with the test secret)."""
    payload = base64.b64encode(json.dumps(data).encode())
    return TimestampSigner(SECRET).sign(payload).decode()


def login_session(
    labeler: Labeler, csrf: str = "c" * 43, email_verified: bool = True
) -> dict[str, object]:
    return {
        "labeler_id": labeler.id,
        "sub": labeler.oidc_sub,
        "email_verified": email_verified,
        "iat": int(time.time()),
        "csrf": csrf,
    }
