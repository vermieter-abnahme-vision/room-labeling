"""Application settings (pydantic-settings). All configuration comes from ENV."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

MIN_SECRET_KEY_BYTES = 32


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    """Typed settings. Never hardcode secrets; everything is read from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    environment: Literal["dev", "prod"] = "prod"

    # --- Storage -----------------------------------------------------------------------
    database_url: str = "sqlite:///./var/db/labeling.db"
    images_dir: Path = Path("./var/images")
    images_in_dir: Path = Path("./images_in")
    videos_in_dir: Path = Path("./videos_in")
    embed_cache_dir: Path = Path("./var/embed_cache")
    models_dir: Path = Path("./var/models")
    export_dir: Path = Path("./var/exports")

    # --- Label schemes -----------------------------------------------------------------
    room_labels: CsvList = ["bad", "kueche", "wohnzimmer", "schlafzimmer", "flur", "unklar"]

    # --- Sampling ----------------------------------------------------------------------
    sampling_strategy: Literal["margin", "least_confidence"] = "margin"
    sampling_pool_size: int = Field(default=40, ge=1)

    # --- Ingestion ---------------------------------------------------------------------
    video_fps: float = Field(default=1.0, gt=0, le=30)
    frame_max_side: int = Field(default=1280, ge=64)
    max_image_bytes: int = Field(default=25 * 1024 * 1024, gt=0)
    max_video_bytes: int = Field(default=8 * 1024**3, gt=0)
    max_image_pixels: int = Field(default=100_000_000, gt=0)
    ffmpeg_bin: str = "ffmpeg"
    ffmpeg_timeout_seconds: int = Field(default=3600, gt=0)

    # --- Model -------------------------------------------------------------------------
    siglip_model: str = "google/siglip-base-patch16-224"
    siglip_revision: str | None = None  # pin a commit hash for reproducibility

    # --- Export ------------------------------------------------------------------------
    export_salt: str | None = None  # salt for pseudonymised labeler ids; falls back to SECRET_KEY

    # --- Web / security ----------------------------------------------------------------
    secret_key: str | None = None
    cookie_secure: bool = True
    session_ttl_seconds: int = Field(default=8 * 3600, gt=0)
    allowed_hosts: CsvList = ["*"]
    hsts_enabled: bool = True
    rate_limit_enabled: bool = True
    rate_limit_login: str = "10/minute"
    rate_limit_submit: str = "120/minute"
    rate_limit_read: str = "240/minute"

    # --- OIDC --------------------------------------------------------------------------
    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_redirect_uri: str | None = None
    oidc_allowed_redirect_uris: CsvList = []
    oidc_scopes: str = "openid email profile"
    oidc_require_email_verified: bool = True
    admin_subs: CsvList = []
    admin_emails: CsvList = []

    _split_validator = field_validator(
        "room_labels",
        "allowed_hosts",
        "oidc_allowed_redirect_uris",
        "admin_subs",
        "admin_emails",
        mode="before",
    )(_split_csv)

    @field_validator("database_url")
    @classmethod
    def _sqlite_only(cls, value: str) -> str:
        if not value.startswith("sqlite"):
            raise ValueError("Only SQLite is supported (WAL mode, few labelers).")
        return value

    @field_validator("room_labels")
    @classmethod
    def _labels_valid(cls, value: list[str]) -> list[str]:
        if len(value) < 2 or len(set(value)) != len(value):
            raise ValueError("ROOM_LABELS needs at least two unique labels.")
        for label in value:
            if not label.replace("_", "").isalnum() or label != label.lower():
                raise ValueError(f"Invalid label '{label}': use lowercase [a-z0-9_] only.")
        return value

    @model_validator(mode="after")
    def _check_secret(self) -> Settings:
        if self.secret_key is not None and len(self.secret_key.encode()) < MIN_SECRET_KEY_BYTES:
            raise ValueError(f"SECRET_KEY must be at least {MIN_SECRET_KEY_BYTES} bytes.")
        return self

    # ---------------------------------------------------------------------------------
    @property
    def effective_redirect_allowlist(self) -> set[str]:
        allowed = set(self.oidc_allowed_redirect_uris)
        if self.oidc_redirect_uri:
            allowed.add(self.oidc_redirect_uri)
        return allowed

    def require_web_config(self) -> None:
        """Fail fast if the web app is started without mandatory auth configuration."""
        missing = [
            name.upper()
            for name in (
                "secret_key",
                "oidc_issuer",
                "oidc_client_id",
                "oidc_client_secret",
                "oidc_redirect_uri",
            )
            if not getattr(self, name)
        ]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        if self.oidc_redirect_uri not in self.effective_redirect_allowlist:
            raise RuntimeError("OIDC_REDIRECT_URI is not in the redirect allowlist.")
        if "openid" not in self.oidc_scopes.split():
            raise RuntimeError("OIDC_SCOPES must include 'openid'.")
        if self.environment == "prod" and not (self.oidc_issuer or "").startswith("https://"):
            raise RuntimeError("OIDC_ISSUER must use https:// in production.")

    @property
    def pseudonym_salt(self) -> str:
        salt = self.export_salt or self.secret_key
        if not salt:
            raise RuntimeError("Set EXPORT_SALT (or SECRET_KEY) to pseudonymise labelers.")
        return salt


@lru_cache
def get_settings() -> Settings:
    return Settings()
