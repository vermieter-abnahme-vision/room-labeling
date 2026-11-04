"""Shared CLI plumbing: settings, DB, logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory

from app.config import Settings, get_settings
from app.db import Database
from app.labelschemes import SchemeRegistry
from app.logging_config import configure_logging


@dataclass(frozen=True)
class Context:
    settings: Settings
    db: Database
    registry: SchemeRegistry


def ensure_schema_current(db: Database) -> None:
    """Exit with a clear hint if the DB is not migrated to the latest Alembic revision."""
    root = Path(__file__).resolve().parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "alembic"))
    head = ScriptDirectory.from_config(config).get_current_head()
    with db.engine.connect() as conn:
        current = MigrationContext.configure(conn).get_current_revision()
    if current != head:
        raise SystemExit(
            f"Database schema is not up to date (current={current}, expected={head}). "
            "Run `make migrate` (or `alembic upgrade head`) first."
        )


def bootstrap(verbose: bool = False, check_schema: bool = True) -> Context:
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    settings = get_settings()
    db = Database(settings.database_url)
    if check_schema:
        ensure_schema_current(db)
    return Context(settings, db, SchemeRegistry.from_settings(settings))
