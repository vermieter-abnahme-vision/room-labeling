"""Shared CLI plumbing: settings, DB, logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.db import Database
from app.labelschemes import SchemeRegistry
from app.logging_config import configure_logging


@dataclass(frozen=True)
class Context:
    settings: Settings
    db: Database
    registry: SchemeRegistry


def bootstrap(verbose: bool = False) -> Context:
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    settings = get_settings()
    return Context(
        settings, Database(settings.database_url), SchemeRegistry.from_settings(settings)
    )
