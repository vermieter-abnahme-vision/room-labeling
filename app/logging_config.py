"""Logging setup. Never log secrets, tokens, e-mail addresses or image content."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
    # httpx/authlib can log full URLs (incl. query strings) at INFO/DEBUG
    for noisy in ("httpx", "httpcore", "authlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
