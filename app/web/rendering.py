"""Jinja2 rendering helper (shared context: CSRF token, current user)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from app.security import ensure_csrf_token

WEB_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))


def render(
    request: Request, name: str, context: dict[str, Any] | None = None, status_code: int = 200
) -> Response:
    ctx: dict[str, Any] = {"csrf_token": ensure_csrf_token(request.session)}
    ctx.update(context or {})
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)
