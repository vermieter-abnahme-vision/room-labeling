"""Authenticated image delivery: sha256 lookup -> server-side resolved path only."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from sqlalchemy import select
from starlette.responses import FileResponse

from app.config import Settings
from app.db import Database
from app.deps import CurrentUser, current_user, get_database, get_settings_dep
from app.models import Image
from app.schemas import SHA256_PATTERN
from app.security import limiter, rate_limits

logger = logging.getLogger(__name__)
router = APIRouter(tags=["images"])

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@router.get("/img/{sha256}", include_in_schema=False)
@limiter.limit(lambda: rate_limits.read)
def get_image(
    request: Request,
    sha256: Annotated[str, Path(pattern=SHA256_PATTERN)],
    _user: CurrentUser = Depends(current_user),
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_settings_dep),
) -> FileResponse:
    with database.read_session() as session:
        rel = session.scalar(select(Image.path).where(Image.sha256 == sha256))
    if rel is None:
        raise HTTPException(status_code=404, detail="Not found")
    root = settings.images_dir.resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        logger.error("image path outside store or missing for a known sha256")
        raise HTTPException(status_code=404, detail="Not found")
    media_type = _MEDIA_TYPES.get(target.suffix.lower(), "application/octet-stream")
    return FileResponse(
        target,
        media_type=media_type,
        headers={"Cache-Control": "private, max-age=86400, immutable"},
    )
