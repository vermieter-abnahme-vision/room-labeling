"""Admin views: stats, agreement, conflict queue, datapack export. All routes require admin."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Path, Request
from starlette.responses import FileResponse, RedirectResponse, Response

from app.config import Settings
from app.db import Database
from app.deps import CurrentUser, get_database, get_registry, get_settings_dep, require_admin
from app.labelschemes import ROOM_SCHEME, InvalidLabelError, SchemeRegistry
from app.schemas import LABEL_PATTERN, SHA256_PATTERN
from app.security import limiter, rate_limits, verify_csrf
from app.services import consensus, export, stats
from app.services.annotations import ImageNotFoundError, get_image_by_sha
from app.web.rendering import render

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("", include_in_schema=False)
def dashboard(
    request: Request,
    user: CurrentUser = Depends(require_admin),
    database: Database = Depends(get_database),
    registry: SchemeRegistry = Depends(get_registry),
) -> Response:
    scheme = registry.get(ROOM_SCHEME)
    with database.read_session() as session:
        context = {
            "user": user,
            "page": "admin",
            "scheme": scheme,
            "progress": stats.progress_stats(session, scheme.name),
            "distribution": stats.label_distribution(session, scheme.name, scheme.labels),
            "agreement": stats.agreement_stats(session, scheme.name),
            "labelers": stats.labeler_stats(session, scheme.name),
        }
    return render(request, "admin_index.html", context)


@router.get("/conflicts", include_in_schema=False)
def conflicts_page(
    request: Request,
    user: CurrentUser = Depends(require_admin),
    database: Database = Depends(get_database),
    registry: SchemeRegistry = Depends(get_registry),
) -> Response:
    scheme = registry.get(ROOM_SCHEME)
    with database.read_session() as session:
        items = consensus.conflict_queue(session, scheme.name)
    return render(
        request,
        "admin_conflicts.html",
        {"user": user, "page": "conflicts", "scheme": scheme, "items": items},
    )


@router.post("/conflicts/resolve", include_in_schema=False, dependencies=[Depends(verify_csrf)])
@limiter.limit(lambda: rate_limits.submit)
def resolve_conflict(
    request: Request,
    sha256: Annotated[str, Form(pattern=SHA256_PATTERN)],
    label: Annotated[str, Form(pattern=LABEL_PATTERN)],
    user: CurrentUser = Depends(require_admin),
    database: Database = Depends(get_database),
    registry: SchemeRegistry = Depends(get_registry),
) -> Response:
    try:
        with database.write_session() as session:
            image = get_image_by_sha(session, sha256)
            consensus.admin_resolve(session, image.id, ROOM_SCHEME, label, user.labeler, registry)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found") from None
    except InvalidLabelError:
        raise HTTPException(status_code=422, detail="Invalid label") from None
    logger.info("admin id=%s resolved a conflict", user.labeler.id)
    return RedirectResponse("/admin/conflicts", status_code=303)


@router.get("/export", include_in_schema=False)
def export_page(
    request: Request,
    user: CurrentUser = Depends(require_admin),
    settings: Settings = Depends(get_settings_dep),
) -> Response:
    return render(
        request,
        "admin_export.html",
        {"user": user, "page": "export", "exports": export.list_exports(settings.export_dir)},
    )


@router.post("/export", include_in_schema=False, dependencies=[Depends(verify_csrf)])
@limiter.limit("6/minute")
def create_export(
    request: Request,
    seed: Annotated[int, Form(ge=0, le=2**31 - 1)] = 0,
    keep_previous_splits: Annotated[bool, Form()] = False,
    library_split: Annotated[str, Form(pattern=r"^(train|group)$")] = "train",
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_settings_dep),
    registry: SchemeRegistry = Depends(get_registry),
) -> Response:
    previous = None
    if keep_previous_splits:
        for name in export.list_exports(settings.export_dir):
            path = export.resolve_export_file(settings.export_dir, name, "manifest.jsonl")
            if path:
                previous = export.assignments_from_manifest(path)
                break
    options = export.ExportOptions(
        scheme=ROOM_SCHEME,
        seed=seed,
        library_split="group" if library_split == "group" else "train",
        pseudonym_salt=settings.pseudonym_salt,
        previous_assignments=previous,
    )
    with database.read_session() as session:
        manifest = export.build_manifest(session, options, registry)
    target = export.write_export(manifest, settings.export_dir)
    logger.info("datapack exported: %s (%d images)", target.name, len(manifest.rows))
    return RedirectResponse("/admin/export", status_code=303)


@router.get("/export/{name}/{filename}", include_in_schema=False)
def download_export(
    name: Annotated[str, Path(pattern=r"^[0-9]{8}T[0-9]{6}Z(-[0-9]+)?$")],
    filename: Annotated[str, Path(pattern=r"^(manifest\.jsonl|meta\.json)$")],
    settings: Settings = Depends(get_settings_dep),
) -> FileResponse:
    path = export.resolve_export_file(settings.export_dir, name, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Not found")
    media = "application/x-ndjson" if filename.endswith(".jsonl") else "application/json"
    return FileResponse(path, media_type=media, filename=f"{name}-{filename}")
