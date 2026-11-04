"""Labeling HTTP API + pages. Thin layer: validates input, calls services, maps to DTOs."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import RedirectResponse, Response

from app.config import Settings
from app.db import Database
from app.deps import (
    CurrentUser,
    current_user,
    get_database,
    get_registry,
    get_settings_dep,
)
from app.labelschemes import LabelScheme, SchemeRegistry, UnknownSchemeError
from app.schemas import (
    SHA256_PATTERN,
    ImageDTO,
    LabelOption,
    NextResponse,
    PreviousLabelResponse,
    ProgressDTO,
    SubmitRequest,
    SubmitResponse,
    SuggestionDTO,
)
from app.security import limiter, rate_limits, verify_csrf
from app.services.annotations import ImageNotFoundError, submit_label
from app.services.sampling import get_labeler_label, next_item
from app.web.rendering import render

router = APIRouter(tags=["labeling"])
_SHA = re.compile(SHA256_PATTERN)
MAX_SKIPPED = 200


def _scheme(registry: SchemeRegistry, name: str) -> LabelScheme:
    try:
        return registry.get(name)
    except UnknownSchemeError:
        raise HTTPException(status_code=404, detail="Unknown scheme") from None


def _parse_skipped(raw: list[str]) -> list[str]:
    cleaned = [s for s in raw if _SHA.match(s)]
    return cleaned[:MAX_SKIPPED]


def build_next_response(
    *,
    database: Database,
    settings: Settings,
    scheme: LabelScheme,
    labeler_id: int,
    skipped: list[str],
) -> NextResponse:
    with database.read_session() as session:
        item = next_item(
            session,
            labeler_id=labeler_id,
            scheme=scheme.name,
            skipped_shas=skipped,
            strategy=settings.sampling_strategy,
            pool_size=settings.sampling_pool_size,
        )
    suggestion = None
    if item.suggestion:
        suggestion = SuggestionDTO(
            label=item.suggestion.label,
            display=scheme.display_name(item.suggestion.label),
            confidence=round(item.suggestion.confidence, 4),
            model_version=item.suggestion.model_version,
            zero_shot="zeroshot" in item.suggestion.model_version,
        )
    image = None
    if item.candidate:
        sha = item.candidate.sha256
        image = ImageDTO(sha256=sha, url=f"/img/{sha}")
    return NextResponse(
        scheme=scheme.name,
        labels=[LabelOption(key=k, display=scheme.display_name(k)) for k in scheme.labels],
        image=image,
        suggestion=suggestion,
        progress=ProgressDTO(done=item.progress.done, total=item.progress.total),
    )


@router.get("/", include_in_schema=False)
def index() -> Response:
    return RedirectResponse("/label", status_code=303)


@router.get("/label", include_in_schema=False)
def label_page(request: Request, user: CurrentUser = Depends(current_user)) -> Response:
    return render(request, "label.html", {"user": user, "page": "label"})


@router.get("/api/next", response_model=NextResponse)
@limiter.limit(lambda: rate_limits.read)
def api_next(
    request: Request,
    scheme: str = Query(default="room", pattern=r"^[a-z0-9_]{1,64}$"),
    skipped: str = Query(default="", max_length=200 * 65),
    user: CurrentUser = Depends(current_user),
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_settings_dep),
    registry: SchemeRegistry = Depends(get_registry),
) -> NextResponse:
    return build_next_response(
        database=database,
        settings=settings,
        scheme=_scheme(registry, scheme),
        labeler_id=user.labeler.id,
        skipped=_parse_skipped(skipped.split(",") if skipped else []),
    )


@router.post("/api/label", response_model=SubmitResponse, dependencies=[Depends(verify_csrf)])
@limiter.limit(lambda: rate_limits.submit)
def api_label(
    request: Request,
    body: SubmitRequest,
    user: CurrentUser = Depends(current_user),
    database: Database = Depends(get_database),
    settings: Settings = Depends(get_settings_dep),
    registry: SchemeRegistry = Depends(get_registry),
) -> SubmitResponse:
    scheme = _scheme(registry, body.scheme)
    if body.label not in scheme.labels:
        raise HTTPException(status_code=422, detail="Invalid label")
    try:
        # The annotation is always bound to the authenticated labeler, never to a client value.
        with database.write_session() as session:
            submit_label(
                session,
                labeler=user.labeler,
                sha256=body.sha256,
                scheme=scheme.name,
                label=body.label,
                registry=registry,
            )
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found") from None
    nxt = build_next_response(
        database=database,
        settings=settings,
        scheme=scheme,
        labeler_id=user.labeler.id,
        skipped=_parse_skipped(body.skipped),
    )
    return SubmitResponse(next=nxt)


@router.get("/api/my-label/{sha256}", response_model=PreviousLabelResponse)
@limiter.limit(lambda: rate_limits.read)
def api_my_label(
    request: Request,
    sha256: str,
    scheme: str = Query(default="room", pattern=r"^[a-z0-9_]{1,64}$"),
    user: CurrentUser = Depends(current_user),
    database: Database = Depends(get_database),
    registry: SchemeRegistry = Depends(get_registry),
) -> PreviousLabelResponse:
    if not _SHA.match(sha256):
        raise HTTPException(status_code=404, detail="Not found")
    scheme_def = _scheme(registry, scheme)
    with database.read_session() as session:
        label = get_labeler_label(session, user.labeler.id, sha256, scheme_def.name)
    return PreviousLabelResponse(label=label)
