"""Which image does a labeler see next?

1. only images this labeler has not labelled in the scheme, and whose status is not final
   (``conflict`` images belong to the admin queue and are not offered either);
2. tier ``needs_more`` before tier ``open``;
3. inside a tier: active learning – least confident / smallest top-2 margin first.
   Images without a prediction come last, in a stable pseudo-random order;
4. class-balanced choice from the most uncertain pool so the labeler is not fed one class
   only. The choice is a pure function of (labeler, labelling progress, DB state), so a
   reload resumes on the same image.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import ColumnElement, Select, and_, case, exists, func, literal, or_, select
from sqlalchemy.orm import Session

from app.models import Annotation, Consensus, ConsensusStatus, Image, ModelPrediction


@dataclass(frozen=True)
class Candidate:
    image_id: int
    sha256: str
    status: ConsensusStatus
    predicted_label: str | None
    confidence: float | None
    margin: float | None


@dataclass(frozen=True)
class Progress:
    done: int
    total: int


def latest_model_version(session: Session, scheme: str) -> str | None:
    return session.scalar(
        select(ModelPrediction.model_version)
        .where(ModelPrediction.scheme == scheme)
        .group_by(ModelPrediction.model_version)
        .order_by(func.max(ModelPrediction.created_at).desc(), ModelPrediction.model_version.desc())
        .limit(1)
    )


def _status_expr() -> ColumnElement[str]:
    return func.coalesce(Consensus.status, literal(ConsensusStatus.OPEN.value))


def _available_filter(labeler_id: int, scheme: str) -> list[ColumnElement[bool]]:
    already_labelled = exists().where(
        Annotation.image_id == Image.id,
        Annotation.labeler_id == labeler_id,
        Annotation.scheme == scheme,
    )
    return [
        ~already_labelled,
        or_(
            Consensus.status.is_(None),
            Consensus.status.in_([ConsensusStatus.OPEN, ConsensusStatus.NEEDS_MORE]),
        ),
    ]


def _base_query(labeler_id: int, scheme: str, version: str | None) -> Select[Any]:
    prediction_join = and_(
        ModelPrediction.image_id == Image.id,
        ModelPrediction.scheme == scheme,
        ModelPrediction.model_version == (version or ""),
    )
    query = (
        select(
            Image.id,
            Image.sha256,
            _status_expr().label("status"),
            ModelPrediction.label,
            ModelPrediction.confidence,
            ModelPrediction.margin,
        )
        .select_from(Image)
        .outerjoin(Consensus, and_(Consensus.image_id == Image.id, Consensus.scheme == scheme))
        .outerjoin(ModelPrediction, prediction_join)
        .where(*_available_filter(labeler_id, scheme))
    )
    return cast("Select[Any]", query)


def _stable_key(labeler_id: int) -> ColumnElement[int]:
    """Cheap deterministic pseudo-random ordering key (identical for repeated calls)."""
    return (Image.id * 7919 + labeler_id * 104729) % 1000003


def count_progress(session: Session, labeler_id: int, scheme: str) -> Progress:
    """done = images this labeler has labelled; total = done + images still offered to them."""
    done = (
        session.scalar(
            select(func.count(func.distinct(Annotation.image_id))).where(
                Annotation.labeler_id == labeler_id, Annotation.scheme == scheme
            )
        )
        or 0
    )
    remaining = (
        session.scalar(
            select(func.count())
            .select_from(Image)
            .outerjoin(Consensus, and_(Consensus.image_id == Image.id, Consensus.scheme == scheme))
            .where(*_available_filter(labeler_id, scheme))
        )
        or 0
    )
    return Progress(done=done, total=done + remaining)


def fetch_candidates(
    session: Session,
    *,
    labeler_id: int,
    scheme: str,
    exclude_ids: Collection[int] = (),
    strategy: str = "margin",
    pool_size: int = 40,
) -> list[Candidate]:
    """Top ``pool_size`` candidates of the best non-empty tier, most uncertain first."""
    version = latest_model_version(session, scheme)
    for tier in (ConsensusStatus.NEEDS_MORE, ConsensusStatus.OPEN):
        query = _base_query(labeler_id, scheme, version)
        if tier is ConsensusStatus.NEEDS_MORE:
            query = query.where(Consensus.status == ConsensusStatus.NEEDS_MORE)
        else:
            query = query.where(
                or_(Consensus.status.is_(None), Consensus.status == ConsensusStatus.OPEN)
            )
        if exclude_ids:
            query = query.where(Image.id.not_in(list(exclude_ids)))

        if strategy == "margin":
            uncertainty: Any = func.coalesce(ModelPrediction.margin, ModelPrediction.confidence)
        else:
            uncertainty = ModelPrediction.confidence
        query = query.order_by(
            case((ModelPrediction.id.is_(None), 1), else_=0),  # predictions first
            uncertainty.asc(),
            _stable_key(labeler_id),
            Image.id,
        ).limit(pool_size)

        rows = session.execute(query).all()
        if rows:
            return [
                Candidate(
                    image_id=r[0],
                    sha256=r[1],
                    status=ConsensusStatus(r[2]),
                    predicted_label=r[3],
                    confidence=r[4],
                    margin=r[5],
                )
                for r in rows
            ]
    return []


def choose_balanced(
    candidates: Sequence[Candidate], *, labeler_id: int, labels_done: int
) -> Candidate | None:
    """Pick class-balanced from an uncertainty-ordered pool, deterministically.

    Round-robins over the predicted classes present in the pool (sorted for stability);
    inside the chosen class the most uncertain candidate (pool order) wins.
    Candidates without a prediction are only used when nothing else is available.
    """
    if not candidates:
        return None
    predicted = [c for c in candidates if c.predicted_label is not None]
    if not predicted:
        return candidates[0]
    classes = sorted({c.predicted_label for c in predicted if c.predicted_label})
    offset = int(hashlib.sha256(f"{labeler_id}".encode()).hexdigest(), 16) % len(classes)
    target = classes[(labels_done + offset) % len(classes)]
    return next(c for c in predicted if c.predicted_label == target)


def next_image(
    session: Session,
    *,
    labeler_id: int,
    scheme: str,
    exclude_ids: Collection[int] = (),
    strategy: str = "margin",
    pool_size: int = 40,
) -> Candidate | None:
    candidates = fetch_candidates(
        session,
        labeler_id=labeler_id,
        scheme=scheme,
        exclude_ids=exclude_ids,
        strategy=strategy,
        pool_size=pool_size,
    )
    labels_done = count_progress(session, labeler_id, scheme).done
    return choose_balanced(candidates, labeler_id=labeler_id, labels_done=labels_done)


@dataclass(frozen=True)
class Suggestion:
    label: str
    confidence: float
    model_version: str


@dataclass(frozen=True)
class NextItem:
    candidate: Candidate | None
    suggestion: Suggestion | None
    progress: Progress


def next_item(
    session: Session,
    *,
    labeler_id: int,
    scheme: str,
    skipped_shas: Collection[str] = (),
    strategy: str = "margin",
    pool_size: int = 40,
) -> NextItem:
    """Everything the UI needs for the next screen: image, model suggestion and progress."""
    exclude_ids: list[int] = []
    if skipped_shas:
        exclude_ids = list(
            session.scalars(select(Image.id).where(Image.sha256.in_(list(skipped_shas))))
        )
    candidate = next_image(
        session,
        labeler_id=labeler_id,
        scheme=scheme,
        exclude_ids=exclude_ids,
        strategy=strategy,
        pool_size=pool_size,
    )
    suggestion: Suggestion | None = None
    if candidate and candidate.predicted_label and candidate.confidence is not None:
        version = latest_model_version(session, scheme) or ""
        suggestion = Suggestion(candidate.predicted_label, candidate.confidence, version)
    return NextItem(candidate, suggestion, count_progress(session, labeler_id, scheme))


def get_labeler_label(session: Session, labeler_id: int, sha256: str, scheme: str) -> str | None:
    """The label this labeler already gave an image (for 'back' navigation)."""
    return session.scalar(
        select(Annotation.label)
        .join(Image, Image.id == Annotation.image_id)
        .where(
            Image.sha256 == sha256,
            Annotation.labeler_id == labeler_id,
            Annotation.scheme == scheme,
        )
    )
