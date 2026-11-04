"""Consensus / conflict logic.

Rules (single-choice schemes):
* a trusted source (``library`` pseudo-labeler) finalises immediately;
* two independent human labels that agree -> final (agreement 1.0);
* two that disagree -> needs_more (a third label is requested);
* three or more labels: a unique plurality of at least two votes wins (final),
  otherwise conflict -> admin review queue;
* ``unklar`` is an ordinary label and may become final;
* an admin decision is sticky: later recomputation never overrides it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.labelschemes import SchemeRegistry, default_registry
from app.models import (
    Annotation,
    Consensus,
    ConsensusStatus,
    Image,
    Labeler,
    utcnow,
)

REQUIRED_AGREEING_LABELS = 2


@dataclass(frozen=True)
class ConsensusResult:
    status: ConsensusStatus
    final_label: str | None
    agreement: float


def evaluate(human_labels: list[str], trusted_label: str | None = None) -> ConsensusResult:
    """Pure consensus function over the labels of *distinct* labelers (one each)."""
    if trusted_label is not None:
        return ConsensusResult(ConsensusStatus.FINAL, trusted_label, 1.0)

    total = len(human_labels)
    if total < REQUIRED_AGREEING_LABELS:
        return ConsensusResult(ConsensusStatus.OPEN, None, 0.0)

    counts = Counter(human_labels).most_common()
    top_label, top_count = counts[0]
    runner_up = counts[1][1] if len(counts) > 1 else 0

    if total == REQUIRED_AGREEING_LABELS:
        if top_count == total:
            return ConsensusResult(ConsensusStatus.FINAL, top_label, 1.0)
        return ConsensusResult(ConsensusStatus.NEEDS_MORE, None, top_count / total)

    if top_count >= REQUIRED_AGREEING_LABELS and top_count > runner_up:
        return ConsensusResult(ConsensusStatus.FINAL, top_label, top_count / total)
    return ConsensusResult(ConsensusStatus.CONFLICT, None, top_count / total)


def _get_or_create(session: Session, image_id: int, scheme: str) -> Consensus:
    row = session.get(Consensus, (image_id, scheme))
    if row is None:
        row = Consensus(image_id=image_id, scheme=scheme, status=ConsensusStatus.OPEN)
        session.add(row)
    return row


def recompute_consensus(
    session: Session,
    image_id: int,
    scheme: str,
    registry: SchemeRegistry | None = None,
) -> Consensus:
    """Recompute and persist the consensus row for one (image, scheme).

    Must run inside the same (write) transaction that modified the annotations.
    """
    scheme_def = (registry or default_registry()).get(scheme)
    if not scheme_def.is_single:
        raise NotImplementedError(f"Consensus for multi-label scheme '{scheme}' is not implemented")

    rows = session.execute(
        select(Annotation.label, Labeler)
        .join(Labeler, Labeler.id == Annotation.labeler_id)
        .where(Annotation.image_id == image_id, Annotation.scheme == scheme)
        .order_by(Annotation.id)
    ).all()

    human: dict[int, str] = {}
    trusted: str | None = None
    for label, labeler in rows:
        if labeler.is_library:
            trusted = label
        else:
            human[labeler.id] = label  # single-choice: the latest wins

    row = _get_or_create(session, image_id, scheme)
    result = evaluate(list(human.values()), trusted)

    if row.resolved_by_id is not None and row.final_label is not None:
        # Sticky admin decision; only refresh the agreement figure.
        row.agreement = result.agreement
    else:
        row.status = result.status
        row.final_label = result.final_label
        row.agreement = result.agreement
    row.updated_at = utcnow()
    session.flush()
    return row


def admin_resolve(
    session: Session,
    image_id: int,
    scheme: str,
    label: str,
    admin: Labeler,
    registry: SchemeRegistry | None = None,
) -> Consensus:
    """Manually set the final label (conflict queue)."""
    scheme_def = (registry or default_registry()).get(scheme)
    scheme_def.validate_label(label)
    row = recompute_consensus(session, image_id, scheme, registry)
    row.final_label = label
    row.status = ConsensusStatus.FINAL
    row.resolved_by_id = admin.id
    row.updated_at = utcnow()
    session.flush()
    return row


def ensure_consensus_rows(session: Session, image_ids: list[int], scheme: str) -> None:
    """Create ``open`` consensus rows for images lacking one (idempotent)."""
    if not image_ids:
        return
    existing = set(
        session.scalars(
            select(Consensus.image_id).where(
                Consensus.scheme == scheme, Consensus.image_id.in_(image_ids)
            )
        )
    )
    for image_id in image_ids:
        if image_id not in existing:
            session.add(Consensus(image_id=image_id, scheme=scheme, status=ConsensusStatus.OPEN))
    session.flush()


@dataclass(frozen=True)
class ConflictItem:
    sha256: str
    group_id: str
    votes: list[tuple[str, str]]  # (labeler display name, label)


def conflict_queue(session: Session, scheme: str, limit: int = 200) -> list[ConflictItem]:
    """Images with status=conflict (admin review queue), oldest first."""
    images = session.execute(
        select(Image)
        .join(Consensus, Consensus.image_id == Image.id)
        .where(Consensus.scheme == scheme, Consensus.status == ConsensusStatus.CONFLICT)
        .order_by(Consensus.updated_at, Image.id)
        .limit(limit)
    ).scalars()
    items: list[ConflictItem] = []
    for image in images:
        votes = session.execute(
            select(Labeler.display_name, Annotation.label)
            .join(Annotation, Annotation.labeler_id == Labeler.id)
            .where(Annotation.image_id == image.id, Annotation.scheme == scheme)
            .order_by(Annotation.id)
        ).all()
        items.append(
            ConflictItem(
                image.sha256, image.group_id, [(name or "labeler", label) for name, label in votes]
            )
        )
    return items
