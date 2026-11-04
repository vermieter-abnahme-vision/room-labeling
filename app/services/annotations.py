"""Writing annotations (idempotent) and keeping consensus in sync."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.labelschemes import SchemeRegistry, default_registry
from app.models import Annotation, Consensus, ConsensusStatus, Image, Labeler
from app.services.consensus import recompute_consensus


class ImageNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class SubmitResult:
    image_id: int
    changed: bool
    status: ConsensusStatus
    final_label: str | None


def get_image_by_sha(session: Session, sha256: str) -> Image:
    image = session.scalar(select(Image).where(Image.sha256 == sha256))
    if image is None:
        raise ImageNotFoundError(sha256)
    return image


def submit_label(
    session: Session,
    *,
    labeler: Labeler,
    sha256: str,
    scheme: str,
    label: str,
    registry: SchemeRegistry | None = None,
) -> SubmitResult:
    """Record ``label`` for ``labeler``. Re-submitting is an update, never a duplicate.

    Call inside a write session (``BEGIN IMMEDIATE``) so that the
    read-modify-write is atomic w.r.t. other labelers.
    """
    scheme_def = (registry or default_registry()).get(scheme)
    scheme_def.validate_label(label)
    image = get_image_by_sha(session, sha256)

    existing = list(
        session.scalars(
            select(Annotation).where(
                Annotation.image_id == image.id,
                Annotation.labeler_id == labeler.id,
                Annotation.scheme == scheme,
            )
        )
    )
    changed = False
    if scheme_def.is_single:
        if len(existing) == 1 and existing[0].label == label:
            pass  # idempotent re-submit
        else:
            if existing:
                session.execute(
                    delete(Annotation).where(Annotation.id.in_([a.id for a in existing]))
                )
            session.add(
                Annotation(image_id=image.id, labeler_id=labeler.id, scheme=scheme, label=label)
            )
            changed = True
    elif label not in {a.label for a in existing}:
        session.add(
            Annotation(image_id=image.id, labeler_id=labeler.id, scheme=scheme, label=label)
        )
        changed = True
    session.flush()

    row: Consensus | None = session.get(Consensus, (image.id, scheme))
    if changed or row is None:
        row = recompute_consensus(session, image.id, scheme, registry)
    return SubmitResult(image.id, changed, row.status, row.final_label)
