"""Progress and inter-annotator agreement (library pseudo-labels are excluded)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import (
    LIBRARY_LABELER_SUB,
    Annotation,
    Consensus,
    ConsensusStatus,
    Image,
    Labeler,
)

# --------------------------------------------------------------------------------------
# Pure metrics
# --------------------------------------------------------------------------------------


def percent_agreement(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters; None if undefined (no data or no label variance)."""
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    count_a = Counter(a for a, _ in pairs)
    count_b = Counter(b for _, b in pairs)
    expected = sum(count_a[k] * count_b[k] for k in count_a) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def krippendorff_alpha_nominal(units: Sequence[Sequence[str]]) -> float | None:
    """Krippendorff's alpha (nominal) over units with a variable number of ratings.

    ``units`` holds, per image, the labels given by distinct labelers. Units with fewer than
    two ratings are not pairable and ignored. Returns None when nothing is pairable.
    """
    coincidences: defaultdict[tuple[str, str], float] = defaultdict(float)
    for unit in units:
        m = len(unit)
        if m < 2:
            continue
        for i, a in enumerate(unit):
            for j, b in enumerate(unit):
                if i != j:
                    coincidences[(a, b)] += 1 / (m - 1)
    total = sum(coincidences.values())
    if total <= 1:
        return None
    marginals: defaultdict[str, float] = defaultdict(float)
    for (a, _), value in coincidences.items():
        marginals[a] += value
    observed_disagreement = sum(v for (a, b), v in coincidences.items() if a != b)
    labels = list(marginals)
    expected_disagreement = sum(
        marginals[a] * marginals[b] for a in labels for b in labels if a != b
    )
    if expected_disagreement == 0:
        return 1.0  # every rating identical -> perfect agreement
    return 1.0 - (total - 1) * observed_disagreement / expected_disagreement


# --------------------------------------------------------------------------------------
# DB-backed stats
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PairAgreement:
    labeler_a: str
    labeler_b: str
    overlap: int
    agreement: float | None
    kappa: float | None


@dataclass(frozen=True)
class AgreementStats:
    pairs: list[PairAgreement]
    krippendorff_alpha: float | None
    overlapping_images: int


@dataclass(frozen=True)
class ProgressStats:
    total: int
    final: int
    needs_more: int
    open: int
    conflict: int


def progress_stats(session: Session, scheme: str) -> ProgressStats:
    total = session.scalar(select(func.count()).select_from(Image)) or 0
    rows = session.execute(
        select(Consensus.status, func.count())
        .where(Consensus.scheme == scheme)
        .group_by(Consensus.status)
    ).all()
    counts = {ConsensusStatus(status): n for status, n in rows}
    with_row = sum(counts.values())
    return ProgressStats(
        total=total,
        final=counts.get(ConsensusStatus.FINAL, 0),
        needs_more=counts.get(ConsensusStatus.NEEDS_MORE, 0),
        open=counts.get(ConsensusStatus.OPEN, 0) + max(total - with_row, 0),
        conflict=counts.get(ConsensusStatus.CONFLICT, 0),
    )


def label_distribution(session: Session, scheme: str, labels: Sequence[str]) -> Mapping[str, int]:
    rows = session.execute(
        select(Consensus.final_label, func.count())
        .where(Consensus.scheme == scheme, Consensus.status == ConsensusStatus.FINAL)
        .group_by(Consensus.final_label)
    ).all()
    found = {label: n for label, n in rows if label is not None}
    result = {label: found.pop(label, 0) for label in labels}
    result.update(found)  # labels removed from config but still present in data
    return result


def _human_annotations(session: Session, scheme: str) -> dict[int, dict[str, str]]:
    """{image_id: {labeler_display_key: label}} for human labelers only."""
    rows = session.execute(
        select(Annotation.image_id, Labeler.id, Labeler.display_name, Annotation.label)
        .join(Labeler, Labeler.id == Annotation.labeler_id)
        .where(Annotation.scheme == scheme, Labeler.oidc_sub != LIBRARY_LABELER_SUB)
    ).all()
    by_image: dict[int, dict[str, str]] = {}
    for image_id, labeler_id, display_name, label in rows:
        key = f"{display_name or 'labeler'}#{labeler_id}"
        by_image.setdefault(image_id, {})[key] = label
    return by_image


def agreement_stats(session: Session, scheme: str) -> AgreementStats:
    by_image = _human_annotations(session, scheme)
    labelers = sorted({key for votes in by_image.values() for key in votes})
    pairs: list[PairAgreement] = []
    for a, b in combinations(labelers, 2):
        shared = [(v[a], v[b]) for v in by_image.values() if a in v and b in v]
        pairs.append(
            PairAgreement(a, b, len(shared), percent_agreement(shared), cohens_kappa(shared))
        )
    units = [list(votes.values()) for votes in by_image.values()]
    return AgreementStats(
        pairs=pairs,
        krippendorff_alpha=krippendorff_alpha_nominal(units),
        overlapping_images=sum(1 for u in units if len(u) >= 2),
    )


@dataclass(frozen=True)
class LabelerStats:
    display_name: str
    labels_given: int


def labeler_stats(session: Session, scheme: str) -> list[LabelerStats]:
    rows = session.execute(
        select(Labeler.display_name, func.count(Annotation.id))
        .join(Annotation, Annotation.labeler_id == Labeler.id)
        .where(Annotation.scheme == scheme, Labeler.oidc_sub != LIBRARY_LABELER_SUB)
        .group_by(Labeler.id)
        .order_by(func.count(Annotation.id).desc())
    ).all()
    return [LabelerStats(name or "labeler", n) for name, n in rows]
