from __future__ import annotations

from sqlalchemy.orm import Session

from app.labelschemes import SchemeRegistry
from app.models import ConsensusStatus, ModelPrediction, utcnow
from app.services.annotations import submit_label
from app.services.sampling import (
    Candidate,
    choose_balanced,
    count_progress,
    fetch_candidates,
    next_image,
    next_item,
)
from tests.conftest import make_image, make_labeler


def _predict(
    session: Session,
    image_id: int,
    label: str,
    confidence: float,
    margin: float,
    version: str = "v1",
) -> None:
    session.add(
        ModelPrediction(
            image_id=image_id,
            scheme="room",
            label=label,
            confidence=confidence,
            margin=margin,
            model_version=version,
            created_at=utcnow(),
        )
    )
    session.flush()


def _label(session, registry, who, image, label):  # type: ignore[no-untyped-def]
    submit_label(
        session, labeler=who, sha256=image.sha256, scheme="room", label=label, registry=registry
    )


def test_excludes_own_labels_and_final(session: Session, registry: SchemeRegistry) -> None:
    a, b, c = (make_labeler(session, n) for n in "abc")
    done_by_a, final_img, fresh = (make_image(session, i) for i in (1, 2, 3))
    _label(session, registry, a, done_by_a, "bad")
    _label(session, registry, b, final_img, "bad")
    _label(session, registry, c, final_img, "bad")  # final
    got = next_image(session, labeler_id=a.id, scheme="room")
    assert got is not None and got.sha256 in {final_img.sha256, fresh.sha256}
    assert got.sha256 == fresh.sha256  # a already did #1; #2 is final
    assert next_image(session, labeler_id=a.id, scheme="room", exclude_ids=[fresh.id]) is None


def test_needs_more_has_priority_over_open(session: Session, registry: SchemeRegistry) -> None:
    a, b, c = (make_labeler(session, n) for n in "abc")
    disputed, fresh = make_image(session, 1), make_image(session, 2)
    _label(session, registry, a, disputed, "bad")
    _label(session, registry, b, disputed, "flur")
    # c has labelled nothing: the disputed (needs_more) image must come first
    got = next_image(session, labeler_id=c.id, scheme="room")
    assert (
        got is not None
        and got.sha256 == disputed.sha256
        and got.status is ConsensusStatus.NEEDS_MORE
    )
    # a and b already voted on it -> they get the fresh one
    assert next_image(session, labeler_id=a.id, scheme="room").sha256 == fresh.sha256  # type: ignore[union-attr]


def test_conflict_images_are_not_offered(session: Session, registry: SchemeRegistry) -> None:
    a, b, c, d = (make_labeler(session, n) for n in "abcd")
    img = make_image(session, 1)
    for who, label in ((a, "bad"), (b, "flur"), (c, "kueche")):
        _label(session, registry, who, img, label)
    assert next_image(session, labeler_id=d.id, scheme="room") is None


def test_active_learning_orders_by_margin(session: Session) -> None:
    a = make_labeler(session, "a")
    sure, unsure, unknown = (make_image(session, i) for i in (1, 2, 3))
    _predict(session, sure.id, "bad", 0.95, 0.9)
    _predict(session, unsure.id, "bad", 0.40, 0.05)
    cands = fetch_candidates(session, labeler_id=a.id, scheme="room")
    assert [c.sha256 for c in cands] == [
        unsure.sha256,
        sure.sha256,
        unknown.sha256,
    ]  # no prediction last


def test_least_confidence_strategy(session: Session) -> None:
    a = make_labeler(session, "a")
    x, y = make_image(session, 1), make_image(session, 2)
    _predict(session, x.id, "bad", 0.5, 0.01)
    _predict(session, y.id, "bad", 0.4, 0.5)
    margin = fetch_candidates(session, labeler_id=a.id, scheme="room", strategy="margin")
    conf = fetch_candidates(session, labeler_id=a.id, scheme="room", strategy="least_confidence")
    assert margin[0].sha256 == x.sha256
    assert conf[0].sha256 == y.sha256


def test_only_latest_model_version_counts(session: Session) -> None:
    a = make_labeler(session, "a")
    x, y = make_image(session, 1), make_image(session, 2)
    _predict(session, x.id, "bad", 0.1, 0.01, version="old")
    _predict(session, y.id, "bad", 0.9, 0.8, version="old")
    # newer version flips the uncertainty order
    _predict(session, x.id, "bad", 0.9, 0.8, version="new")
    _predict(session, y.id, "bad", 0.1, 0.01, version="new")
    # ensure 'new' is newer
    for p in session.query(ModelPrediction).filter_by(model_version="new"):
        p.created_at = utcnow().replace(year=2099)
    session.flush()
    cands = fetch_candidates(session, labeler_id=a.id, scheme="room")
    assert cands[0].sha256 == y.sha256


def test_deterministic_resume_and_class_balance() -> None:
    cands = [
        Candidate(i, f"{i:064x}", ConsensusStatus.OPEN, label, 0.5, 0.1 + i / 100)
        for i, label in enumerate(["bad", "bad", "bad", "flur", "flur", "kueche"], start=1)
    ]
    first = choose_balanced(cands, labeler_id=7, labels_done=3)
    assert first == choose_balanced(cands, labeler_id=7, labels_done=3)  # deterministic
    seen = {choose_balanced(cands, labeler_id=7, labels_done=n).predicted_label for n in range(6)}  # type: ignore[union-attr]
    assert seen == {"bad", "flur", "kueche"}  # cycles through all classes, no single-class bias
    assert choose_balanced([], labeler_id=1, labels_done=0) is None


def test_progress_and_next_item(session: Session, registry: SchemeRegistry) -> None:
    a, b = make_labeler(session, "a"), make_labeler(session, "b")
    imgs = [make_image(session, i) for i in range(1, 4)]
    _predict(session, imgs[0].id, "kueche", 0.4, 0.1)
    assert count_progress(session, a.id, "room").total == 3
    _label(session, registry, a, imgs[1], "bad")
    prog = count_progress(session, a.id, "room")
    assert (prog.done, prog.total) == (1, 3)
    item = next_item(session, labeler_id=a.id, scheme="room", skipped_shas=[imgs[2].sha256])
    assert item.candidate is not None and item.candidate.sha256 == imgs[0].sha256
    assert item.suggestion is not None and item.suggestion.label == "kueche"
    _ = b
