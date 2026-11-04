from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.labelschemes import SchemeRegistry
from app.models import Consensus, ConsensusStatus
from app.services.annotations import submit_label
from app.services.consensus import admin_resolve, conflict_queue, evaluate
from tests.conftest import make_image, make_labeler

S = ConsensusStatus


@pytest.mark.parametrize(
    ("labels", "status", "final", "agreement"),
    [
        ([], S.OPEN, None, 0.0),
        (["bad"], S.OPEN, None, 0.0),
        (["bad", "bad"], S.FINAL, "bad", 1.0),
        (["bad", "flur"], S.NEEDS_MORE, None, 0.5),
        (["bad", "flur", "bad"], S.FINAL, "bad", 2 / 3),
        (["bad", "flur", "kueche"], S.CONFLICT, None, 1 / 3),
        (["unklar", "unklar"], S.FINAL, "unklar", 1.0),  # unklar is a normal label
        (["bad", "bad", "flur", "flur"], S.CONFLICT, None, 0.5),
        (["bad", "bad", "bad", "flur"], S.FINAL, "bad", 0.75),
    ],
)
def test_evaluate(
    labels: list[str], status: ConsensusStatus, final: str | None, agreement: float
) -> None:
    result = evaluate(labels)
    assert (result.status, result.final_label) == (status, final)
    assert result.agreement == pytest.approx(agreement)


def test_trusted_label_finalises() -> None:
    result = evaluate([], trusted_label="kueche")
    assert (result.status, result.final_label, result.agreement) == (S.FINAL, "kueche", 1.0)


def _submit(session: Session, registry: SchemeRegistry, who, image, label: str):  # type: ignore[no-untyped-def]
    return submit_label(
        session, labeler=who, sha256=image.sha256, scheme="room", label=label, registry=registry
    )


def test_full_flow_agree(session: Session, registry: SchemeRegistry) -> None:
    a, b = make_labeler(session, "a"), make_labeler(session, "b")
    img = make_image(session, 1)
    assert _submit(session, registry, a, img, "bad").status is S.OPEN
    result = _submit(session, registry, b, img, "bad")
    assert (result.status, result.final_label) == (S.FINAL, "bad")


def test_full_flow_third_label_majority(session: Session, registry: SchemeRegistry) -> None:
    a, b, c = (make_labeler(session, n) for n in "abc")
    img = make_image(session, 1)
    _submit(session, registry, a, img, "bad")
    assert _submit(session, registry, b, img, "flur").status is S.NEEDS_MORE
    result = _submit(session, registry, c, img, "flur")
    assert (result.status, result.final_label) == (S.FINAL, "flur")


def test_three_way_split_goes_to_conflict_and_admin_resolves(
    session: Session, registry: SchemeRegistry
) -> None:
    a, b, c = (make_labeler(session, n) for n in "abc")
    admin = make_labeler(session, "admin")
    img = make_image(session, 1)
    for who, label in ((a, "bad"), (b, "flur"), (c, "kueche")):
        result = _submit(session, registry, who, img, label)
    assert result.status is S.CONFLICT
    queue = conflict_queue(session, "room")
    assert [item.sha256 for item in queue] == [img.sha256]
    assert sorted(label for _, label in queue[0].votes) == ["bad", "flur", "kueche"]

    row = admin_resolve(session, img.id, "room", "flur", admin, registry)
    assert (row.status, row.final_label, row.resolved_by_id) == (S.FINAL, "flur", admin.id)
    assert conflict_queue(session, "room") == []


def test_admin_decision_is_sticky(session: Session, registry: SchemeRegistry) -> None:
    a, b, c, d = (make_labeler(session, n) for n in "abcd")
    admin = make_labeler(session, "admin")
    img = make_image(session, 1)
    for who, label in ((a, "bad"), (b, "flur"), (c, "kueche")):
        _submit(session, registry, who, img, label)
    admin_resolve(session, img.id, "room", "flur", admin, registry)
    _submit(session, registry, d, img, "bad")  # late 4th label must not override the admin
    row = session.get(Consensus, (img.id, "room"))
    assert row is not None
    assert (row.status, row.final_label) == (S.FINAL, "flur")


def test_admin_resolve_rejects_unknown_label(session: Session, registry: SchemeRegistry) -> None:
    from app.labelschemes import InvalidLabelError

    admin = make_labeler(session, "admin")
    img = make_image(session, 1)
    with pytest.raises(InvalidLabelError):
        admin_resolve(session, img.id, "room", "garage", admin, registry)
