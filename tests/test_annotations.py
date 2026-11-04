from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.labelschemes import InvalidLabelError, SchemeRegistry, UnknownSchemeError
from app.models import Annotation, ConsensusStatus
from app.services.annotations import ImageNotFoundError, submit_label
from tests.conftest import make_image, make_labeler


def _count(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Annotation)) or 0


def test_resubmit_same_label_is_noop(session: Session, registry: SchemeRegistry) -> None:
    a, b = make_labeler(session, "a"), make_labeler(session, "b")
    img = make_image(session, 1)
    first = submit_label(
        session, labeler=a, sha256=img.sha256, scheme="room", label="bad", registry=registry
    )
    again = submit_label(
        session, labeler=a, sha256=img.sha256, scheme="room", label="bad", registry=registry
    )
    assert first.changed and not again.changed
    assert _count(session) == 1
    # the same labeler must never count twice towards consensus
    assert again.status is ConsensusStatus.OPEN
    _ = b


def test_resubmit_different_label_updates(session: Session, registry: SchemeRegistry) -> None:
    a = make_labeler(session, "a")
    img = make_image(session, 1)
    submit_label(
        session, labeler=a, sha256=img.sha256, scheme="room", label="bad", registry=registry
    )
    result = submit_label(
        session, labeler=a, sha256=img.sha256, scheme="room", label="flur", registry=registry
    )
    assert result.changed
    labels = list(session.scalars(select(Annotation.label)))
    assert labels == ["flur"]


def test_validation(session: Session, registry: SchemeRegistry) -> None:
    a = make_labeler(session, "a")
    img = make_image(session, 1)
    with pytest.raises(InvalidLabelError):
        submit_label(
            session, labeler=a, sha256=img.sha256, scheme="room", label="garage", registry=registry
        )
    with pytest.raises(UnknownSchemeError):
        submit_label(
            session, labeler=a, sha256=img.sha256, scheme="windows", label="bad", registry=registry
        )
    with pytest.raises(ImageNotFoundError):
        submit_label(
            session, labeler=a, sha256="f" * 64, scheme="room", label="bad", registry=registry
        )
