from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.labelschemes import SchemeRegistry
from app.models import Annotation
from app.services.annotations import submit_label
from app.services.ingest import get_or_create_library_labeler
from app.services.stats import (
    agreement_stats,
    cohens_kappa,
    krippendorff_alpha_nominal,
    label_distribution,
    percent_agreement,
    progress_stats,
)
from tests.conftest import make_image, make_labeler


def test_percent_agreement_and_kappa() -> None:
    pairs = [("a", "a"), ("a", "a"), ("b", "b"), ("a", "b")]
    assert percent_agreement(pairs) == 0.75
    assert cohens_kappa(pairs) == pytest.approx(0.5)
    assert percent_agreement([]) is None
    assert cohens_kappa([("a", "a")]) is None  # no variance -> undefined


def test_krippendorff_known_values() -> None:
    assert krippendorff_alpha_nominal([["a", "a"], ["b", "b"], ["a", "a"]]) == pytest.approx(1.0)
    # Two raters, 4 units, one disagreement: classic worked example -> alpha = 0.5333...
    units = [["a", "a"], ["a", "a"], ["b", "b"], ["a", "b"]]
    assert krippendorff_alpha_nominal(units) == pytest.approx(0.5333333, abs=1e-6)
    assert krippendorff_alpha_nominal([["a", "b"], ["b", "a"]]) == pytest.approx(-0.5)
    assert krippendorff_alpha_nominal([["a"], ["b"]]) is None  # nothing pairable
    # incomplete data (3 raters on one unit) still works
    assert krippendorff_alpha_nominal([["a", "a", "a"], ["a", "b"]]) is not None


def test_db_stats_exclude_library(session: Session, registry: SchemeRegistry) -> None:
    a, b = make_labeler(session, "a"), make_labeler(session, "b")
    imgs = [make_image(session, i) for i in range(1, 5)]
    lib = get_or_create_library_labeler(session)
    session.add(Annotation(image_id=imgs[3].id, labeler_id=lib.id, scheme="room", label="bad"))
    for img, la, lb in (
        (imgs[0], "bad", "bad"),
        (imgs[1], "bad", "flur"),
        (imgs[2], "kueche", "kueche"),
    ):
        submit_label(
            session, labeler=a, sha256=img.sha256, scheme="room", label=la, registry=registry
        )
        submit_label(
            session, labeler=b, sha256=img.sha256, scheme="room", label=lb, registry=registry
        )

    agr = agreement_stats(session, "room")
    assert len(agr.pairs) == 1 and agr.pairs[0].overlap == 3
    assert agr.pairs[0].agreement == pytest.approx(2 / 3)
    assert agr.overlapping_images == 3

    prog = progress_stats(session, "room")
    assert (prog.total, prog.final, prog.needs_more, prog.open) == (4, 2, 1, 1)
    dist = label_distribution(session, "room", registry.get("room").labels)
    assert dist["bad"] == 1 and dist["kueche"] == 1 and dist["flur"] == 0
