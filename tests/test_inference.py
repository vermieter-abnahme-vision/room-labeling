from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.labelschemes import SchemeRegistry
from app.models import ModelPrediction
from app.services import inference as inf
from app.services.annotations import submit_label
from tests.conftest import make_image, make_labeler

DIM = 8


class FakeEmbedder:
    """Deterministic stand-in for SigLIP: the 'image' is a class prototype plus noise."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_images(self, paths: Sequence[Path]) -> inf.Array:
        self.calls += 1
        vecs = []
        for p in paths:
            k = int(p.stem.split("_")[0]) % 3
            v = np.zeros(DIM, np.float32)
            v[k] = 1.0
            vecs.append(v)
        return np.stack(vecs)

    def embed_texts(self, texts: Sequence[str]) -> inf.Array:
        out = np.zeros((len(texts), DIM), np.float32)
        for i in range(len(texts)):
            out[i, i % DIM] = 1.0
        return out

    def zero_shot_scale_bias(self) -> tuple[float, float]:
        return 10.0, -5.0


def test_embedding_cache_roundtrip_and_key_validation(tmp_path: Path) -> None:
    cache = inf.EmbeddingCache(tmp_path, "m")
    sha = "ab" * 32
    assert not cache.has(sha)
    cache.save(sha, np.arange(4, dtype=np.float32))
    assert cache.has(sha) and cache.load(sha).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        cache.path_for("../../etc/passwd")


def test_compute_missing_embeddings_only_embeds_uncached(tmp_path: Path) -> None:
    cache = inf.EmbeddingCache(tmp_path / "c", "m")
    emb = FakeEmbedder()
    items = [(f"{i:064x}", tmp_path / f"{i}_x.jpg") for i in range(5)]
    done, failed = inf.compute_missing_embeddings(emb, cache, items, batch_size=2)
    assert (done, failed) == (5, [])
    done2, _ = inf.compute_missing_embeddings(emb, cache, items, batch_size=2)
    assert done2 == 0


def test_train_probe_predicts_and_reports_group_cv(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    labels, groups, rows = [], [], []
    for i in range(60):
        k = i % 3
        v = np.zeros(DIM, np.float32)
        v[k] = 1.0
        rows.append(v + rng.normal(0, 0.05, DIM).astype(np.float32))
        labels.append(["bad", "flur", "kueche"][k])
        groups.append(f"g{i % 6}")
    clf, result = inf.train_probe(np.stack(rows), labels, groups, tmp_path, version="v-test")
    assert result.cv_accuracy is not None and result.cv_accuracy > 0.9
    assert result.model_path.exists() and result.classes == ["bad", "flur", "kueche"]
    probs = clf.predict_proba(np.stack(rows[:3]))
    assert [result.classes[int(np.argmax(p))] for p in probs] == labels[:3]


def test_train_probe_needs_enough_data(tmp_path: Path) -> None:
    x = np.zeros((4, DIM), np.float32)
    with pytest.raises(inf.NotEnoughDataError):
        inf.train_probe(x, ["bad", "bad", "bad", "flur"], ["a"] * 4, tmp_path)


def test_zero_shot_and_write_predictions(session: Session, registry: SchemeRegistry) -> None:
    scheme = registry.get("room")
    images = [make_image(session, i) for i in range(1, 4)]
    vecs = np.eye(DIM, dtype=np.float32)[:3]
    probs, classes = inf.zero_shot_probabilities(FakeEmbedder(), scheme, vecs)
    assert probs.shape == (3, len(scheme.labels)) and np.allclose(probs.sum(1), 1.0, atol=1e-5)
    assert classes == list(scheme.labels)
    n = inf.write_predictions(
        session,
        scheme="room",
        version="siglip-zeroshot-x",
        image_ids=[i.id for i in images],
        probs=probs,
        classes=classes,
    )
    assert n == 3
    preds = list(session.scalars(select(ModelPrediction).order_by(ModelPrediction.image_id)))
    assert [p.label for p in preds] == classes[:3]
    assert all(0 <= p.margin <= p.confidence <= 1 for p in preds)  # type: ignore[operator]
    # idempotent per version
    inf.write_predictions(
        session,
        scheme="room",
        version="siglip-zeroshot-x",
        image_ids=[i.id for i in images],
        probs=probs,
        classes=classes,
    )
    assert len(list(session.scalars(select(ModelPrediction)))) == 3


def test_training_samples_use_finals_and_optionally_open(
    session: Session, registry: SchemeRegistry
) -> None:
    scheme = registry.get("room")
    a, b = make_labeler(session, "a"), make_labeler(session, "b")
    final_img, single_img, split_img = (make_image(session, i) for i in (1, 2, 3))
    for who in (a, b):
        submit_label(
            session,
            labeler=who,
            sha256=final_img.sha256,
            scheme="room",
            label="bad",
            registry=registry,
        )
    submit_label(
        session, labeler=a, sha256=single_img.sha256, scheme="room", label="flur", registry=registry
    )
    submit_label(
        session, labeler=a, sha256=split_img.sha256, scheme="room", label="flur", registry=registry
    )
    submit_label(
        session, labeler=b, sha256=split_img.sha256, scheme="room", label="bad", registry=registry
    )
    base = {s.sha256 for s in inf.training_samples(session, scheme)}
    assert base == {final_img.sha256}
    wide = {s.sha256 for s in inf.training_samples(session, scheme, include_unfinalized=True)}
    assert wide == {final_img.sha256, single_img.sha256}  # disagreeing image excluded
