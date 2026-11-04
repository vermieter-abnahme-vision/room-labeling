from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.config import Settings
from app.labelschemes import SchemeRegistry
from app.models import Annotation, ImageSource, ModelPrediction
from app.services.annotations import submit_label
from app.services.export import (
    ExportOptions,
    assign_splits,
    assignments_from_manifest,
    build_manifest,
    list_exports,
    materialize_imagefolder,
    pseudonymize,
    read_manifest,
    resolve_export_file,
    write_export,
)
from app.services.ingest import get_or_create_library_labeler
from tests.conftest import make_image, make_labeler, write_jpeg

SALT = "salt-salt-salt"
NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_assign_splits_is_group_disjoint_and_roughly_balanced() -> None:
    sizes = {f"g{i}": 10 + i for i in range(30)}
    first = assign_splits(sizes, seed=1)
    assert set(first) == set(sizes)
    assert set(first.values()) == {"train", "val", "test"}
    totals = {
        s: sum(sizes[g] for g, v in first.items() if v == s) for s in ("train", "val", "test")
    }
    all_images = sum(sizes.values())
    assert totals["train"] / all_images == pytest.approx(0.8, abs=0.06)
    # deterministic
    assert first == assign_splits(sizes, seed=1)


def test_assign_splits_respects_fixed_and_forced() -> None:
    sizes = {"a": 50, "b": 50, "c": 50, "lib": 500}
    result = assign_splits(
        sizes, seed=3, fixed={"a": "test", "ghost": "val"}, forced={"lib": "train"}
    )
    assert result["a"] == "test" and result["lib"] == "train"
    assert "ghost" not in result


def _label_final(session, registry, labelers, image, label):  # type: ignore[no-untyped-def]
    for who in labelers:
        submit_label(
            session, labeler=who, sha256=image.sha256, scheme="room", label=label, registry=registry
        )


def _seed_db(session: Session, registry: SchemeRegistry):  # type: ignore[no-untyped-def]
    a, b = (
        make_labeler(session, "a", sub="sub-a", email="a@secret.example"),
        make_labeler(session, "b", sub="sub-b"),
    )
    lib = get_or_create_library_labeler(session)
    rows = []
    n = 0
    for group in ("vid1", "vid2", "vid3", "vid4", "vid5"):
        for _ in range(4):
            n += 1
            img = make_image(session, n, group=group)
            _label_final(session, registry, (a, b), img, "bad" if n % 2 else "flur")
            rows.append(img)
    for _ in range(3):
        n += 1
        img = make_image(session, n, group="library:places365:kitchen", source=ImageSource.LIBRARY)
        session.add(Annotation(image_id=img.id, labeler_id=lib.id, scheme="room", label="kueche"))
        session.flush()
        from app.services.consensus import recompute_consensus

        recompute_consensus(session, img.id, "room", registry)
    open_img = make_image(session, 99, group="vid1")  # unlabeled -> must not be exported
    session.add(
        ModelPrediction(
            image_id=rows[0].id,
            scheme="room",
            label="bad",
            confidence=0.9,
            margin=0.8,
            model_version="m1",
        )
    )
    session.flush()
    return rows, open_img


def test_manifest_content_and_no_leakage(session: Session, registry: SchemeRegistry) -> None:
    _, open_img = _seed_db(session, registry)
    manifest = build_manifest(
        session, ExportOptions(pseudonym_salt=SALT, seed=7), registry, now=NOW
    )
    rows = manifest.rows
    assert len(rows) == 23 and open_img.sha256 not in {r["sha256"] for r in rows}

    # no group in two splits
    splits_by_group: dict[str, set[str]] = {}
    for r in rows:
        splits_by_group.setdefault(r["group_id"], set()).add(r["suggested_split"])
    assert all(len(v) == 1 for v in splits_by_group.values())
    # library images are train-only by default
    assert {r["suggested_split"] for r in rows if r["source"] == "library"} == {"train"}

    first = (
        rows[0] if rows[0]["model_prediction"] else next(r for r in rows if r["model_prediction"])
    )
    assert set(first) == {
        "sha256",
        "path",
        "group_id",
        "source",
        "scheme",
        "final_label",
        "agreement",
        "resolution",
        "raw_labels",
        "model_prediction",
        "suggested_split",
    }
    assert first["model_prediction"] == {"label": "bad", "confidence": 0.9, "model_version": "m1"}
    assert first["agreement"] == 1.0 and first["resolution"] == "consensus"

    # labelers are pseudonymised: no sub / e-mail leak into the datapack
    blob = manifest.to_jsonl()
    assert "sub-a" not in blob and "secret.example" not in blob
    assert {v["labeler"] for r in rows for v in r["raw_labels"]} >= {
        pseudonymize(SALT, "sub-a"),
        "library",
    }
    assert manifest.meta["created_at"] == "2026-01-02T03:04:05Z"
    assert manifest.meta["format_version"] == 1 and manifest.meta["n_images"] == 23


def test_manifest_is_reproducible(session: Session, registry: SchemeRegistry) -> None:
    _seed_db(session, registry)
    opts = ExportOptions(pseudonym_salt=SALT, seed=7)
    one = build_manifest(session, opts, registry, now=NOW)
    two = build_manifest(session, opts, registry, now=datetime(2030, 1, 1, tzinfo=UTC))
    assert one.to_jsonl() == two.to_jsonl()
    assert one.meta["manifest_sha256"] == two.meta["manifest_sha256"]


def test_write_and_previous_assignments_are_stable(
    session: Session, registry: SchemeRegistry, tmp_path: Path
) -> None:
    _seed_db(session, registry)
    export_dir = tmp_path / "exports"
    m1 = build_manifest(session, ExportOptions(pseudonym_salt=SALT, seed=1), registry)
    target = write_export(m1, export_dir, now=NOW)
    assert list_exports(export_dir) == [target.name]
    prev = assignments_from_manifest(target / "manifest.jsonl")
    # different seed, but previous assignments pinned -> identical split per group
    m2 = build_manifest(
        session, ExportOptions(pseudonym_salt=SALT, seed=999, previous_assignments=prev), registry
    )
    assert {r["group_id"]: r["suggested_split"] for r in m2.rows if r["source"] != "library"} == {
        g: s for g, s in prev.items() if not g.startswith("library:")
    }
    meta = json.loads((target / "meta.json").read_text())
    assert meta["manifest_sha256"] == m1.meta["manifest_sha256"]
    # same-second second export does not overwrite
    again = write_export(m1, export_dir, now=NOW)
    assert again != target


def test_export_file_resolution_blocks_traversal(tmp_path: Path) -> None:
    (tmp_path / "20260102T030405Z").mkdir()
    (tmp_path / "20260102T030405Z" / "meta.json").write_text("{}")
    assert resolve_export_file(tmp_path, "20260102T030405Z", "meta.json") is not None
    assert resolve_export_file(tmp_path, "../etc", "meta.json") is None
    assert resolve_export_file(tmp_path, "20260102T030405Z", "../../etc/passwd") is None
    assert resolve_export_file(tmp_path, "20260102T030405Z", "manifest.jsonl") is None


def test_materialize_imagefolder(
    session: Session, registry: SchemeRegistry, settings: Settings, tmp_path: Path
) -> None:
    rows, _ = _seed_db(session, registry)
    for img in rows:
        write_jpeg(settings.images_dir / img.path, (img.id * 5 % 255, 10, 10))
    manifest = build_manifest(session, ExportOptions(pseudonym_salt=SALT), registry)
    # only keep the non-library rows whose files exist
    kept = [r for r in manifest.rows if r["source"] != "library"]
    mpath = tmp_path / "m.jsonl"
    mpath.write_text("".join(json.dumps(r) + "\n" for r in kept))
    out = tmp_path / "data"
    counts = materialize_imagefolder(mpath, settings.images_dir, out, mode="hardlink")
    assert sum(counts.values()) == len(kept)
    for r in kept:
        f = out / r["suggested_split"] / r["final_label"] / f"{r['sha256']}.jpg"
        assert f.is_file()
    # idempotent
    assert materialize_imagefolder(mpath, settings.images_dir, out)["existing"] == len(kept)
    assert len(read_manifest(mpath)) == len(kept)


def test_materialize_rejects_traversal_in_manifest(tmp_path: Path) -> None:
    evil = {
        "sha256": "a" * 64,
        "path": "../../etc/passwd",
        "suggested_split": "train",
        "final_label": "bad",
    }
    m = tmp_path / "m.jsonl"
    m.write_text(json.dumps(evil) + "\n")
    with pytest.raises(FileNotFoundError):
        materialize_imagefolder(m, tmp_path / "images", tmp_path / "out")
    evil["final_label"] = "../x"
    m.write_text(json.dumps(evil) + "\n")
    with pytest.raises(ValueError, match="invalid manifest row"):
        materialize_imagefolder(m, tmp_path / "images", tmp_path / "out")
