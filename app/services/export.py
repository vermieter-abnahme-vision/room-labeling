"""Datapack export: JSONL manifest (source of truth) + meta.json, and imagefolder materialisation.

Splits are computed per ``group_id`` (video / apartment / source), so a group never appears in
two splits (no leakage). Re-exports can keep earlier assignments via ``previous_assignments``
so a group never migrates from train to test between dataset versions.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__
from app.labelschemes import SchemeRegistry, default_registry
from app.models import (
    LIBRARY_LABELER_SUB,
    Annotation,
    Consensus,
    ConsensusStatus,
    Image,
    ImageSource,
    Labeler,
    ModelPrediction,
)
from app.services.sampling import latest_model_version

FORMAT_VERSION = 1
SPLITS: tuple[str, ...] = ("train", "val", "test")
DEFAULT_RATIOS: Mapping[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
EXPORT_NAME_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z(-[0-9]+)?$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
LibrarySplit = Literal["train", "group"]
MaterializeMode = Literal["hardlink", "copy", "symlink"]


# --------------------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------------------
def _group_hash(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def assign_splits(
    group_sizes: Mapping[str, int],
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    *,
    seed: int = 0,
    fixed: Mapping[str, str] | None = None,
    forced: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign each group to exactly one split (group-disjoint, size-balanced, deterministic).

    * ``forced`` groups are assigned as given and do not count towards the ratio targets;
    * ``fixed`` groups keep a previous assignment (stable across re-exports);
    * remaining groups are placed largest-first into the split with the largest deficit,
      ties broken by a seeded hash (reproducible shuffle).
    """
    if not ratios or any(r < 0 for r in ratios.values()) or sum(ratios.values()) <= 0:
        raise ValueError("ratios must be non-negative with a positive sum")
    norm = sum(ratios.values())
    ratio = {split: value / norm for split, value in ratios.items()}
    order = list(ratio)

    assignment: dict[str, str] = {}
    for group, split in (forced or {}).items():
        if group in group_sizes:
            assignment[group] = split
    free_total = sum(size for g, size in group_sizes.items() if g not in assignment)
    current: Counter[str] = Counter()
    for group, split in (fixed or {}).items():
        if group in group_sizes and group not in assignment and split in ratio:
            assignment[group] = split
            current[split] += group_sizes[group]

    remaining = sorted(
        (g for g in group_sizes if g not in assignment),
        key=lambda g: (-group_sizes[g], _group_hash(seed, g)),
    )
    for group in remaining:
        deficits = {s: ratio[s] * free_total - current[s] for s in order}
        best = max(order, key=lambda s: (deficits[s], -order.index(s)))
        assignment[group] = best
        current[best] += group_sizes[group]
    return assignment


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def pseudonymize(salt: str, labeler_sub: str) -> str:
    if labeler_sub == LIBRARY_LABELER_SUB:
        return "library"
    digest = hmac.new(salt.encode(), labeler_sub.encode(), hashlib.sha256).hexdigest()
    return f"L-{digest[:12]}"


@dataclass(frozen=True)
class ExportOptions:
    scheme: str = "room"
    ratios: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    seed: int = 0
    library_split: LibrarySplit = "train"
    pseudonym_salt: str = ""
    previous_assignments: Mapping[str, str] | None = None


@dataclass(frozen=True)
class Manifest:
    rows: list[dict[str, Any]]
    meta: dict[str, Any]

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in self.rows
        )


def build_manifest(
    session: Session,
    options: ExportOptions,
    registry: SchemeRegistry | None = None,
    now: datetime | None = None,
) -> Manifest:
    scheme_def = (registry or default_registry()).get(options.scheme)
    if not options.pseudonym_salt:
        raise ValueError("pseudonym_salt required")

    finals = session.execute(
        select(Image, Consensus)
        .join(Consensus, Consensus.image_id == Image.id)
        .where(
            Consensus.scheme == options.scheme,
            Consensus.status == ConsensusStatus.FINAL,
            Consensus.final_label.is_not(None),
        )
        .order_by(Image.sha256)
    ).all()

    raw: dict[int, list[dict[str, str]]] = defaultdict(list)
    for image_id, sub, label in session.execute(
        select(Annotation.image_id, Labeler.oidc_sub, Annotation.label)
        .join(Labeler, Labeler.id == Annotation.labeler_id)
        .where(Annotation.scheme == options.scheme)
        .order_by(Annotation.image_id, Annotation.id)
    ):
        raw[image_id].append({"labeler": pseudonymize(options.pseudonym_salt, sub), "label": label})
    for votes in raw.values():
        votes.sort(key=lambda v: (v["labeler"], v["label"]))

    version = latest_model_version(session, options.scheme)
    predictions: dict[int, ModelPrediction] = {}
    if version:
        predictions = {
            p.image_id: p
            for p in session.scalars(
                select(ModelPrediction).where(
                    ModelPrediction.scheme == options.scheme,
                    ModelPrediction.model_version == version,
                )
            )
        }

    group_sizes: Counter[str] = Counter(image.group_id for image, _ in finals)
    forced: dict[str, str] = {}
    if options.library_split == "train":
        for image, _ in finals:
            if image.source is ImageSource.LIBRARY:
                forced[image.group_id] = "train"
    split_of = assign_splits(
        group_sizes,
        options.ratios,
        seed=options.seed,
        fixed=options.previous_assignments,
        forced=forced,
    )

    rows: list[dict[str, Any]] = []
    for image, cons in finals:
        pred = predictions.get(image.id)
        if cons.resolved_by_id is not None:
            resolution = "admin"
        elif image.source is ImageSource.LIBRARY:
            resolution = "library"
        else:
            resolution = "consensus"
        rows.append(
            {
                "sha256": image.sha256,
                "path": image.path,
                "group_id": image.group_id,
                "source": image.source.value,
                "scheme": options.scheme,
                "final_label": cons.final_label,
                "agreement": round(cons.agreement, 6),
                "resolution": resolution,
                "raw_labels": raw.get(image.id, []),
                "model_prediction": (
                    {
                        "label": pred.label,
                        "confidence": round(pred.confidence, 6),
                        "model_version": pred.model_version,
                    }
                    if pred
                    else None
                ),
                "suggested_split": split_of[image.group_id],
            }
        )

    counts: dict[str, dict[str, int]] = {s: {} for s in options.ratios}
    for row in rows:
        bucket = counts[row["suggested_split"]]
        bucket[row["final_label"]] = bucket.get(row["final_label"], 0) + 1
    created = (now or datetime.now(UTC)).astimezone(UTC)
    body = Manifest(rows=rows, meta={})
    meta: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "app_version": __version__,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scheme": options.scheme,
        "labels": list(scheme_def.labels),
        "seed": options.seed,
        "ratios": dict(options.ratios),
        "library_split": options.library_split,
        "n_images": len(rows),
        "n_groups": len(group_sizes),
        "counts": {s: dict(sorted(c.items())) for s, c in counts.items()},
        "images_per_split": {s: sum(c.values()) for s, c in counts.items()},
        "model_version": version,
        "manifest_sha256": hashlib.sha256(body.to_jsonl().encode()).hexdigest(),
        "split_assignments": dict(sorted(split_of.items())),
        "warnings": [f"split '{s}' is empty" for s, c in counts.items() if not c and rows],
    }
    return Manifest(rows=rows, meta=meta)


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------
def write_export(manifest: Manifest, export_dir: Path, now: datetime | None = None) -> Path:
    """Write ``manifest.jsonl`` + ``meta.json`` into a fresh timestamped directory."""
    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / stamp
    suffix = 1
    while target.exists():
        suffix += 1
        target = export_dir / f"{stamp}-{suffix}"
    tmp = export_dir / f".{target.name}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    (tmp / "manifest.jsonl").write_text(manifest.to_jsonl(), encoding="utf-8")
    (tmp / "meta.json").write_text(
        json.dumps(manifest.meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.rename(target)
    return target


def list_exports(export_dir: Path) -> list[str]:
    if not export_dir.is_dir():
        return []
    return sorted(
        (p.name for p in export_dir.iterdir() if p.is_dir() and EXPORT_NAME_RE.match(p.name)),
        reverse=True,
    )


def resolve_export_file(export_dir: Path, name: str, filename: str) -> Path | None:
    """Safely resolve a downloadable export file; None if invalid (no traversal possible)."""
    if not EXPORT_NAME_RE.match(name) or filename not in {"manifest.jsonl", "meta.json"}:
        return None
    path = (export_dir / name / filename).resolve()
    if not path.is_relative_to(export_dir.resolve()) or not path.is_file():
        return None
    return path


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def assignments_from_manifest(path: Path) -> dict[str, str]:
    return {row["group_id"]: row["suggested_split"] for row in read_manifest(path)}


def materialize_imagefolder(
    manifest_path: Path,
    images_dir: Path,
    out_dir: Path,
    mode: MaterializeMode = "hardlink",
    verify: bool = False,
) -> Counter[str]:
    """Create ``out_dir/<split>/<label>/<sha>.<ext>`` from the manifest (manifest stays leading)."""
    root = images_dir.resolve()
    counter: Counter[str] = Counter()
    for row in read_manifest(manifest_path):
        sha = row["sha256"]
        split, label = row["suggested_split"], row["final_label"]
        if not SHA_RE.match(sha) or split not in SPLITS or not re.match(r"^[a-z0-9_]+$", label):
            raise ValueError(f"invalid manifest row for {sha!r}")
        src = (images_dir / row["path"]).resolve()
        if not src.is_relative_to(root) or not src.is_file():
            raise FileNotFoundError(f"image missing in store: {row['path']}")
        if verify and hashlib.sha256(src.read_bytes()).hexdigest() != sha:
            raise ValueError(f"checksum mismatch: {row['path']}")
        dest_dir = out_dir / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{sha}{src.suffix}"
        if dest.exists() or dest.is_symlink():
            counter["existing"] += 1
            continue
        if mode == "symlink":
            dest.symlink_to(src)
        elif mode == "hardlink":
            try:
                os.link(src, dest)
            except OSError:  # cross-device etc.
                shutil.copyfile(src, dest)
        else:
            shutil.copyfile(src, dest)
        counter[split] += 1
    return counter


def parse_ratios(text: str) -> dict[str, float]:
    parts = [float(p) for p in text.split(",")]
    if len(parts) != len(SPLITS):
        raise ValueError("ratios need three values: train,val,test")
    return dict(zip(SPLITS, parts, strict=True))
