"""SigLIP embeddings, zero-shot fallback and the logistic-regression probe.

Heavy dependencies (torch/transformers) are imported lazily inside ``SigLipEmbedder`` so the web
app, the tests and the other services never need them.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.labelschemes import LabelScheme
from app.models import (
    LIBRARY_LABELER_SUB,
    Annotation,
    Consensus,
    ConsensusStatus,
    Image,
    Labeler,
    ModelPrediction,
)

logger = logging.getLogger(__name__)

Array = NDArray[np.float32]
MIN_SAMPLES_PER_CLASS = 3
MIN_CLASSES = 2
_SAFE_TAG = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------------------
# Embedding cache
# --------------------------------------------------------------------------------------
def model_tag(model_name: str, revision: str | None = None) -> str:
    tag = _SAFE_TAG.sub("-", model_name).strip("-")
    return f"{tag}@{_SAFE_TAG.sub('-', revision)}" if revision else tag


class EmbeddingCache:
    """On-disk cache of image embeddings, keyed by image sha256 (``.npy``)."""

    def __init__(self, root: Path, tag: str) -> None:
        self.root = root / tag

    def path_for(self, sha256: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("invalid sha256")
        return self.root / sha256[:2] / f"{sha256}.npy"

    def has(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    def save(self, sha256: str, vector: Array) -> None:
        path = self.path_for(sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, vector.astype(np.float32), allow_pickle=False)
        tmp.replace(path)

    def load(self, sha256: str) -> Array:
        return np.load(self.path_for(sha256), allow_pickle=False)  # type: ignore[no-any-return]


class Embedder(Protocol):
    def embed_images(self, paths: Sequence[Path]) -> Array: ...
    def embed_texts(self, texts: Sequence[str]) -> Array: ...
    def zero_shot_scale_bias(self) -> tuple[float, float]: ...


def pick_device(torch_module: Any, requested: str = "auto") -> str:
    """auto: CUDA, then Apple GPU (MPS), then CPU."""
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class SigLipEmbedder:
    """transformers SigLIP wrapper (L2-normalised embeddings)."""

    def __init__(self, model_name: str, revision: str | None = None, device: str = "auto") -> None:
        import os

        # Let ops without an MPS kernel fall back to the CPU instead of crashing (Apple GPU).
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = pick_device(torch, device)
        logger.info("SigLIP running on device: %s", self.device)
        # trust_remote_code is never enabled: only weights + config are loaded.
        self.model = (
            AutoModel.from_pretrained(model_name, revision=revision, trust_remote_code=False)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name, revision=revision, trust_remote_code=False
        )

    def embed_images(self, paths: Sequence[Path]) -> Array:
        from PIL import Image as PILImage
        from PIL import ImageOps

        images = []
        for path in paths:
            with PILImage.open(path) as img:
                images.append(ImageOps.exif_transpose(img).convert("RGB"))
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        return _normalize(_as_tensor(feats).cpu().numpy())

    def embed_texts(self, texts: Sequence[str]) -> Array:
        inputs = self.processor(
            text=list(texts), padding="max_length", max_length=64, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return _normalize(_as_tensor(feats).cpu().numpy())

    def zero_shot_scale_bias(self) -> tuple[float, float]:
        return (
            float(self.model.logit_scale.exp().item()),
            float(self.model.logit_bias.item()),
        )


def _as_tensor(output: Any) -> Any:
    """transformers >= 5 returns a ModelOutput (pooler_output) instead of a bare tensor."""
    return output if hasattr(output, "cpu") else output.pooler_output


def _normalize(matrix: Any) -> Array:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normalized: Array = (arr / np.clip(norms, 1e-12, None)).astype(np.float32)
    return normalized


def compute_missing_embeddings(
    embedder: Embedder,
    cache: EmbeddingCache,
    items: Iterable[tuple[str, Path]],
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[int, list[str]]:
    """Embed every (sha256, path) not yet cached. Returns (newly_embedded, failed_shas)."""
    todo = [(sha, path) for sha, path in items if not cache.has(sha)]
    failed: list[str] = []
    done = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        try:
            vectors = embedder.embed_images([p for _, p in batch])
        except Exception:
            logger.exception("batch failed, retrying items individually")
            vectors_list: list[Array | None] = []
            for _, path in batch:
                try:
                    vectors_list.append(embedder.embed_images([path])[0])
                except Exception:
                    logger.exception("embedding failed for one image")
                    vectors_list.append(None)
            for (sha, _), vec in zip(batch, vectors_list, strict=True):
                if vec is None:
                    failed.append(sha)
                else:
                    cache.save(sha, vec)
                    done += 1
        else:
            for (sha, _), vec in zip(batch, vectors, strict=True):
                cache.save(sha, vec)
                done += 1
        if progress:
            progress(min(start + batch_size, len(todo)), len(todo))
    return done, failed


# --------------------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LabeledSample:
    sha256: str
    group_id: str
    label: str


def training_samples(
    session: Session, scheme: LabelScheme, include_unfinalized: bool = False
) -> list[LabeledSample]:
    """Labelled images: finals (incl. library) and optionally the plurality of open images."""
    samples: dict[int, LabeledSample] = {}
    for image, cons in session.execute(
        select(Image, Consensus)
        .join(Consensus, Consensus.image_id == Image.id)
        .where(
            Consensus.scheme == scheme.name,
            Consensus.status == ConsensusStatus.FINAL,
            Consensus.final_label.is_not(None),
        )
    ):
        if cons.final_label in scheme.labels:
            samples[image.id] = LabeledSample(image.sha256, image.group_id, cons.final_label)
    if include_unfinalized:
        votes: dict[int, list[str]] = {}
        rows = session.execute(
            select(Image, Annotation.label)
            .join(Annotation, Annotation.image_id == Image.id)
            .join(Labeler, Labeler.id == Annotation.labeler_id)
            .outerjoin(
                Consensus, (Consensus.image_id == Image.id) & (Consensus.scheme == scheme.name)
            )
            .where(
                Annotation.scheme == scheme.name,
                Labeler.oidc_sub != LIBRARY_LABELER_SUB,
                Consensus.status.in_([ConsensusStatus.OPEN, ConsensusStatus.NEEDS_MORE]),
            )
        )
        images: dict[int, Image] = {}
        for image, label in rows:
            images[image.id] = image
            votes.setdefault(image.id, []).append(label)
        for image_id, labels in votes.items():
            top = max(set(labels), key=labels.count)
            if labels.count(top) == len(labels) and top in scheme.labels:
                img = images[image_id]
                samples.setdefault(image_id, LabeledSample(img.sha256, img.group_id, top))
    return list(samples.values())


# --------------------------------------------------------------------------------------
# Probe
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeResult:
    version: str
    classes: list[str]
    n_samples: int
    cv_accuracy: float | None
    model_path: Path


class NotEnoughDataError(RuntimeError):
    pass


def new_model_version(prefix: str = "probe", now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"siglip-{prefix}-{stamp}"


def train_probe(
    x: Array,
    labels: Sequence[str],
    groups: Sequence[str],
    models_dir: Path,
    *,
    version: str | None = None,
    c: float = 1.0,
) -> tuple[LogisticRegression, ProbeResult]:
    """Fit class-balanced logistic regression; report group-aware CV accuracy (no leakage)."""
    classes = sorted(set(labels))
    counts = {k: list(labels).count(k) for k in classes}
    usable = [k for k in classes if counts[k] >= MIN_SAMPLES_PER_CLASS]
    if len(usable) < MIN_CLASSES:
        raise NotEnoughDataError(
            f"need >= {MIN_CLASSES} classes with >= {MIN_SAMPLES_PER_CLASS} samples, got {counts}"
        )
    mask = np.array([lab in usable for lab in labels])
    xs, ys = x[mask], np.array(labels)[mask]
    gs = np.array(groups)[mask]

    clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
    cv_accuracy: float | None = None
    n_groups = len(set(gs))
    if n_groups >= 3:
        splitter = GroupKFold(n_splits=min(5, n_groups))
        try:
            pred = cross_val_predict(clf, xs, ys, groups=gs, cv=splitter)
            cv_accuracy = float(np.mean(pred == ys))
        except ValueError:
            logger.warning("cross-validation skipped (a training fold lacked classes)")
    clf.fit(xs, ys)

    version = version or new_model_version()
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{version}.joblib"
    joblib.dump({"classifier": clf, "classes": list(clf.classes_), "version": version}, model_path)
    return clf, ProbeResult(version, list(clf.classes_), int(mask.sum()), cv_accuracy, model_path)


def top_two(probs: Array) -> tuple[int, float, float]:
    order = np.argsort(probs)[::-1]
    top = float(probs[order[0]])
    second = float(probs[order[1]]) if len(order) > 1 else 0.0
    return int(order[0]), top, top - second


def zero_shot_probabilities(
    embedder: Embedder, scheme: LabelScheme, image_vectors: Array
) -> tuple[Array, list[str]]:
    """Softmax over SigLIP logits (scale * cosine + bias) of the scheme's text prompts."""
    labels = list(scheme.labels)
    text = embedder.embed_texts([scheme.prompt(label) for label in labels])
    scale, bias = embedder.zero_shot_scale_bias()
    logits = scale * image_vectors @ text.T + bias
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32), labels


def write_predictions(
    session: Session,
    *,
    scheme: str,
    version: str,
    image_ids: Sequence[int],
    probs: Array,
    classes: Sequence[str],
) -> int:
    """Replace the predictions of ``version`` for the given images (idempotent)."""
    session.execute(
        delete(ModelPrediction).where(
            ModelPrediction.scheme == scheme,
            ModelPrediction.model_version == version,
            ModelPrediction.image_id.in_(list(image_ids)),
        )
    )
    for image_id, row in zip(image_ids, probs, strict=True):
        idx, confidence, margin = top_two(row)
        session.add(
            ModelPrediction(
                image_id=image_id,
                scheme=scheme,
                label=classes[idx],
                confidence=confidence,
                margin=margin,
                model_version=version,
            )
        )
    session.flush()
    return len(image_ids)
