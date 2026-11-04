"""Ingestion: folder / video / library -> content-addressed image store + ``images`` rows.

Security notes: ffmpeg is always called with an argument list (never a shell); only local
files are ever read (``-protocol_whitelist file``), no URLs are fetched; file names are
sanitised; type and size are validated.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.labelschemes import ROOM_SCHEME, SchemeRegistry
from app.models import (
    LIBRARY_LABELER_SUB,
    Annotation,
    Image,
    ImageSource,
    Labeler,
)
from app.services.consensus import ensure_consensus_rows, recompute_consensus

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
LIBRARY_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
_FORMAT_TO_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
_GROUP_MAX = 200


class IngestError(Exception):
    """Recoverable ingest problem for a single input (reported, never silently dropped)."""


@dataclass
class IngestReport:
    added: int = 0
    duplicates: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: IngestReport) -> None:
        self.added += other.added
        self.duplicates += other.duplicates
        self.skipped += other.skipped
        self.errors.extend(other.errors)


def sanitize_group_id(raw: str) -> str:
    """Normalise a folder/file name into a safe, stable group id."""
    text = unicodedata.normalize("NFKC", raw).strip()
    text = re.sub(r"[^\w.\-:+ ]", "_", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "_", text).strip("._")
    return (text or "unnamed")[:_GROUP_MAX]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def validate_image_file(path: Path, settings: Settings, allowed_ext: set[str]) -> str:
    """Return the canonical extension (without dot) or raise IngestError."""
    if path.suffix.lower() not in allowed_ext:
        raise IngestError(f"unsupported extension: {path.suffix!r}")
    size = path.stat().st_size
    if size == 0 or size > settings.max_image_bytes:
        raise IngestError(f"file size out of range: {size} bytes")
    PILImage.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with PILImage.open(path) as img:
            fmt = img.format or ""
            width, height = img.size
            img.verify()
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError) as exc:
        raise IngestError(f"not a valid image: {exc.__class__.__name__}") from exc
    if fmt not in _FORMAT_TO_EXT:
        raise IngestError(f"unsupported image format: {fmt}")
    if width * height > settings.max_image_pixels:
        raise IngestError("image too large")
    return _FORMAT_TO_EXT[fmt]


def _store_path(sha256: str, ext: str) -> str:
    return f"{sha256[:2]}/{sha256}.{ext}"


def register_image(
    session: Session,
    settings: Settings,
    src: Path,
    *,
    source: ImageSource,
    group_id: str,
    extra: dict[str, Any] | None = None,
    allowed_ext: set[str] | None = None,
    registry: SchemeRegistry | None = None,
) -> tuple[Image, bool]:
    """Validate ``src``, copy it into the store (content-addressed) and create the row.

    Returns ``(image, created)``; ``created`` is False when the sha256 already exists.
    """
    ext = validate_image_file(src, settings, allowed_ext or IMAGE_EXTENSIONS)
    digest = sha256_file(src)
    existing = session.scalar(select(Image).where(Image.sha256 == digest))
    if existing is not None:
        return existing, False

    rel = _store_path(digest, ext)
    target = (settings.images_dir / rel).resolve()
    root = settings.images_dir.resolve()
    if not target.is_relative_to(root):  # defence in depth; rel is built from hex only
        raise IngestError("invalid target path")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = target.with_suffix(target.suffix + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(target)

    image = Image(
        sha256=digest,
        path=rel,
        source=source,
        group_id=sanitize_group_id(group_id),
        extra=extra,
    )
    session.add(image)
    session.flush()
    ensure_consensus_rows(session, [image.id], ROOM_SCHEME)
    return image, True


# --------------------------------------------------------------------------------------
# Folder
# --------------------------------------------------------------------------------------
def _iter_files(root: Path, extensions: set[str]) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() in extensions:
            yield path


def _maybe_commit(session: Session, report: IngestReport, commit_every: int | None) -> None:
    """Keep write transactions short: commit after every ``commit_every`` new images."""
    if commit_every and report.added and report.added % commit_every == 0:
        session.commit()


def ingest_folder(
    session: Session, settings: Settings, root: Path, commit_every: int | None = 200
) -> IngestReport:
    """Ingest JPGs below ``root``; ``group_id`` = first-level sub-directory name."""
    report = IngestReport()
    if not root.is_dir():
        raise IngestError(f"not a directory: {root}")
    for path in _iter_files(root, IMAGE_EXTENSIONS):
        relative = path.relative_to(root)
        group = relative.parts[0] if len(relative.parts) > 1 else "root"
        try:
            _, created = register_image(
                session,
                settings,
                path,
                source=ImageSource.FOLDER,
                group_id=group,
                extra={"original_name": path.name},
            )
        except IngestError as exc:
            report.skipped += 1
            report.errors.append(f"{relative}: {exc}")
            logger.warning("skipped image (%s): %s", path.name, exc)
            continue
        if created:
            report.added += 1
            _maybe_commit(session, report, commit_every)
        else:
            report.duplicates += 1
    return report


# --------------------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------------------
def build_ffmpeg_command(settings: Settings, video: Path, out_dir: Path) -> list[str]:
    side = settings.frame_max_side
    scale = f"scale=w='if(gt(iw,ih),min({side},iw),-2)':h='if(gt(iw,ih),-2,min({side},ih))'"
    return [
        settings.ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file",
        "-i",
        str(video.resolve()),
        "-vf",
        f"fps={settings.video_fps},{scale}",
        "-q:v",
        "2",
        "-f",
        "image2",
        str(out_dir / "frame_%06d.jpg"),
    ]


def extract_frames(settings: Settings, video: Path, out_dir: Path) -> list[Path]:
    """Run ffmpeg (argument list, no shell) and return extracted frames in order."""
    if shutil.which(settings.ffmpeg_bin) is None:
        raise IngestError(f"ffmpeg not found: {settings.ffmpeg_bin}")
    cmd = build_ffmpeg_command(settings, video, out_dir)
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.ffmpeg_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise IngestError("ffmpeg timed out") from exc
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-3:]
        raise IngestError(f"ffmpeg failed (exit {proc.returncode}): {' | '.join(tail)}")
    return sorted(out_dir.glob("frame_*.jpg"))


def ingest_video(
    session: Session, settings: Settings, video: Path, commit_every: int | None = 200
) -> IngestReport:
    """Extract frames at ``video_fps`` and ingest them; ``group_id`` = video file stem."""
    report = IngestReport()
    if video.suffix.lower() not in VIDEO_EXTENSIONS or video.is_symlink() or not video.is_file():
        raise IngestError(f"unsupported video: {video.name}")
    size = video.stat().st_size
    if size == 0 or size > settings.max_video_bytes:
        raise IngestError(f"video size out of range: {size} bytes")

    group = sanitize_group_id(video.stem)
    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        frames = extract_frames(settings, video, Path(tmp))
        for index, frame in enumerate(frames):
            try:
                _, created = register_image(
                    session,
                    settings,
                    frame,
                    source=ImageSource.UPLOAD_VIDEO,
                    group_id=group,
                    extra={
                        "video": video.name,
                        "frame_index": index,
                        "timestamp_s": round(index / settings.video_fps, 3),
                    },
                )
            except IngestError as exc:
                report.skipped += 1
                report.errors.append(f"{video.name}#{index}: {exc}")
                continue
            if created:
                report.added += 1
                _maybe_commit(session, report, commit_every)
            else:
                report.duplicates += 1
    return report


def ingest_videos_dir(session: Session, settings: Settings, root: Path) -> IngestReport:
    report = IngestReport()
    if not root.is_dir():
        raise IngestError(f"not a directory: {root}")
    for video in _iter_files(root, VIDEO_EXTENSIONS):
        try:
            report.merge(ingest_video(session, settings, video))
        except IngestError as exc:
            report.skipped += 1
            report.errors.append(f"{video.name}: {exc}")
            logger.error("video ingest failed (%s): %s", video.name, exc)
    return report


# --------------------------------------------------------------------------------------
# Library (pre-labelled public datasets)
# --------------------------------------------------------------------------------------
def get_or_create_library_labeler(session: Session) -> Labeler:
    labeler = session.scalar(select(Labeler).where(Labeler.oidc_sub == LIBRARY_LABELER_SUB))
    if labeler is None:
        labeler = Labeler(
            oidc_sub=LIBRARY_LABELER_SUB, email=None, display_name="library", is_admin=False
        )
        session.add(labeler)
        session.flush()
    return labeler


def ingest_library_image(
    session: Session,
    settings: Settings,
    path: Path,
    *,
    dataset: str,
    source_class: str,
    label: str,
    registry: SchemeRegistry,
) -> tuple[Image, bool]:
    """Ingest one pre-labelled library image with a label from the ``room`` scheme."""
    registry.get(ROOM_SCHEME).validate_label(label)
    image, created = register_image(
        session,
        settings,
        path,
        source=ImageSource.LIBRARY,
        group_id=f"library:{dataset}:{source_class}",
        extra={"dataset": dataset, "source_class": source_class},
        allowed_ext=LIBRARY_IMAGE_EXTENSIONS,
        registry=registry,
    )
    if created:
        labeler = get_or_create_library_labeler(session)
        session.add(
            Annotation(image_id=image.id, labeler_id=labeler.id, scheme=ROOM_SCHEME, label=label)
        )
        session.flush()
        recompute_consensus(session, image.id, ROOM_SCHEME, registry)
    return image, created
