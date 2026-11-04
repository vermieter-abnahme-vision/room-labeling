from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.labelschemes import InvalidLabelError, SchemeRegistry
from app.models import Annotation, Consensus, ConsensusStatus, Image, ImageSource
from app.services import ingest
from tests.conftest import write_jpeg


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_sanitize_group_id() -> None:
    assert ingest.sanitize_group_id("../../etc/passwd") == "etc_passwd"
    assert "/" not in ingest.sanitize_group_id("a/b\\c")
    assert ingest.sanitize_group_id("  Wohnung 1 ") == "Wohnung_1"
    assert ingest.sanitize_group_id("...") == "unnamed"
    assert len(ingest.sanitize_group_id("x" * 1000)) == 200


def test_folder_ingest_dedups_by_sha256(session: Session, settings: Settings) -> None:
    root = settings.images_in_dir
    write_jpeg(root / "wohnung_a" / "1.jpg", (200, 10, 10))
    write_jpeg(root / "wohnung_a" / "2.jpg", (10, 200, 10))
    write_jpeg(root / "wohnung_b" / "x.jpg", (10, 10, 200))
    shutil.copyfile(root / "wohnung_a" / "1.jpg", root / "wohnung_b" / "dup.jpg")  # same bytes
    write_jpeg(root / "top.jpg", (90, 90, 90))

    report = ingest.ingest_folder(session, settings, root)
    assert (report.added, report.duplicates, report.skipped) == (4, 1, 0)
    groups = {i.group_id for i in session.scalars(select(Image))}
    assert groups == {"wohnung_a", "wohnung_b", "root"}

    again = ingest.ingest_folder(session, settings, root)  # idempotent
    assert (again.added, again.duplicates) == (0, 5)
    assert _count(session, Image) == 4
    assert _count(session, Consensus) == 4  # every image gets an open consensus row
    img = session.scalars(select(Image)).first()
    assert img is not None
    stored = settings.images_dir / img.path
    assert stored.is_file() and stored.name == f"{img.sha256}.jpg"
    assert ingest.sha256_file(stored) == img.sha256


def test_folder_ingest_rejects_invalid_files(session: Session, settings: Settings) -> None:
    root = settings.images_in_dir
    (root / "g").mkdir(parents=True)
    (root / "g" / "fake.jpg").write_bytes(b"not an image at all")
    (root / "g" / "empty.jpg").write_bytes(b"")
    (root / "g" / "notes.txt").write_text("ignored")
    write_jpeg(root / "g" / "ok.jpg", (1, 2, 3))
    report = ingest.ingest_folder(session, settings, root)
    assert report.added == 1 and report.skipped == 2 and len(report.errors) == 2


def test_oversized_image_is_rejected(session: Session, settings: Settings) -> None:
    small = settings.model_copy(update={"max_image_bytes": 100})
    write_jpeg(settings.images_in_dir / "g" / "a.jpg", (5, 5, 5))
    report = ingest.ingest_folder(session, small, settings.images_in_dir)
    assert report.added == 0 and report.skipped == 1


def test_symlinks_are_not_followed(session: Session, settings: Settings, tmp_path: Path) -> None:
    outside = write_jpeg(tmp_path / "outside" / "secret.jpg", (7, 7, 7))
    (settings.images_in_dir / "g").mkdir(parents=True)
    (settings.images_in_dir / "g" / "link.jpg").symlink_to(outside)
    report = ingest.ingest_folder(session, settings, settings.images_in_dir)
    assert report.added == 0


def test_ffmpeg_command_is_argument_list_without_network(
    settings: Settings, tmp_path: Path
) -> None:
    cmd = ingest.build_ffmpeg_command(settings, tmp_path / "v.mp4", tmp_path)
    assert isinstance(cmd, list) and all(isinstance(c, str) for c in cmd)
    assert cmd[cmd.index("-protocol_whitelist") + 1] == "file"
    assert any(c.startswith("fps=1.0") for c in cmd)
    assert Path(cmd[cmd.index("-i") + 1]).is_absolute()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_ingest_extracts_frames(session: Session, settings: Settings, tmp_path: Path) -> None:
    video = settings.videos_in_dir / "Wohnung 7.mp4"
    video.parent.mkdir(parents=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=5",
            "-t",
            "3",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    report = ingest.ingest_video(session, settings, video)
    assert report.added >= 2 and report.skipped == 0
    images = list(session.scalars(select(Image)))
    assert {i.group_id for i in images} == {"Wohnung_7"}
    assert {i.source for i in images} == {ImageSource.UPLOAD_VIDEO}
    assert images[0].extra is not None and "timestamp_s" in images[0].extra


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_broken_video_fails_cleanly(session: Session, settings: Settings) -> None:
    bad = settings.videos_in_dir / "broken.mp4"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\x00" * 2048)
    with pytest.raises(ingest.IngestError, match="ffmpeg failed"):
        ingest.ingest_video(session, settings, bad)
    report = ingest.ingest_videos_dir(session, settings, settings.videos_in_dir)
    assert report.skipped == 1 and report.errors


def test_missing_ffmpeg_binary(session: Session, settings: Settings, tmp_path: Path) -> None:
    cfg = settings.model_copy(update={"ffmpeg_bin": "definitely-not-ffmpeg"})
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    with pytest.raises(ingest.IngestError, match="ffmpeg not found"):
        ingest.ingest_video(session, cfg, video)


def test_library_ingest_creates_trusted_final_label(
    session: Session, settings: Settings, registry: SchemeRegistry, tmp_path: Path
) -> None:
    src = write_jpeg(tmp_path / "lib" / "k1.jpg", (123, 45, 67))
    image, created = ingest.ingest_library_image(
        session,
        settings,
        src,
        dataset="places365",
        source_class="kitchen",
        label="kueche",
        registry=registry,
    )
    assert created and image.source is ImageSource.LIBRARY
    assert image.group_id == "library:places365:kitchen"
    cons = session.get(Consensus, (image.id, "room"))
    assert cons is not None and (cons.status, cons.final_label) == (ConsensusStatus.FINAL, "kueche")
    assert _count(session, Annotation) == 1
    _, created_again = ingest.ingest_library_image(
        session,
        settings,
        src,
        dataset="places365",
        source_class="kitchen",
        label="kueche",
        registry=registry,
    )
    assert not created_again and _count(session, Annotation) == 1
    with pytest.raises(InvalidLabelError):
        ingest.ingest_library_image(
            session, settings, src, dataset="x", source_class="y", label="garage", registry=registry
        )
