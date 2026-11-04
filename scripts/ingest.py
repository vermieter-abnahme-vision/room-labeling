"""Ingest folder images (./images_in) and/or videos (./videos_in) into the image store."""

from __future__ import annotations

import argparse
import logging
import sys

from app.services import ingest
from scripts._common import bootstrap

logger = logging.getLogger("scripts.ingest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "folder", "video"], default="all")
    parser.add_argument("--fps", type=float, help="override VIDEO_FPS for this run")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ctx = bootstrap(args.verbose)
    settings = ctx.settings
    if args.fps:
        settings = settings.model_copy(update={"video_fps": args.fps})

    total = ingest.IngestReport()
    with ctx.db.write_session() as session:
        if args.only in ("all", "folder") and settings.images_in_dir.is_dir():
            total.merge(ingest.ingest_folder(session, settings, settings.images_in_dir))
        if args.only in ("all", "video") and settings.videos_in_dir.is_dir():
            total.merge(ingest.ingest_videos_dir(session, settings, settings.videos_in_dir))

    logger.info(
        "done: %d added, %d duplicates, %d skipped", total.added, total.duplicates, total.skipped
    )
    for err in total.errors:
        logger.warning("skipped: %s", err)
    return 1 if total.errors and not total.added else 0


if __name__ == "__main__":
    sys.exit(main())
