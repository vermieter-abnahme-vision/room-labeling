"""Export the datapack: manifest.jsonl (source of truth) + meta.json."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.labelschemes import ROOM_SCHEME
from app.services import export
from scripts._common import bootstrap

logger = logging.getLogger("scripts.export")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="export directory (default EXPORT_DIR)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ratios", default="0.8,0.1,0.1", help="train,val,test")
    parser.add_argument("--library-split", choices=["train", "group"], default="train")
    parser.add_argument(
        "--previous", type=Path, help="earlier manifest.jsonl: keep its group->split assignments"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ctx = bootstrap(args.verbose)
    s = ctx.settings
    options = export.ExportOptions(
        scheme=ROOM_SCHEME,
        ratios=export.parse_ratios(args.ratios),
        seed=args.seed,
        library_split=args.library_split,
        pseudonym_salt=s.pseudonym_salt,
        previous_assignments=export.assignments_from_manifest(args.previous)
        if args.previous
        else None,
    )
    with ctx.db.read_session() as session:
        manifest = export.build_manifest(session, options, ctx.registry)
    target = export.write_export(manifest, args.out or s.export_dir)
    logger.info("exported %d images to %s", len(manifest.rows), target)
    logger.info("per split: %s", manifest.meta["images_per_split"])
    for warning in manifest.meta["warnings"]:
        logger.warning(warning)
    return 0


if __name__ == "__main__":
    sys.exit(main())
