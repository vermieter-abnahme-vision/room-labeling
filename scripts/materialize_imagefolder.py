"""Create data/<split>/<label>/<sha>.jpg from a manifest (hardlinks by default).

The manifest stays authoritative; this folder is a disposable view for ImageFolder-style loaders.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app.services.export import materialize_imagefolder
from scripts._common import bootstrap

logger = logging.getLogger("scripts.materialize")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="path to manifest.jsonl")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--verify", action="store_true", help="re-hash every file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ctx = bootstrap(args.verbose)
    counts = materialize_imagefolder(
        args.manifest, ctx.settings.images_dir, args.out, args.mode, args.verify
    )
    logger.info("materialised into %s: %s", args.out, dict(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
