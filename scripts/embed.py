"""Compute SigLIP image embeddings for all images (cached on disk, key = sha256)."""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from app.models import Image
from app.services.inference import (
    EmbeddingCache,
    SigLipEmbedder,
    compute_missing_embeddings,
    model_tag,
)
from scripts._common import bootstrap

logger = logging.getLogger("scripts.embed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device", default="auto", help="auto (cuda > mps > cpu) | cpu | cuda | mps"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ctx = bootstrap(args.verbose)
    s = ctx.settings
    cache = EmbeddingCache(s.embed_cache_dir, model_tag(s.siglip_model, s.siglip_revision))
    with ctx.db.read_session() as session:
        items = [
            (sha, s.images_dir / path)
            for sha, path in session.execute(select(Image.sha256, Image.path))
        ]
    missing = sum(1 for sha, _ in items if not cache.has(sha))
    logger.info("%d images, %d to embed", len(items), missing)
    if not missing:
        return 0

    embedder = SigLipEmbedder(s.siglip_model, s.siglip_revision, args.device)
    done, failed = compute_missing_embeddings(
        embedder,
        cache,
        items,
        args.batch_size,
        progress=lambda i, n: logger.info("embedded %d / %d", i, n),
    )
    logger.info("embedded %d images, %d failed", done, len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
