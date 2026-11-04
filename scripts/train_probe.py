"""Train the logistic-regression probe on labelled embeddings and write predictions for ALL images.

Falls back to SigLIP zero-shot (text prompts) when there is not yet enough labelled data.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
from sqlalchemy import select

from app.labelschemes import ROOM_SCHEME
from app.models import Image
from app.services import inference
from scripts._common import bootstrap

logger = logging.getLogger("scripts.train_probe")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-shot", action="store_true", help="force zero-shot predictions")
    parser.add_argument(
        "--include-unfinalized",
        action="store_true",
        help="also train on single/unanimous human labels of not yet final images",
    )
    parser.add_argument("--c", type=float, default=1.0, help="inverse L2 strength")
    parser.add_argument("--device", default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ctx = bootstrap(args.verbose)
    s = ctx.settings
    scheme = ctx.registry.get(ROOM_SCHEME)
    cache = inference.EmbeddingCache(
        s.embed_cache_dir, inference.model_tag(s.siglip_model, s.siglip_revision)
    )

    with ctx.db.read_session() as session:
        images = [(i.id, i.sha256) for i in session.execute(select(Image.id, Image.sha256))]
        samples = inference.training_samples(session, scheme, args.include_unfinalized)

    embedded = [(iid, sha) for iid, sha in images if cache.has(sha)]
    if len(embedded) < len(images):
        logger.warning(
            "%d images have no embedding yet (run scripts.embed)", len(images) - len(embedded)
        )
    if not embedded:
        logger.error("no embeddings found – run `make embed` first")
        return 1
    matrix = np.stack([cache.load(sha) for _, sha in embedded])

    version: str
    probs: inference.Array
    classes: list[str]
    use_zero_shot = args.zero_shot
    if not use_zero_shot:
        usable = [x for x in samples if cache.has(x.sha256)]
        try:
            clf, result = inference.train_probe(
                np.stack([cache.load(x.sha256) for x in usable])
                if usable
                else np.empty((0, matrix.shape[1]), np.float32),
                [x.label for x in usable],
                [x.group_id for x in usable],
                s.models_dir,
                c=args.c,
            )
        except inference.NotEnoughDataError as exc:
            logger.warning("probe not trainable (%s) -> zero-shot fallback", exc)
            use_zero_shot = True
        else:
            version, classes = result.version, result.classes
            probs = clf.predict_proba(matrix).astype(np.float32)
            logger.info(
                "probe %s: %d samples, group-CV accuracy=%s, saved to %s",
                version,
                result.n_samples,
                f"{result.cv_accuracy:.3f}" if result.cv_accuracy is not None else "n/a",
                result.model_path,
            )
    if use_zero_shot:
        embedder = inference.SigLipEmbedder(s.siglip_model, s.siglip_revision, args.device)
        probs, classes = inference.zero_shot_probabilities(embedder, scheme, matrix)
        version = inference.new_model_version("zeroshot")
        logger.info("zero-shot predictions with prompts for %s", classes)

    with ctx.db.write_session() as session:
        n = inference.write_predictions(
            session,
            scheme=scheme.name,
            version=version,
            image_ids=[iid for iid, _ in embedded],
            probs=probs,
            classes=classes,
        )
    logger.info("wrote %d predictions as model_version=%s", n, version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
