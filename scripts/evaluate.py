"""Evaluate the probe honestly on a group-disjoint split of a datapack manifest.

Trains on the train (+val) rows of the manifest and scores on the test rows - never on images
from the same apartment/video. Optionally compares against SigLIP zero-shot.

    make evaluate MANIFEST=var/exports/<stamp>/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from app.labelschemes import ROOM_SCHEME
from app.services import inference
from app.services.evaluation import evaluate_predictions, format_report
from app.services.export import read_manifest
from scripts._common import bootstrap

logger = logging.getLogger("scripts.evaluate")
MIN_TEST_SAMPLES = 30


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="manifest.jsonl from an export")
    parser.add_argument("--train-splits", default="train,val", help="comma separated")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--zero-shot", action="store_true", help="also score SigLIP zero-shot")
    parser.add_argument("--device", default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    ctx = bootstrap(args.verbose, check_schema=False)
    s = ctx.settings
    scheme = ctx.registry.get(ROOM_SCHEME)
    cache = inference.EmbeddingCache(
        s.embed_cache_dir, inference.model_tag(s.siglip_model, s.siglip_revision)
    )

    train_splits = {x.strip() for x in args.train_splits.split(",")}
    rows = read_manifest(args.manifest)
    train = [r for r in rows if r["suggested_split"] in train_splits and cache.has(r["sha256"])]
    test = [r for r in rows if r["suggested_split"] == args.test_split and cache.has(r["sha256"])]
    skipped = len(rows) - sum(1 for r in rows if cache.has(r["sha256"]))
    if skipped:
        logger.warning("%d manifest images have no embedding (run `make embed`)", skipped)
    if not train or not test:
        logger.error(
            "empty train (%d) or test (%d) set - export more labelled data", len(train), len(test)
        )
        return 1
    if len(test) < MIN_TEST_SAMPLES:
        logger.warning("only %d test images - metrics will be very noisy", len(test))

    x_train = np.stack([cache.load(r["sha256"]) for r in train])
    x_test = np.stack([cache.load(r["sha256"]) for r in test])
    y_test = [r["final_label"] for r in test]
    clf, result = inference.train_probe(
        x_train,
        [r["final_label"] for r in train],
        [r["group_id"] for r in train],
        None,
        version="eval",
    )
    pred = clf.predict(x_test)
    probe_report = evaluate_predictions(y_test, [str(p) for p in pred], scheme.labels)
    print(format_report(probe_report, f"probe (train on {len(train)} imgs, test on {len(test)})"))

    output: dict[str, object] = {
        "manifest": str(args.manifest),
        "train_images": len(train),
        "test_images": len(test),
        "probe": {"accuracy": probe_report.accuracy, "macro_f1": probe_report.macro_f1},
    }
    if args.zero_shot:
        embedder = inference.SigLipEmbedder(s.siglip_model, s.siglip_revision, args.device)
        probs, classes = inference.zero_shot_probabilities(embedder, scheme, x_test)
        zs_pred = [classes[int(i)] for i in probs.argmax(axis=1)]
        zs_report = evaluate_predictions(y_test, zs_pred, scheme.labels)
        print()
        print(format_report(zs_report, "SigLIP zero-shot (same test images)"))
        output["zero_shot"] = {"accuracy": zs_report.accuracy, "macro_f1": zs_report.macro_f1}

    target = args.manifest.with_name("evaluation.json")
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    logger.info("saved %s (trained on %d samples)", target, result.n_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
