from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.services import inference as inf
from app.services.evaluation import evaluate_predictions, format_report
from scripts import evaluate as evaluate_script


def test_metrics_known_values() -> None:
    y_true = ["a", "a", "a", "b", "b", "c"]
    y_pred = ["a", "a", "b", "b", "b", "a"]
    r = evaluate_predictions(y_true, y_pred, ["a", "b", "c", "d"])
    assert r.n == 6 and r.accuracy == pytest.approx(4 / 6)
    by = {c.label: c for c in r.per_class}
    assert by["a"].precision == pytest.approx(2 / 3) and by["a"].recall == pytest.approx(2 / 3)
    assert by["b"].precision == pytest.approx(2 / 3) and by["b"].recall == 1.0
    assert by["c"].recall == 0.0 and by["c"].f1 == 0.0
    assert "d" not in r.labels  # absent class is not reported
    assert r.macro_f1 == pytest.approx((2 / 3 + 0.8 + 0.0) / 3)
    assert r.majority_baseline == pytest.approx(0.5)
    assert r.confusion[0] == [2, 1, 0]
    assert "accuracy=66.7%" in format_report(r, "t")
    with pytest.raises(ValueError):
        evaluate_predictions([], [], ["a"])


def test_eval_mask_scores_only_real_samples() -> None:
    rng = np.random.default_rng(1)
    x = np.vstack([np.eye(3, dtype=np.float32)[i % 3] + rng.normal(0, 0.05, 3) for i in range(60)])
    y = [["a", "b", "c"][i % 3] for i in range(60)]
    groups = [f"g{i % 6}" for i in range(60)]
    _, with_mask = inf.train_probe(x, y, groups, None, eval_mask=[i < 20 for i in range(60)])
    assert with_mask.cv_accuracy is not None and with_mask.model_path == Path("(not saved)")
    _, tiny = inf.train_probe(x, y, groups, None, eval_mask=[i < 5 for i in range(60)])
    assert tiny.cv_accuracy is None  # too few scored samples to report a number


def test_evaluate_script_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("EMBED_CACHE_DIR", str(cache_dir))
    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    cache = inf.EmbeddingCache(cache_dir, inf.model_tag(s.siglip_model, s.siglip_revision))
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        k = i % 3
        sha = f"{i:064x}"
        vec = np.eye(8, dtype=np.float32)[k] + rng.normal(0, 0.05, 8).astype(np.float32)
        cache.save(sha, vec)
        rows.append(
            {
                "sha256": sha,
                "group_id": f"g{i % 8}",
                "final_label": ["bad", "flur", "kueche"][k],
                "suggested_split": "test" if i % 8 == 7 else "train",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    try:
        assert evaluate_script.main([str(manifest)]) == 0
        result = json.loads((tmp_path / "evaluation.json").read_text())
        assert result["test_images"] == 10 and result["probe"]["accuracy"] > 0.9
    finally:
        get_settings.cache_clear()
