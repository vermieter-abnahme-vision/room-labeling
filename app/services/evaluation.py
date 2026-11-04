"""Model evaluation on the datapack: honest metrics on a group-disjoint split."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    support: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class EvalReport:
    n: int
    accuracy: float
    macro_f1: float
    per_class: list[ClassMetrics]
    labels: list[str]
    confusion: list[list[int]]  # rows = true label, cols = predicted label
    majority_baseline: float


def _div(num: float, den: float) -> float | None:
    return num / den if den else None


def evaluate_predictions(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> EvalReport:
    """Accuracy, per-class precision/recall/F1, macro-F1 (classes with support), confusion."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("need equally long, non-empty label sequences")
    present = [lab for lab in labels if lab in set(y_true) | set(y_pred)]
    index = {lab: i for i, lab in enumerate(present)}
    confusion = [[0] * len(present) for _ in present]
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[index[t]][index[p]] += 1

    per_class: list[ClassMetrics] = []
    f1_values: list[float] = []
    for lab in present:
        i = index[lab]
        tp = confusion[i][i]
        predicted = sum(row[i] for row in confusion)
        support = sum(confusion[i])
        precision, recall = _div(tp, predicted), _div(tp, support)
        f1 = _div(2 * precision * recall, precision + recall) if precision and recall else 0.0
        per_class.append(ClassMetrics(lab, support, precision, recall, f1))
        if support:
            f1_values.append(f1 or 0.0)
    correct = sum(confusion[i][i] for i in range(len(present)))
    majority = Counter(y_true).most_common(1)[0][1] / len(y_true)
    return EvalReport(
        n=len(y_true),
        accuracy=correct / len(y_true),
        macro_f1=sum(f1_values) / len(f1_values) if f1_values else 0.0,
        per_class=per_class,
        labels=present,
        confusion=confusion,
        majority_baseline=majority,
    )


def format_report(report: EvalReport, title: str) -> str:
    def pct(value: float | None) -> str:
        return "  -  " if value is None else f"{value * 100:5.1f}"

    lines = [
        f"== {title} ==",
        f"n={report.n}  accuracy={report.accuracy * 100:.1f}%  "
        f"macro-F1={report.macro_f1 * 100:.1f}%  "
        f"(majority-class baseline {report.majority_baseline * 100:.1f}%)",
        f"{'label':<14}{'support':>8}{'prec%':>8}{'rec%':>8}{'f1%':>8}",
    ]
    for c in report.per_class:
        lines.append(
            f"{c.label:<14}{c.support:>8}{pct(c.precision):>8}{pct(c.recall):>8}{pct(c.f1):>8}"
        )
    width = max(len(lab) for lab in report.labels) + 1
    lines.append("confusion (rows=true, cols=predicted):")
    lines.append(" " * 14 + " ".join(lab[:8].rjust(8) for lab in report.labels))
    for lab, row in zip(report.labels, report.confusion, strict=True):
        lines.append(lab.ljust(max(14, width)) + " ".join(str(v).rjust(8) for v in row))
    return "\n".join(lines)
