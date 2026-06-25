"""
metrics_utils.py - Small, dependency-free helpers for detection metrics.

Pure Python so it's easy to read, easy to test, and works anywhere. These compute
the same precision / recall / F1 that scikit-learn does, but spelled out so you can
see (and explain) exactly what each number is. tests/test_metrics_utils.py checks
them against known values and cross-checks against scikit-learn.
"""

from dataclasses import dataclass


@dataclass
class Counts:
    tp: int  # true positives  - events correctly caught
    fp: int  # false positives - false alarms
    fn: int  # false negatives - missed events
    tn: int  # true negatives  - correctly ignored non-events


def counts_from_predictions(y_true, y_pred):
    """Tally TP/FP/FN/TN for binary labels (1 = event, 0 = not)."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return Counts(tp, fp, fn, tn)


def precision(c: Counts) -> float:
    """Of everything we flagged, how much was real. Guards against /0."""
    return c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0


def recall(c: Counts) -> float:
    """Of everything real, how much we caught. The one to prioritize for FIRE."""
    return c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0


def f1(c: Counts) -> float:
    """Harmonic mean of precision and recall."""
    p, r = precision(c), recall(c)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def report(y_true, y_pred) -> dict:
    """Convenience: return all three plus the raw counts as a dict."""
    c = counts_from_predictions(y_true, y_pred)
    return {"precision": precision(c), "recall": recall(c), "f1": f1(c),
            "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn}
