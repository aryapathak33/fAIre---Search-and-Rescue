"""
Tests for demo/metrics_utils.py - real, passing unit tests.

Run from the repo root:
    pip install pytest
    pytest -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))

import metrics_utils as m


def test_counts_basic():
    # y_true:  1 1 1 0 0
    # y_pred:  1 0 1 1 0   -> tp=2, fn=1, fp=1, tn=1
    c = m.counts_from_predictions([1, 1, 1, 0, 0], [1, 0, 1, 1, 0])
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 1)


def test_precision_recall_f1_known_values():
    c = m.counts_from_predictions([1, 1, 1, 0, 0], [1, 0, 1, 1, 0])
    assert abs(m.precision(c) - 2 / 3) < 1e-9   # 2 caught / 3 flagged
    assert abs(m.recall(c) - 2 / 3) < 1e-9      # 2 caught / 3 real
    assert abs(m.f1(c) - 2 / 3) < 1e-9


def test_no_divide_by_zero():
    # nothing predicted positive -> precision defined as 0, no crash
    c = m.counts_from_predictions([0, 0, 0], [0, 0, 0])
    assert m.precision(c) == 0.0
    assert m.recall(c) == 0.0
    assert m.f1(c) == 0.0


def test_matches_sklearn():
    """Cross-check our hand-written metrics against scikit-learn."""
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    c = m.counts_from_predictions(y_true, y_pred)
    assert abs(m.precision(c) - precision_score(y_true, y_pred)) < 1e-9
    assert abs(m.recall(c) - recall_score(y_true, y_pred)) < 1e-9
    assert abs(m.f1(c) - f1_score(y_true, y_pred)) < 1e-9


def test_report_keys():
    r = m.report([1, 0, 1], [1, 0, 0])
    assert set(r) == {"precision", "recall", "f1", "tp", "fp", "fn", "tn"}
