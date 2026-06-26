"""
Sweep confidence thresholds to choose a recall-first operating point.

This script demonstrates how to select a confidence threshold for a recall-first
operating point. The bundled demo keeps the idea runnable without a private dataset.

Run demo:
    python demo/tune_threshold.py --demo --min-precision 0.80
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune recall-first threshold")
    parser.add_argument("--demo", action="store_true", help="Run on bundled sklearn digits demo")
    parser.add_argument("--min-precision", type=float, default=0.80)
    return parser.parse_args()


def get_demo_scores():
    digits = load_digits()
    x = digits.data
    y = (digits.target == 3).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    return y_test, scores


def tune_threshold(y_true, scores, min_precision: float):
    best = None
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (scores >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        row = {
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if precision >= min_precision and (best is None or recall > best["recall"]):
            best = row
    return best


def main() -> None:
    args = parse_args()
    if not args.demo:
        raise SystemExit("Use --demo to run the built-in threshold tuning example.")

    y_true, scores = get_demo_scores()
    best = tune_threshold(y_true, scores, args.min_precision)
    if best is None:
        print(f"No threshold met min precision {args.min_precision:.2f}")
        return

    print("Recall-first threshold choice")
    print(f"minimum precision: {args.min_precision:.2f}")
    print(f"threshold: {best['threshold']:.2f}")
    print(f"precision: {best['precision']:.3f}")
    print(f"recall: {best['recall']:.3f}")
    print(f"f1: {best['f1']:.3f}")


if __name__ == "__main__":
    main()
