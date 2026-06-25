"""
tune_threshold.py - Pick the best confidence threshold for a recall-first goal.

"Tuning" doesn't have to mean fancy hyperparameter search. The most decision-relevant
knob for a detector is the confidence threshold, and for FIRE the right objective is:
*catch as many people as possible (max recall) while keeping false alarms acceptable
(precision above a floor).* This script sweeps thresholds and picks that operating
point. It runs on the same bundled demo data as demo_metrics.py, so you can see it
work, then point the same idea at your real model's probabilities.

Run:
    python tune_threshold.py --min-precision 0.85
"""

import argparse
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import metrics_utils as mu

TARGET_CLASS = 3


def get_demo_probabilities():
    digits = load_digits()
    X, y = digits.data, (digits.target == TARGET_CLASS).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1], y_te


def tune(proba, y_true, min_precision):
    """Return the threshold giving the highest recall while precision >= floor."""
    best = None
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba >= t).astype(int)
        r = mu.report(y_true, pred)
        if r["precision"] >= min_precision:
            if best is None or r["recall"] > best["recall"]:
                best = {"threshold": round(float(t), 2), **r}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-precision", type=float, default=0.85,
                    help="Lowest acceptable precision (false-alarm tolerance)")
    args = ap.parse_args()

    proba, y_true = get_demo_probabilities()
    best = tune(proba, y_true, args.min_precision)

    if best is None:
        print(f"No threshold reached precision >= {args.min_precision}. Lower the floor.")
        return
    print(f"Recall-first tuning (precision floor = {args.min_precision}):")
    print(f"  chosen threshold : {best['threshold']}")
    print(f"  precision        : {best['precision']:.3f}")
    print(f"  recall           : {best['recall']:.3f}   <- maximized")
    print(f"  f1               : {best['f1']:.3f}")
    print(f"  caught {best['tp']} events, missed {best['fn']}, {best['fp']} false alarms")


if __name__ == "__main__":
    main()
