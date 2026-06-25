"""
demo_metrics.py - A REAL, runnable demo of the evaluation pipeline you'll use on FIRE.

It does NOT use your fire data. It uses a small bundled image dataset (8x8 digit
images, ships with scikit-learn, no download) and sets up a *rare-event binary
detection* task -- "is this the target class or not" -- because that is exactly
the shape of the FIRE problem: distress signals are rare, so precision and recall
diverge and the confidence threshold matters.

Run it:
    python demo_metrics.py

It prints a precision/recall/F1 report, shows the precision/recall trade-off as
you move the threshold, and saves a confusion matrix image. This same machinery is
what evaluate.py runs on your real FIRE model -- only the data source changes.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, average_precision_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_CLASS = 3          # the "positive" / rare event we want to detect
RANDOM_STATE = 42


def main():
    # --- load a real (small) image dataset, offline ---
    digits = load_digits()
    X = digits.data                       # 1797 samples, 64 features (8x8 pixels)
    y = (digits.target == TARGET_CLASS).astype(int)   # 1 = target event, 0 = everything else
    pos_rate = y.mean()
    print(f"Dataset: {len(y)} images | positive (event) rate: {pos_rate:.1%} "
          f"(imbalanced, like a rare distress signal)\n")

    # --- split, keeping the class balance the same in train and test ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)

    # --- a simple, honest classifier ---
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    clf.fit(X_tr, y_tr)

    # probabilities for the positive class -> lets us move the threshold
    proba = clf.predict_proba(X_te)[:, 1]

    # ---------- metrics at the default 0.5 threshold ----------
    pred = (proba >= 0.5).astype(int)
    print("=== Report at default threshold (0.50) ===")
    print(classification_report(y_te, pred, target_names=["not-event", "EVENT"], digits=3))
    print(f"Average Precision (area under PR curve): {average_precision_score(y_te, proba):.3f}\n")

    # ---------- the precision / recall trade-off ----------
    print("=== How the threshold trades precision vs recall ===")
    print(f"{'threshold':>10} {'precision':>10} {'recall':>8} {'f1':>7}")
    for t in [0.30, 0.50, 0.70, 0.90]:
        p = (proba >= t).astype(int)
        print(f"{t:>10.2f} {precision_score(y_te, p, zero_division=0):>10.3f} "
              f"{recall_score(y_te, p):>8.3f} {f1_score(y_te, p):>7.3f}")
    print("\nNote: lower threshold -> catch more events (higher recall) but more false alarms")
    print("(lower precision). For FIRE you'd tune toward RECALL -- a miss is a person left behind.\n")

    # ---------- confusion matrix image ----------
    cm = confusion_matrix(y_te, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["not-event", "EVENT"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix (threshold 0.50)")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=130)
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
