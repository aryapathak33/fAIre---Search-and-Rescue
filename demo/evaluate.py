"""
Evaluate the trained fAIre classifier on a held-out test set.

Expected folder layout:
    data/test/distress/*.jpg
    data/test/no_distress/*.jpg

Run:
    python demo/evaluate.py --data data --weights models/fire_model.pt --threshold 0.50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fAIre classifier")
    parser.add_argument("--data", required=True, help="Dataset root containing test/")
    parser.add_argument("--weights", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None, help="Optional binary threshold for positive class")
    parser.add_argument("--positive-class", default=None, help="Class to treat as positive for binary thresholding")
    parser.add_argument("--out-dir", default="media", help="Where to save metrics and confusion matrix")
    return parser.parse_args()


def load_checkpoint(weights_path: str, device: str):
    checkpoint = torch.load(weights_path, map_location=device)
    classes = checkpoint["classes"]
    img_size = checkpoint.get("img_size", 224)
    mean = checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("imagenet_std", [0.229, 0.224, 0.225])

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, classes, img_size, mean, std


def infer_positive_class(classes: list[str], requested: str | None) -> int:
    if requested is not None:
        if requested not in classes:
            raise ValueError(f"positive class '{requested}' not found in classes: {classes}")
        return classes.index(requested)

    preferred_keywords = ["distress", "person", "human", "hazard", "fire"]
    for keyword in preferred_keywords:
        for idx, class_name in enumerate(classes):
            if keyword in class_name.lower() and "no" not in class_name.lower():
                return idx
    return 1 if len(classes) > 1 else 0


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, img_size, mean, std = load_checkpoint(args.weights, device)

    test_dir = Path(args.data) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected test folder at {test_dir}")

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_ds = datasets.ImageFolder(test_dir, transform=tf)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if test_ds.classes != classes:
        raise ValueError(f"Checkpoint classes {classes} do not match test classes {test_ds.classes}")

    y_true, y_pred, y_score = [], [], []
    positive_idx = infer_positive_class(classes, args.positive_class)

    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()

            if args.threshold is None or len(classes) != 2:
                preds = probs.argmax(dim=1)
            else:
                preds = (probs[:, positive_idx] >= args.threshold).long()
                if positive_idx == 0:
                    preds = 1 - preds

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs[:, positive_idx].tolist())

    print("\n=== Held-out test-set report ===")
    report_text = classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0)
    print(report_text)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False, xticks_rotation=30)
    ax.set_title("fAIre confusion matrix")
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=160)

    metrics = {
        "classes": classes,
        "positive_class": classes[positive_idx],
        "threshold": args.threshold,
        "test_images": len(test_ds),
        "per_class": {
            class_name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, class_name in enumerate(classes)
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved confusion matrix -> {cm_path}")
    print(f"Saved metrics JSON -> {metrics_path}")
    print("Put the measured precision/recall/F1 numbers into README.md.")


if __name__ == "__main__":
    main()
