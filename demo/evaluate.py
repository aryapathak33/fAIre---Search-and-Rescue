"""
evaluate.py - Produce the REAL metrics for your trained classifier.

This is the script that generates the numbers you put in your README -- on YOUR
test set, measured honestly. It loads the model train_classifier.py saved, runs it
over a test/ folder, and prints precision / recall / F1 per class plus a confusion
matrix image. Same report style as demo/demo_metrics.py.

Folder layout it expects:
    data/test/
      distress/      *.jpg
      no_distress/   *.jpg

Install:  pip install torch torchvision scikit-learn matplotlib
Run:      python evaluate.py --data data --weights ../models/fire_model.pt
"""

import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained classifier")
    p.add_argument("--data", required=True, help="Dataset root containing a test/ subfolder")
    p.add_argument("--weights", required=True, help="Path to the saved .pt model")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.weights, map_location=device)
    classes = ckpt["classes"]

    weights = MobileNet_V2_Weights.DEFAULT
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(weights.meta["mean"], weights.meta["std"]),
    ])
    test_ds = datasets.ImageFolder(f"{args.data}/test", transform=eval_tf)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size)

    model = mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_dl:
            preds = model(imgs.to(device)).argmax(1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    print("\n=== Test-set report (these are the numbers for your README) ===")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=130)
    print("Saved confusion_matrix.png  (put this in media/ for your README)")


if __name__ == "__main__":
    main()
