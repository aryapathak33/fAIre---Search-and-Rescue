"""
train_classifier.py - Train an image classifier on YOUR images (transfer learning).

This is how you'd turn a folder of labeled images (e.g. distress / no-distress, or
smoke / no-smoke) into a real model. It starts from a network pretrained on millions
of images (MobileNetV2) and fine-tunes it on your small dataset -- the standard,
honest way to get good results from a few hundred images. This is the grown-up
version of what Teachable Machine did for you under the hood.

Folder layout it expects (ImageFolder format):
    data/
      train/
        distress/      *.jpg
        no_distress/   *.jpg
      val/
        distress/      *.jpg
        no_distress/   *.jpg

Install:  pip install torch torchvision
Run:      python train_classifier.py --data data --epochs 10 --out ../models/fire_model.pt
Tip:      no GPU? Use Google Colab (free GPU) -- upload your data folder and run there.
"""

import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train a transfer-learning image classifier")
    p.add_argument("--data", required=True, help="Dataset root with train/ and val/ subfolders")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="../models/fire_model.pt")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    weights = MobileNet_V2_Weights.DEFAULT
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),       # simple augmentation
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(weights.meta["mean"], weights.meta["std"]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(weights.meta["mean"], weights.meta["std"]),
    ])

    train_ds = datasets.ImageFolder(f"{args.data}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{args.data}/val", transform=eval_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)
    classes = train_ds.classes
    print(f"Classes: {classes}  | train={len(train_ds)}  val={len(val_ds)}")

    # start from pretrained, replace the final layer for our classes
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        train_loss = running / len(train_ds)

        # quick val accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"epoch {epoch:>2}/{args.epochs}  train_loss={train_loss:.3f}  "
              f"val_acc={correct/total:.3f}")

    torch.save({"state_dict": model.state_dict(), "classes": classes}, args.out)
    print(f"Saved model -> {args.out}")
    print("Now run evaluate.py on your test set to get precision/recall/F1.")


if __name__ == "__main__":
    main()
