"""
Train the fAIre image classifier with transfer learning.

Expected folder layout:
    data/
      train/
        distress/
        no_distress/
      val/
        distress/
        no_distress/

Run from repo root:
    python training/train.py --data data --epochs 10 --out models/fire_model.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fAIre transfer-learning classifier")
    parser.add_argument("--data", required=True, help="Dataset root containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/fire_model.pt", help="Output checkpoint path")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train final classifier layer")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def load_dataloaders(data_dir: str, batch_size: int, img_size: int) -> Tuple[DataLoader, DataLoader, list[str]]:
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected {train_dir} and {val_dir}. Run data/prepare_data.py or create ImageFolder folders."
        )

    train_tf, eval_tf = build_transforms(img_size)
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    if train_ds.classes != val_ds.classes:
        raise ValueError(f"Train classes {train_ds.classes} do not match val classes {val_ds.classes}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, train_ds.classes


def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: str,
) -> Dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    train_dl, val_dl, classes = load_dataloaders(args.data, args.batch_size, args.img_size)
    print(f"Classes: {classes}")
    print(f"Train images: {len(train_dl.dataset)} | Val images: {len(val_dl.dataset)}")

    model = build_model(num_classes=len(classes), freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, train_dl, criterion, optimizer, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_dl, criterion, optimizer=None, device=device)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(model, classes, args, best_val_acc, history)
            print(f"Saved best checkpoint -> {args.out}")

    print("\nDone. Next run:")
    print(f"python demo/evaluate.py --data {args.data} --weights {args.out}")


def save_checkpoint(
    model: nn.Module,
    classes: list[str],
    args: argparse.Namespace,
    best_val_acc: float,
    history: list[dict],
) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": "mobilenet_v2",
        "state_dict": model.state_dict(),
        "classes": classes,
        "img_size": args.img_size,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "best_val_accuracy": best_val_acc,
        "history": history,
    }
    torch.save(checkpoint, out_path)

    metadata_path = out_path.with_suffix(".json")
    metadata = {k: checkpoint[k] for k in ["model_name", "classes", "img_size", "best_val_accuracy"]}
    metadata["epochs_requested"] = args.epochs
    metadata["learning_rate"] = args.lr
    metadata["freeze_backbone"] = args.freeze_backbone
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
