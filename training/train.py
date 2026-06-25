"""
train.py - Train / retrain the FIRE detection model.

This is a SKELETON for you to drop your real training code into.
Replace each `TODO` block with the actual code you wrote when you retrained
your Teachable Machine model in TensorFlow/PyTorch. The structure below is just
a clean shell (argument parsing, the standard train loop stages) so the file
reads well on GitHub — the logic should be yours.

Usage:
    python training/train.py --data data/ --epochs 30 --out models/fire_model
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Train the FIRE detection model")
    p.add_argument("--data", required=True, help="Path to training data directory")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", default="models/fire_model", help="Where to save weights")
    return p.parse_args()


def load_data(data_dir, batch_size):
    """TODO: paste your data-loading / augmentation code here.
    Return whatever your training step expects (e.g. train/val datasets)."""
    raise NotImplementedError("Add your real data loading code from your machine.")


def build_model():
    """TODO: paste the model definition you retrained (the CNN architecture
    you moved to from Teachable Machine)."""
    raise NotImplementedError("Add your real model definition.")


def train(model, train_data, val_data, epochs, lr):
    """TODO: paste your real training loop / .fit() call here."""
    raise NotImplementedError("Add your real training loop.")


def main():
    args = parse_args()
    train_data, val_data = load_data(args.data, args.batch_size)
    model = build_model()
    train(model, train_data, val_data, args.epochs, args.lr)
    # TODO: save weights to args.out
    print(f"Done. Save your trained weights to: {args.out}")


if __name__ == "__main__":
    main()
