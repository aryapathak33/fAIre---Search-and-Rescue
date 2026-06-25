"""
prepare_data.py - Prepare / preprocess the dataset for training.

SKELETON: paste your real data-prep code into the TODO blocks (resizing,
labeling, train/val split, augmentation, etc.). Keep large raw data OUT of git
(see .gitignore) and store a few small examples in data/samples/ so the repo
still demonstrates the format.

Usage:
    python data/prepare_data.py --raw data/raw --out data/processed
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Prepare FIRE training data")
    p.add_argument("--raw", required=True, help="Raw data directory")
    p.add_argument("--out", required=True, help="Output (processed) directory")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--val-split", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    # TODO: paste your real preprocessing pipeline here.
    raise NotImplementedError("Add your real data-prep code from your machine.")


if __name__ == "__main__":
    main()
