"""
Prepare fAIre image data for training.

Input format:
    data/raw/
      distress/*.jpg
      no_distress/*.jpg

Output format:
    data/
      train/distress/*.jpg
      val/distress/*.jpg
      test/distress/*.jpg
      ...

Run:
    python data/prepare_data.py --raw data/raw --out data --val-split 0.15 --test-split 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split and resize fAIre dataset")
    parser.add_argument("--raw", required=True, help="Raw dataset folder with one subfolder per class")
    parser.add_argument("--out", required=True, help="Output dataset root")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-original", action="store_true", help="Copy without resizing/converting")
    return parser.parse_args()


def iter_images(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def save_resized_image(src: Path, dst: Path, img_size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((img_size, img_size))
        canvas = Image.new("RGB", (img_size, img_size), color=(0, 0, 0))
        x = (img_size - img.width) // 2
        y = (img_size - img.height) // 2
        canvas.paste(img, (x, y))
        canvas.save(dst.with_suffix(".jpg"), quality=92)


def split_files(files: list[Path], val_split: float, test_split: float) -> dict[str, list[Path]]:
    n_total = len(files)
    n_test = int(round(n_total * test_split))
    n_val = int(round(n_total * val_split))

    test_files = files[:n_test]
    val_files = files[n_test : n_test + n_val]
    train_files = files[n_test + n_val :]

    return {"train": train_files, "val": val_files, "test": test_files}


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw folder not found: {raw_dir}")
    if args.val_split + args.test_split >= 1.0:
        raise ValueError("val_split + test_split must be less than 1.0")

    random.seed(args.seed)
    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found inside {raw_dir}")

    summary = {}
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        files = list(iter_images(class_dir))
        random.shuffle(files)
        if len(files) < 5:
            print(f"Warning: class '{class_name}' only has {len(files)} images")

        splits = split_files(files, args.val_split, args.test_split)
        summary[class_name] = {split: len(paths) for split, paths in splits.items()}

        for split, paths in splits.items():
            for idx, src in enumerate(paths):
                dst = out_dir / split / class_name / f"{class_name}_{idx:05d}.jpg"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if args.copy_original:
                    shutil.copy2(src, dst)
                else:
                    save_resized_image(src, dst, args.img_size)

    print("Prepared dataset:")
    for class_name, counts in summary.items():
        print(f"  {class_name}: train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"Output -> {out_dir}")


if __name__ == "__main__":
    main()
