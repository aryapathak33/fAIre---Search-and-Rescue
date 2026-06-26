"""
Validate an ImageFolder-style fAIre dataset before training.

Checks:
- train/val/test folders exist
- class names match across splits
- images can be opened by Pillow
- class counts are shown clearly
- exact duplicate files are flagged by SHA-1 hash

Example:
    python data/validate_dataset.py --data data
    python data/validate_dataset.py --data data/sanity
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageOps


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fAIre ImageFolder dataset")
    parser.add_argument("--data", required=True, help="Dataset root containing train/val/test")
    parser.add_argument("--min-per-class", type=int, default=5, help="Warn if any split/class has fewer images")
    return parser.parse_args()


def iter_images(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            yield path


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> Tuple[bool, str]:
    try:
        with Image.open(path) as image:
            ImageOps.exif_transpose(image).verify()
        return True, ""
    except Exception as exc:  # noqa: BLE001 - intentionally reports corrupt image details
        return False, str(exc)


def class_names(split_dir: Path) -> List[str]:
    return sorted(path.name for path in split_dir.iterdir() if path.is_dir())


def main() -> None:
    args = parse_args()
    root = Path(args.data)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    missing = [split for split in SPLITS if not (root / split).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required split folders: {missing}")

    classes_by_split = {split: class_names(root / split) for split in SPLITS}
    reference_classes = classes_by_split["train"]
    if not reference_classes:
        raise ValueError(f"No class folders found in {root / 'train'}")

    for split, classes in classes_by_split.items():
        if classes != reference_classes:
            raise ValueError(
                f"Class mismatch in {split}: {classes}. Expected same classes as train: {reference_classes}"
            )

    counts: Dict[str, Dict[str, int]] = {split: {} for split in SPLITS}
    hash_to_paths: Dict[str, List[Path]] = defaultdict(list)
    corrupt: List[Tuple[Path, str]] = []

    for split in SPLITS:
        for class_name in reference_classes:
            paths = list(iter_images(root / split / class_name))
            counts[split][class_name] = len(paths)
            for image_path in paths:
                ok, message = validate_image(image_path)
                if not ok:
                    corrupt.append((image_path, message))
                else:
                    hash_to_paths[sha1_file(image_path)].append(image_path)

    print("Dataset validation report")
    print(f"root: {root}")
    print(f"classes: {', '.join(reference_classes)}")
    print()
    print("counts:")
    for split in SPLITS:
        print(f"  {split}:")
        for class_name in reference_classes:
            count = counts[split][class_name]
            warning = "  <-- low count" if count < args.min_per_class else ""
            print(f"    {class_name}: {count}{warning}")

    duplicates = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    if duplicates:
        print("\nduplicate files detected:")
        for paths in duplicates[:10]:
            print("  " + " | ".join(str(path) for path in paths))
        if len(duplicates) > 10:
            print(f"  ... {len(duplicates) - 10} more duplicate groups")
    else:
        print("\nduplicate files detected: none")

    if corrupt:
        print("\ncorrupt/unreadable images:")
        for path, message in corrupt:
            print(f"  {path}: {message}")
        raise SystemExit(1)

    print("\nOK: dataset is trainable by training/train.py")


if __name__ == "__main__":
    main()
