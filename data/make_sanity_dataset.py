"""
Create a tiny synthetic image dataset for checking that the fAIre training
pipeline runs end-to-end.

This is NOT a real benchmark dataset. It creates simple generated images so the
repo can be tested without publishing private fire-scene photos or large files.
Use it as a smoke test for data prep, training, evaluation, and inference.

Example:
    python data/make_sanity_dataset.py --out data/raw_sanity --images-per-class 60
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageDraw, ImageFilter


ImageSize = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small synthetic fAIre sanity-check dataset")
    parser.add_argument("--out", default="data/raw_sanity", help="Output folder with distress/no_distress classes")
    parser.add_argument("--images-per-class", type=int, default=60)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_background(size: ImageSize) -> Image.Image:
    width, height = size
    base = Image.new("RGB", size, color=(random.randint(15, 45), random.randint(15, 45), random.randint(15, 45)))
    draw = ImageDraw.Draw(base)

    # Draw muted room/ground-like blocks so backgrounds are not identical.
    for _ in range(random.randint(8, 18)):
        x0 = random.randint(0, width - 1)
        y0 = random.randint(0, height - 1)
        x1 = min(width, x0 + random.randint(20, 90))
        y1 = min(height, y0 + random.randint(10, 70))
        shade = random.randint(25, 85)
        color = (shade, shade + random.randint(-10, 10), shade + random.randint(-10, 10))
        draw.rectangle([x0, y0, x1, y1], fill=color)

    return base.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))


def draw_flame_like_blob(image: Image.Image) -> Image.Image:
    """Draw a rough flame/smoke proxy for a positive-class sanity image."""
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    cx = random.randint(width // 3, 2 * width // 3)
    base_y = random.randint(height // 2, height - 20)
    flame_h = random.randint(height // 4, height // 2)
    flame_w = random.randint(width // 8, width // 4)

    # Outer orange/red blob.
    points = []
    for i in range(18):
        angle = 2 * math.pi * i / 18
        radius_x = flame_w * random.uniform(0.55, 1.15)
        radius_y = flame_h * random.uniform(0.45, 1.05)
        x = cx + math.cos(angle) * radius_x
        y = base_y - flame_h / 2 + math.sin(angle) * radius_y
        if y > base_y:
            y = base_y + random.randint(-8, 8)
        points.append((x, y))
    draw.polygon(points, fill=(220, 55, 20, 190))

    # Inner yellow core.
    inner = [
        (cx, base_y - flame_h * 0.90),
        (cx - flame_w * 0.45, base_y - flame_h * 0.20),
        (cx, base_y),
        (cx + flame_w * 0.45, base_y - flame_h * 0.20),
    ]
    draw.polygon(inner, fill=(255, 195, 40, 210))

    # Smoke haze above.
    for _ in range(random.randint(3, 7)):
        rx = random.randint(cx - flame_w, cx + flame_w)
        ry = random.randint(max(0, base_y - flame_h - 35), max(1, base_y - flame_h // 2))
        r = random.randint(12, 38)
        draw.ellipse([rx - r, ry - r, rx + r, ry + r], fill=(125, 125, 125, random.randint(40, 90)))

    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))


def draw_neutral_scene(image: Image.Image) -> Image.Image:
    """Draw a negative-class scene without fire-colored blobs."""
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    for _ in range(random.randint(3, 8)):
        x0 = random.randint(0, width - 40)
        y0 = random.randint(0, height - 40)
        x1 = min(width, x0 + random.randint(20, 75))
        y1 = min(height, y0 + random.randint(20, 75))
        shade = random.randint(35, 110)
        draw.rectangle([x0, y0, x1, y1], fill=(shade, shade, shade + random.randint(-15, 15), 150))
    return image


def save_class_images(out_dir: Path, class_name: str, count: int, size: int) -> None:
    class_dir = out_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        image = random_background((size, size))
        if class_name == "distress":
            image = draw_flame_like_blob(image)
        else:
            image = draw_neutral_scene(image)
        image.save(class_dir / f"{class_name}_{idx:04d}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    if args.images_per_class < 10:
        raise ValueError("Use at least 10 images per class so train/val/test splits are meaningful.")

    random.seed(args.seed)
    out_dir = Path(args.out)
    save_class_images(out_dir, "distress", args.images_per_class, args.size)
    save_class_images(out_dir, "no_distress", args.images_per_class, args.size)

    print("Created synthetic sanity-check dataset")
    print(f"  output: {out_dir}")
    print(f"  distress: {args.images_per_class}")
    print(f"  no_distress: {args.images_per_class}")
    print("Next:")
    print(f"  python data/prepare_data.py --raw {out_dir} --out data/sanity")
    print("  python data/validate_dataset.py --data data/sanity")


if __name__ == "__main__":
    main()
