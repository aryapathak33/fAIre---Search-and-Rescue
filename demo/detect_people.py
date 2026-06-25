"""
detect_people.py - Detect people in an image (the FIRE-relevant demo).

Uses a model pretrained on COCO to find people and draw boxes. This is a real,
working detector you can run today on a single image -- the closest simple stand-in
for "find a person in the frame." It runs on CPU (slower) or GPU.

First run downloads the pretrained weights (~a few hundred MB) automatically.

Install:
    pip install torch torchvision pillow
Run:
    python detect_people.py --input some_photo.jpg --out detected.jpg --conf 0.7

What to notice: each detection has a confidence score. Raising --conf gives fewer,
surer boxes (higher precision, lower recall); lowering it catches more but with more
false boxes. That is the same precision/recall trade-off demo_metrics.py shows.
"""

import argparse
import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image, write_jpeg


def parse_args():
    p = argparse.ArgumentParser(description="Detect people in an image")
    p.add_argument("--input", required=True, help="Path to an input image")
    p.add_argument("--out", default="detected.jpg", help="Where to save the annotated image")
    p.add_argument("--conf", type=float, default=0.7, help="Confidence threshold (0-1)")
    return p.parse_args()


def main():
    args = parse_args()

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    categories = weights.meta["categories"]          # COCO labels; "person" is index 1
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    img = read_image(args.input)                     # uint8 tensor [3, H, W]
    batch = [preprocess(img)]

    with torch.no_grad():
        prediction = model(batch)[0]

    # keep only people above the confidence threshold
    keep = []
    for box, label, score in zip(prediction["boxes"],
                                 prediction["labels"],
                                 prediction["scores"]):
        if categories[label] == "person" and score >= args.conf:
            keep.append((box, float(score)))

    print(f"Found {len(keep)} person(s) at conf >= {args.conf:.2f}")
    for i, (_, score) in enumerate(keep, 1):
        print(f"  person {i}: confidence {score:.3f}")

    if keep:
        boxes = torch.stack([b for b, _ in keep])
        labels = [f"person {s:.2f}" for _, s in keep]
        annotated = draw_bounding_boxes(img, boxes, labels=labels, width=3)
        write_jpeg(annotated, args.out)
        print(f"Saved annotated image -> {args.out}")
    else:
        print("No people above threshold; try lowering --conf.")


if __name__ == "__main__":
    main()
