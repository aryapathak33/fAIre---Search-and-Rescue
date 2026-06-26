"""
Export a trained fAIre PyTorch checkpoint to ONNX for edge-deployment experiments.

This is useful when testing the project on accelerator-oriented runtimes later.
It does not replace normal evaluation; export only after the PyTorch checkpoint
has already been trained and tested.

Example:
    python training/export_onnx.py --weights models/fire_model.pt --out models/fire_model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fAIre checkpoint to ONNX")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint produced by training/train.py")
    parser.add_argument("--out", default="models/fire_model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def load_model(weights_path: str, device: str):
    checkpoint = torch.load(weights_path, map_location=device)
    classes = checkpoint["classes"]
    img_size = checkpoint.get("img_size", 224)
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, classes, img_size


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, img_size = load_model(args.weights, device)

    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )

    print(f"Exported ONNX model -> {out_path}")
    print(f"classes: {classes}")


if __name__ == "__main__":
    main()
