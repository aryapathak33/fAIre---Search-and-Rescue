"""
Run fAIre inference on an image, video file, or webcam.

Examples:
    python inference/detect.py --input data/samples/example.jpg --weights models/fire_model.pt --out media/prediction.jpg
    python inference/detect.py --input media/demo_input.mp4 --weights models/fire_model.pt --out media/demo_output.mp4
    python inference/detect.py --input 0 --weights models/fire_model.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

try:
    from inference.risk_engine import SensorReading, build_alert
except ModuleNotFoundError:
    # Allows running as: python inference/detect.py
    from risk_engine import SensorReading, build_alert


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fAIre detection")
    parser.add_argument("--input", required=True, help="Image path, video path, or webcam index like 0")
    parser.add_argument("--weights", required=True, help="Trained checkpoint path")
    parser.add_argument("--conf", type=float, default=0.50, help="Alert threshold")
    parser.add_argument("--out", default=None, help="Optional output image/video path")
    parser.add_argument("--display", action="store_true", help="Show frames in a window")
    parser.add_argument("--positive-class", default=None, help="Class treated as the alert class")
    return parser.parse_args()


def load_model(weights_path: str, device: str):
    checkpoint = torch.load(weights_path, map_location=device)
    classes = checkpoint["classes"]
    img_size = checkpoint.get("img_size", 224)
    mean = checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("imagenet_std", [0.229, 0.224, 0.225])

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return model, classes, transform


def choose_positive_class(classes: list[str], requested: str | None) -> int:
    if requested is not None:
        if requested not in classes:
            raise ValueError(f"positive class '{requested}' not found in checkpoint classes: {classes}")
        return classes.index(requested)

    for keyword in ["distress", "person", "human", "hazard", "fire"]:
        for idx, name in enumerate(classes):
            lowered = name.lower()
            if keyword in lowered and "no" not in lowered:
                return idx
    return 1 if len(classes) > 1 else 0


def predict_frame(
    model: torch.nn.Module,
    frame_bgr,
    transform: transforms.Compose,
    classes: list[str],
    positive_idx: int,
    device: str,
) -> Tuple[str, float, dict]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs).item())
    label = classes[pred_idx]
    confidence = float(probs[positive_idx].item())
    all_probs = {classes[i]: round(float(probs[i].item()), 4) for i in range(len(classes))}
    return label, confidence, all_probs


def annotate_frame(frame, alert, threshold: float):
    color = (0, 0, 255) if alert.confidence >= threshold else (0, 180, 255)
    text = f"{alert.label}: {alert.confidence:.2f} | risk={alert.risk_score:.2f} | {alert.priority}"
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 760), 52), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return frame


def emit_alert(label: str, confidence: float, threshold: float, probs: dict) -> None:
    # Sensor values are None here. When the robot hardware is connected, populate
    # SensorReading from serial data before building the alert.
    alert = build_alert(label=label, confidence=confidence, sensors=SensorReading())
    event = {
        "timestamp": round(time.time(), 3),
        "alert": confidence >= threshold,
        "threshold": threshold,
        "probabilities": probs,
        **alert.to_dict(),
    }
    print(json.dumps(event))


def run_image(args, model, classes, transform, positive_idx, device: str) -> None:
    frame = cv2.imread(args.input)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    label, confidence, probs = predict_frame(model, frame, transform, classes, positive_idx, device)
    emit_alert(label, confidence, args.conf, probs)
    alert = build_alert(label, confidence)
    annotated = annotate_frame(frame, alert, args.conf)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, annotated)
        print(f"Saved annotated image -> {args.out}")

    if args.display:
        cv2.imshow("fAIre detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_or_stream(args, model, classes, transform, positive_idx, device: str) -> None:
    source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input: {args.input}")

    writer = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        label, confidence, probs = predict_frame(model, frame, transform, classes, positive_idx, device)
        alert = build_alert(label, confidence)
        annotated = annotate_frame(frame, alert, args.conf)

        if frame_idx % 10 == 0:
            emit_alert(label, confidence, args.conf, probs)

        if writer:
            writer.write(annotated)
        if args.display:
            cv2.imshow("fAIre detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video -> {args.out}")
    if args.display:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes, transform = load_model(args.weights, device)
    positive_idx = choose_positive_class(classes, args.positive_class)

    suffix = Path(args.input).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        run_image(args, model, classes, transform, positive_idx, device)
    elif suffix in VIDEO_EXTENSIONS or args.input.isdigit():
        run_video_or_stream(args, model, classes, transform, positive_idx, device)
    else:
        raise ValueError("Input must be an image, video, or webcam index like 0")


if __name__ == "__main__":
    main()
