"""
detect.py - Run FIRE detection on an image, a video file, or a live stream.

SKELETON: drop your real inference code into the TODO blocks. The shell handles
argument parsing and the input-type branching so the file is readable on GitHub;
the detection/postprocessing logic should be the code you actually wrote.

Usage:
    python inference/detect.py --input data/samples/example.jpg --weights models/fire_model.pt
    python inference/detect.py --input path/to/video.mp4 --weights models/fire_model.pt
    python inference/detect.py --input 0 --weights models/fire_model.pt   # webcam / stream
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Run FIRE detection")
    p.add_argument("--input", required=True, help="Image path, video path, or stream index")
    p.add_argument("--weights", required=True, help="Path to trained model weights")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    return p.parse_args()


def load_model(weights):
    """TODO: load your retrained model from `weights`."""
    raise NotImplementedError("Add your real model-loading code.")


def predict(model, frame, conf):
    """TODO: run a single frame through the model and return detections
    (e.g. distress signals, hazards, flashover-risk flag)."""
    raise NotImplementedError("Add your real inference + postprocessing code.")


def main():
    args = parse_args()
    model = load_model(args.weights)
    # TODO: branch on input type (image / video / stream), call predict(),
    #       and emit the real-time alert/relay output your system produces.
    print("Wire up your real inference loop here.")


if __name__ == "__main__":
    main()
