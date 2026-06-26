# fAIre: Search-and-Rescue Robot Vision System

> A low-cost AI robotics project for firefighter search-and-rescue: detect possible human distress or fire-scene hazards from camera/sensor input, prioritize recall, and relay alerts that help firefighters search faster.

![fAIre demo](media/demo.gif)

**Demo video:** https://youtu.be/ll3O9wNHNsc

---

## Why I built this

Firefighters entering a burning structure have to make fast decisions with limited visibility, incomplete building information, and constant risk. fAIre started as a self-taught robotics project and grew into an end-to-end system combining hardware sensing, computer vision, and a recall-first alerting pipeline.

The project is intentionally designed around one key search-and-rescue principle: **missing a person is worse than triggering an extra false alarm**. That is why the training and threshold-tuning code prioritizes recall over raw accuracy.

## What it does

fAIre is organized as a deployable perception pipeline for a rover or portable robot platform:

- **Camera-based detection** — classifies frames as possible distress/person-present vs. background/no-distress.
- **Sensor relay** — reads fire-scene sensor values from a microcontroller over serial when hardware is connected.
- **Risk scoring** — combines model confidence and sensor severity into a search-priority score.
- **Real-time output** — prints structured alerts that could be sent to a firefighter dashboard, radio bridge, or command laptop.
- **Evaluation pipeline** — measures precision, recall, F1, and confusion matrix on a held-out test set.

## System architecture

```text
Camera / robot sensors
        |
        v
Preprocessing + frame normalization
        |
        v
CNN image classifier / detector
        |
        v
Risk engine: confidence + sensor severity + threshold
        |
        v
Alert stream for firefighter / operator
```

The current public version is a **frame-level classifier pipeline**. It determines whether a frame likely contains a target condition. The next major upgrade is object detection with bounding boxes so the robot can localize the person/hazard inside the frame.

Full writeups:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/TRAINING.md`](docs/TRAINING.md)
- [`docs/HARDWARE.md`](docs/HARDWARE.md)

## Repository structure

```text
fAIre---Search-and-Rescue/
├── data/                 # dataset preparation script + sample format notes
├── demo/                 # evaluation and threshold-tuning scripts
├── docs/                 # architecture, hardware, and training writeups
├── hardware/             # Arduino sketch for sensor streaming
├── inference/            # model loading, image/video/webcam inference, risk engine
├── training/             # transfer-learning training pipeline
├── configs/              # optional config examples
├── tests/                # unit tests
├── models/               # trained weights are ignored; store links or small metadata only
└── media/                # screenshots, demo GIF, confusion matrix
```

## Quickstart

Clone the repo:

```bash
git clone https://github.com/aryapathak33/fAIre---Search-and-Rescue.git
cd fAIre---Search-and-Rescue
```

Create a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset format

The training code expects ImageFolder format:

```text
data/
├── train/
│   ├── distress/
│   └── no_distress/
├── val/
│   ├── distress/
│   └── no_distress/
└── test/
    ├── distress/
    └── no_distress/
```

To split raw class folders into train/val/test:

```bash
python data/prepare_data.py --raw data/raw --out data --val-split 0.15 --test-split 0.15
```

Expected raw format:

```text
data/raw/
├── distress/
└── no_distress/
```

## Train the model

```bash
python training/train.py --data data --epochs 10 --out models/fire_model.pt
```

The training script uses transfer learning with MobileNetV2. That gives the project a realistic path from a small labeled dataset to a usable prototype without pretending to train a huge vision model from scratch.

## Evaluate the model

```bash
python demo/evaluate.py --data data --weights models/fire_model.pt --threshold 0.50
```

This prints precision, recall, F1, and saves:

```text
media/confusion_matrix.png
media/metrics.json
```

## Run inference

Image:

```bash
python inference/detect.py --input data/samples/example.jpg --weights models/fire_model.pt --conf 0.50 --out media/prediction.jpg
```

Video:

```bash
python inference/detect.py --input media/demo_input.mp4 --weights models/fire_model.pt --conf 0.50 --out media/demo_output.mp4
```

Webcam:

```bash
python inference/detect.py --input 0 --weights models/fire_model.pt --conf 0.50
```

## Results

Real test-set metrics should be added after running `demo/evaluate.py` on the held-out test folder.

| Metric | Value | Notes |
|---|---:|---|
| Precision | Add measured value | From held-out test set |
| Recall | Add measured value | Primary metric for search-and-rescue |
| F1 | Add measured value | Balance of precision and recall |
| Test images | Add count | Never used during training |
| Hardware | Add device | Example: laptop CPU/GPU, Raspberry Pi, Jetson |

I am intentionally not publishing inflated or unmeasured numbers. For this type of safety-oriented project, an honest evaluation story matters more than a perfect-looking accuracy score.

## Engineering decisions

- **Transfer learning instead of training from scratch** — better for small datasets and faster iteration.
- **Recall-first thresholding** — in search-and-rescue, a false negative is worse than a false positive.
- **Modular design** — data prep, training, inference, risk scoring, and hardware streaming are separated so each part can be improved independently.
- **Hardware-aware pipeline** — the model is lightweight enough to eventually deploy on edge hardware instead of only running as a notebook demo.

## Roadmap

- Add real demo GIF and confusion matrix to `media/`.
- Add measured test-set metrics after retraining.
- Add bounding-box object detection for localization.
- Add thermal camera support.
- Build a lightweight dashboard for live alerts.
- Run inference on edge hardware such as Raspberry Pi or Jetson.

## About

Built by Arya Pathak. This project started as a robotics experiment and evolved into a full AI + hardware system focused on search-and-rescue constraints: low visibility, limited time, and the need to prioritize human detection.
