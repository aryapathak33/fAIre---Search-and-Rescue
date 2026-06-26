# Architecture

fAIre is built as an end-to-end robotics perception pipeline rather than a standalone notebook. The goal is to show how camera frames and sensor readings move through the system until they become a useful firefighter alert.

## End-to-end flow

```text
[robot camera + sensors]
          |
          v
[data preprocessing]
  - resize image
  - normalize pixels
  - optional augmentation during training
          |
          v
[vision model]
  - MobileNetV2 transfer-learning classifier
  - outputs class probabilities
          |
          v
[risk engine]
  - combines model confidence + sensor severity
  - prioritizes recall over raw accuracy
          |
          v
[real-time alert]
  - JSON-style alert stream
  - annotated image/video output
  - future dashboard/radio bridge
```

## Components

### 1. Sensing layer

The hardware side can include a camera, ultrasonic distance sensing, and fire-scene environmental sensors such as smoke, temperature, or gas sensors. The Arduino sketch in `hardware/` streams sensor values over serial so the perception code can eventually combine visual confidence with physical risk indicators.

### 2. Data layer

`data/prepare_data.py` converts raw class folders into a reproducible train/validation/test split. This matters because a recruiter or reviewer can see that the model is not being evaluated on images it already saw during training.

### 3. Model layer

The public code uses MobileNetV2 transfer learning. That is a practical design choice for a robotics prototype because MobileNet is lightweight, pretrained, and much easier to deploy on edge hardware than a giant model.

### 4. Inference layer

`inference/detect.py` loads the trained checkpoint and runs detection on:

- one image,
- a video file,
- or a webcam/live stream.

The current version is frame-level classification. It answers: **does this frame likely contain the target condition?** The next upgrade is object detection so the system can also answer: **where in the frame is the person or hazard?**

### 5. Risk / alert layer

`inference/risk_engine.py` is intentionally separate from the CNN. The CNN predicts visual confidence. The risk engine turns that confidence into an operational priority using a simple rule:

> High visual confidence + severe sensor readings = higher search priority.

That separation makes the project look more like a real system and less like a single ML script.

## Design decisions

### Recall over precision

In normal classification, people often chase accuracy. In search-and-rescue, accuracy can be misleading. If a person is present and the robot misses them, that is much worse than sending a firefighter to double-check a false alarm.

So the model should be tuned with this priority:

1. Maximize recall: catch as many true distress/person frames as possible.
2. Keep precision acceptable: avoid too many false alarms.
3. Report F1 as the balance point.

### Transfer learning over training from scratch

The dataset for a student robotics project will not be millions of images. Transfer learning is the right engineering move because the CNN already understands generic visual features like edges, textures, and shapes. The project only needs to fine-tune the final layers for the fire/search-and-rescue task.

### Lightweight model over huge model

A robot has power, latency, and compute constraints. MobileNetV2 is a reasonable prototype choice because it is small enough to move toward Raspberry Pi / Jetson-style deployment while still being strong enough for image classification.

## Recruiter takeaway

This repo demonstrates more than model training. It shows system thinking:

- dataset preparation,
- transfer learning,
- evaluation with real metrics,
- threshold tuning for a safety-oriented objective,
- image/video inference,
- and hardware-aware alerting logic.
