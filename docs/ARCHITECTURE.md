# fAIre architecture

fAIre is organized as a small robotics perception system. The code is split into independent layers so the robot side, AI side, flashover-warning side, and alert side can each improve without turning the project into one giant script.

## 1. Inputs

The system starts with two kinds of input:

- **Camera frames** from an image, video file, webcam, or robot-mounted camera.
- **Sensor readings** from a microcontroller, such as temperature, smoke/gas, CO-style analog values, VOC/gas values, oxygen percentage, heat flux, or distance.

The camera gives visual context. The sensors provide environmental context.

## 2. Data preparation

`data/prepare_data.py` converts raw class folders into a reproducible train/validation/test split.

```text
data/raw/class_name/*.jpg
        |
        v
data/train/class_name/*.jpg
data/val/class_name/*.jpg
data/test/class_name/*.jpg
```

The training and evaluation scripts use `torchvision.datasets.ImageFolder`, so every folder name becomes a class label.

## 3. Training layer

`training/train.py` trains a MobileNetV2 image classifier using PyTorch transfer learning.

Why MobileNetV2:

- small enough for robotics/edge experiments,
- strong enough for a first vision prototype,
- pretrained on general image features,
- easy to fine-tune on a custom dataset.

The script saves the best checkpoint by validation accuracy and writes lightweight metadata beside the model.

## 4. Evaluation layer

`demo/evaluate.py` runs the trained checkpoint on the held-out test split and reports:

- precision,
- recall,
- F1 score,
- per-class support,
- confusion matrix.

For this project, recall is especially important because a missed person/distress frame is more serious than an extra false alarm.

## 5. Inference layer

`inference/detect.py` loads the saved PyTorch checkpoint and runs inference on:

- a still image,
- a video file,
- or a webcam stream.

For every frame, the script computes class probabilities and emits a structured event. If an output path is provided, it also saves an annotated image or video.

## 6. Flashover warning layer

`inference/flashover_predictor.py` computes a project-defined decimal flashover warning index from `0.00` to `1.00`.

```text
flashover_index = 0.55 * thermal_score
                + 0.30 * gas_chemistry_score
                + 0.15 * temperature_trend_score
```

Inputs can include:

- temperature,
- heat flux,
- smoke,
- CO,
- VOC / combustible gas proxy,
- oxygen percentage,
- rate of temperature rise.

The index is not an official certification metric. It is a transparent prototype score that helps the robot identify areas where heat and chemistry signals may be moving toward rapid fire growth.

<p align="center">
  <img src="../media/flashover_index.png" alt="fAIre flashover index" width="820">
</p>

## 7. Risk layer

`inference/risk_engine.py` keeps alert logic separate from the model.

The CNN answers:

> What does the frame look like?

The flashover layer answers:

> Are environmental signals moving toward rapid fire growth?

The risk engine answers:

> How urgent is this frame when vision, sensor severity, and flashover warning are combined?

Current formula:

```text
risk_score = 0.60 * vision_confidence
           + 0.25 * sensor_severity
           + 0.15 * flashover_index
```

This is deliberately simple. It is easy to test, easy to tune, and easy to replace with a better model later.

## 8. Hardware layer

`hardware/fire_robot_controller.ino` streams sensor values over serial in this format:

```text
temperature_c,smoke_raw,co_raw,voc_raw,distance_cm
```

That lets the Python side combine camera confidence with physical sensor readings.

<p align="center">
  <img src="../media/system_overview.png" alt="fAIre system overview" width="820">
</p>

## Design choices

### Recall-first evaluation

A search-and-rescue robot should be cautious. If the model sees something that may be a person or distress signal, the system should prefer surfacing that area for review instead of silently ignoring it.

### Modular pipeline

The project is split into data, training, evaluation, inference, flashover scoring, risk scoring, and hardware folders. That makes it easier to replace individual pieces later.

### Lightweight model

MobileNetV2 is a practical first model because it supports real-time experimentation better than a huge network and keeps the path open for edge deployment.

### Flashover-aware safety logic

A camera-only robot can miss dangerous environmental context. The flashover module gives the project a place to study temperature, smoke, oxygen, CO, and gas behavior without pretending the prototype is a certified firefighting tool.

## Future architecture upgrades

- Object detection instead of frame classification.
- Live serial parsing inside the inference loop.
- Thermal-camera stream support.
- Calibrated gas sensor support for CO, CO2, O2, and VOCs.
- Robot navigation loop using obstacle distance.
- Dashboard for operator alerts.
