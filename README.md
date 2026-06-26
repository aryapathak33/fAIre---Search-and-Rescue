# FIRE: Search & Rescue AI for Firefighters

> Autonomous perception system that detects human distress signals and critical fire hazards inside burning buildings and relays them to firefighters in real time.

<!-- TODO: replace with your demo GIF. A 3–6s autoplaying loop here is the single most important thing on this page. -->
<!-- Drag a .gif or .mp4 into this README in GitHub's editor and it will embed automatically. -->
![FIRE demo](media/demo.gif)

▶️ **Full demo video:** (https://youtu.be/ll3O9wNHNsc)

---

## What it does

FIRE is a computer-vision system for search-and-rescue robotics in fire environments. It is designed to be carried by (or mounted on) a rover that enters a structure ahead of firefighters and continuously reports:

- **Human distress detection** — locates people / distress signals in low-visibility, smoke-filled conditions.
- **Hazard recognition** — flags critical points of explosion and other fire hazards.
- **Flashover risk alerting** — `[FILL IN: 1 sentence on how you flag imminent flashover conditions]`.
- **Real-time relay** — streams location + hazard data back to firefighters as it moves.

`[FILL IN: 2–3 sentences in your own words on the problem and why you built it — the fire-station conversations are a great detail to include here.]`

## Origin

This project started in 2017 as a self-taught hardware experiment and was rebuilt into its current form in 2023.

- **2017 — first prototype.** Started with an Arduino and `[FILL IN: which sensors]`, experimenting with 3D imaging from sensor data.
- **First models.** Learned to train image-classification models for the first time using Google Teachable Machine.
- **Field research.** Visited fire stations in `[FILL IN: neighborhood / cities]` to interview firefighters first-hand and understand what data would actually be useful in the field.
- **2023 — rebuild.** Retrained the perception models from scratch in `[FILL IN: TensorFlow or PyTorch]` and restructured the system into the pipeline in this repo.

## How it works

```
[ sensors / camera ]  ->  [ preprocessing ]  ->  [ detection model ]  ->  [ alerting + relay ]
   hardware build          data/                  inference/               (real-time output)
```

`[FILL IN: a couple of sentences describing the flow in your own words. Mention the model architecture you retrained, e.g. a CNN-based image classifier / detector.]`

A fuller breakdown lives in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), the hardware build in [`docs/HARDWARE.md`](docs/HARDWARE.md), and the training process in [`docs/TRAINING.md`](docs/TRAINING.md).

## Results

> ⚠️ **These are ILLUSTRATIVE sample numbers showing the realistic shape of a strong
> college-level result — NOT measured values. Replace every one with your own output
> from `demo/evaluate.py` before publishing.** Shipping these as-is would be claiming
> results you didn't measure.

Distress-signal detection (held-out test set of `[FILL IN: # images]`, `[FILL IN: hardware]`):

| Metric | Sample value | What it means |
|---|---|---|
| Precision | 0.89 | Of frames flagged as a person, 89% were |
| Recall | 0.94 | Of people actually present, 94% were caught |
| F1 | 0.91 | Balance of the two |
| Accuracy | 0.93 | Overall correct (test set ~`[FILL IN]`% positive) |
| Inference speed | 22 FPS | on `[FILL IN: e.g. laptop GPU / Jetson Orin]` |
| Training set | ~`[FILL IN: e.g. 1,200]` images | self-collected + `[FILL IN: source]` |

Note that **recall (0.94) is tuned higher than precision (0.89)** on purpose: in
search-and-rescue a missed person is far costlier than a false alarm, so the
confidence threshold is set to favor catching everyone. (`demo/demo_metrics.py`
demonstrates this trade-off live.)

> Why these look real and "99.9%" wouldn't: precision and recall differ, nothing is
> a round perfect number, and every figure names its test set and hardware.

## Repository structure

```
fire-search-rescue/
├── demo/            # runnable examples: metrics, person detection, training, tuning
├── tests/           # unit tests (run: pytest)
├── configs/         # training/tuning config templates
├── training/        # model training scripts (your real robot code)
├── inference/       # run detection on images / video / stream
├── data/            # data prep + (small) sample data; large data is gitignored
├── models/          # trained model weights (or a link if too large for GitHub)
├── hardware/        # Arduino sketches, wiring notes, sensor setup
├── docs/            # architecture, hardware, training write-ups
└── media/           # demo gif / screenshots
```

## Running the tests

```bash
pip install pytest
pytest -v
```

The metrics utilities are covered by real unit tests (and cross-checked against
scikit-learn); the classifier smoke test runs once you have torch installed.

## Quickstart

```bash
git clone https://github.com/USERNAME/fire-search-rescue.git
cd fire-search-rescue
pip install -r requirements.txt
```

Run detection on a sample image:

```bash
python inference/detect.py --input data/samples/example.jpg --weights models/fire_model.pt
```

Train (or retrain) the model:

```bash
python training/train.py --data data/ --epochs 30
```

`[FILL IN: adjust the commands above to match your actual scripts once you paste them in.]`

## Roadmap

- [ ] `[FILL IN: next thing you'd build — e.g. deploy on Jetson, add thermal camera, live-stream dashboard]`
- [ ] `[FILL IN]`

## About

Built by Arya Pathak. `[FILL IN: one line — e.g. "Started in middle school, still iterating."]`
Field research conducted with firefighters in `[FILL IN: location]`.

<!-- If any part of the system is private/sensitive, it's fine to say so:
     "Core X module shown here; full integration code kept private." Honesty reads as strength. -->
