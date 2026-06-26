# Dataset notes

fAIre keeps the public repository lightweight: the code, dataset format, and validation tools are committed, but the full image dataset is not.

## Class layout

The current classifier expects two classes by default:

```text
distress/
no_distress/
```

Those names are intentionally broad. Depending on the experiment, `distress` can mean a visible fire-scene target, a person-like silhouette in a hazardous area, or another positive class. The important rule is that class meaning must be consistent across `train`, `val`, and `test`.

## Raw data format

Put raw images here locally:

```text
data/raw/
├── distress/
└── no_distress/
```

Then split them into train/validation/test:

```bash
python data/prepare_data.py --raw data/raw --out data --val-split 0.15 --test-split 0.15
```

## Validate before training

Before training, run:

```bash
python data/validate_dataset.py --data data
```

The validator checks:

- `train`, `val`, and `test` folders exist,
- the same class names appear in each split,
- images can be opened by Pillow,
- class counts are shown clearly,
- exact duplicate files are flagged.

## Sanity-check dataset

For a quick software test, generate a tiny synthetic dataset:

```bash
python data/make_sanity_dataset.py --out data/raw_sanity --images-per-class 60
python data/prepare_data.py --raw data/raw_sanity --out data/sanity
python data/validate_dataset.py --data data/sanity
```

This dataset is only for verifying that the training and evaluation code runs end-to-end. It should not be reported as real fire-scene model performance.

## Real dataset plan

For a real training run, the dataset should include:

- different lighting conditions,
- smoke and non-smoke scenes,
- fire-colored distractors that are not hazards,
- non-fire backgrounds from similar camera angles,
- images from the robot's actual camera when possible,
- a held-out test split that is not touched during training or threshold tuning.

## Metrics policy

Do not publish precision, recall, or F1 until they come from a held-out test split and the exact evaluation command is documented. The repo should stay honest: runnable code first, real metrics only when the data and model checkpoint support them.
