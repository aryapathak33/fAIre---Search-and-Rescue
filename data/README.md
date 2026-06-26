# Data folder

This folder contains data-preparation and dataset-quality scripts for fAIre.

Large datasets are not committed to git. Keep raw images locally in:

```text
data/raw/<class_name>/*.jpg
```

Default class layout:

```text
data/raw/
├── distress/
└── no_distress/
```

## Prepare data

Create train/validation/test splits:

```bash
python data/prepare_data.py --raw data/raw --out data --val-split 0.15 --test-split 0.15
```

Generated layout:

```text
data/train/<class_name>/
data/val/<class_name>/
data/test/<class_name>/
```

## Validate data

Run this before training:

```bash
python data/validate_dataset.py --data data
```

It checks split folders, class consistency, readable images, counts, and exact duplicate files.

## Sanity-check without private data

To verify that the code path works without real images:

```bash
python data/make_sanity_dataset.py --out data/raw_sanity --images-per-class 60
python data/prepare_data.py --raw data/raw_sanity --out data/sanity
python data/validate_dataset.py --data data/sanity
```

The synthetic sanity dataset is not a benchmark. It is only a quick pipeline test.
