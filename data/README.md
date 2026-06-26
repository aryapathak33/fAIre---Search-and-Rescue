# Data folder

This folder contains data-preparation code and the expected dataset layout for fAIre.

Large datasets are not committed to git. Keep raw images locally in:

```text
data/raw/<class_name>/*.jpg
```

Then create train/validation/test splits:

```bash
python data/prepare_data.py --raw data/raw --out data --val-split 0.15 --test-split 0.15
```

The generated training layout is:

```text
data/train/<class_name>/
data/val/<class_name>/
data/test/<class_name>/
```
