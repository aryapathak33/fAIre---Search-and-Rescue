# Training process

From first model to retrained detector.

## 1. Starting point — Google Teachable Machine
`[FILL IN: what you trained it to recognize, what classes, roughly how many
images, and what you learned doing it the first time.]`

## 2. Retraining in `[FILL IN: TensorFlow / PyTorch]`
`[FILL IN: why you moved off Teachable Machine, the architecture you used,
how you collected/labeled data, augmentations, train/val split.]`

## 3. Evaluation
`[FILL IN: how you measured accuracy/precision/recall, on what test set,
and the numbers you got. Honest + specific beats impressive + vague.]`

## Reproducing
```bash
python ../training/train.py --data ../data --epochs 30
```
