# demo/ — runnable examples

Three small, real programs that show the techniques behind FIRE. Start here.

## 1. `demo_metrics.py` — the evaluation pipeline (runs in seconds, no setup)
Trains a classifier on a tiny bundled dataset set up as **rare-event detection**
(mirrors "distress signal vs not"), and prints precision/recall/F1 + the
precision–recall trade-off + a confusion matrix. This is the exact reporting style
you'll use on your real model.
```bash
pip install scikit-learn matplotlib
python demo_metrics.py
```
Sample output (the EVENT row is the rare class):
```
   not-event   precision 0.988  recall 0.992  f1 0.990
       EVENT   precision 0.925  recall 0.891  f1 0.907
```
See `sample_confusion_matrix.png` for what the saved image looks like.

## 2. `detect_people.py` — find people in an image (FIRE-relevant)
A real detector (pretrained on COCO) that boxes people and prints confidences.
First run downloads weights automatically. CPU works (slower).
```bash
pip install torch torchvision pillow
python detect_people.py --input your_photo.jpg --out detected.jpg --conf 0.7
```

## 3. `train_classifier.py` + `evaluate.py` — train on YOUR images, get YOUR numbers
Transfer-learning: fine-tune a pretrained network on your labeled image folders,
then measure it honestly. `evaluate.py` produces the precision/recall/F1 you put in
the README — measured on your test set, not made up.
```bash
pip install torch torchvision scikit-learn matplotlib
python train_classifier.py --data data --epochs 10 --out ../models/fire_model.pt
python evaluate.py --data data --weights ../models/fire_model.pt
```
No GPU? Run these two in Google Colab (free GPU): upload your `data/` folder, paste
the scripts into a notebook, done.

---
**The honest workflow:** run `demo_metrics.py` to understand the report → collect/label
your images → `train_classifier.py` → `evaluate.py` → copy *those* real numbers into
the main README. Never ship the illustrative numbers; ship your measured ones.
