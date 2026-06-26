# How to push these upgrades to GitHub

## Option A: easiest GitHub website method

1. Open your repo: `aryapathak33/fAIre---Search-and-Rescue`.
2. Click the file you want to replace, for example `README.md`.
3. Click the pencil icon.
4. Paste the replacement content.
5. Commit with a clear message like `Update README with system overview`.
6. Repeat for the core files.

Recommended order:

1. `requirements.txt`
2. `README.md`
3. `docs/ARCHITECTURE.md`
4. `docs/TRAINING.md`
5. `docs/HARDWARE.md`
6. `training/train.py`
7. `data/prepare_data.py`
8. `inference/risk_engine.py`
9. `inference/detect.py`
10. `demo/evaluate.py`
11. `hardware/fire_robot_controller.ino`
12. `tests/test_risk_engine.py`

## Option B: VS Code / terminal method

Clone your repo:

```bash
git clone https://github.com/aryapathak33/fAIre---Search-and-Rescue.git
cd fAIre---Search-and-Rescue
```

Copy these replacement files into the matching folders.

Then run:

```bash
git status
git add .
git commit -m "Build out fAIre training inference and hardware pipeline"
git push origin main
```

## Test locally before pushing

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
```

Training will only work once you add real data in the expected folder format.
