# Updating the GitHub repo

This folder contains replacement files for the fAIre repo. Copy the contents into the cloned repository, replace matching files, then commit and push.

## Recommended terminal flow

From Git Bash:

```bash
cd ~/fAIre---Search-and-Rescue
explorer .
```

Copy the files from this upgrade folder into the Explorer window that opens. Choose **Replace** when Windows asks.

Then run:

```bash
git status
git add .
git commit -m "Polish fAIre robot project"
git push origin main
```

## Better commit message

Use this message:

```bash
git commit -m "Polish fAIre robot project"
```

That will look cleaner on GitHub than a long implementation-style message.

## Test locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -v
```

Training requires a real local dataset in the expected `data/raw/<class_name>/` format.
