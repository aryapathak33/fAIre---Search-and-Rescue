# Upload guide

Use this when copying the updated project files into the existing GitHub repo on your laptop.

## 1. Open the local repo

In Git Bash, go into the repo folder:

```bash
cd ~/fAIre---Search-and-Rescue
explorer .
```

That opens the exact folder in Windows File Explorer.

## 2. Copy the update files

Extract the ZIP from ChatGPT. Copy everything inside the extracted folder into your local repo folder.

When Windows asks, choose:

```text
Replace the files in the destination
```

This update should keep your existing `media/demo_thumbnail.jpg` if it is already in the repo. The new README references that file for the clickable demo image.

## 3. Check formatting locally

Run:

```bash
git status
pytest -v
```

Optional sanity-data check:

```bash
python data/make_sanity_dataset.py --out data/raw_sanity --images-per-class 20
python data/prepare_data.py --raw data/raw_sanity --out data/sanity
python data/validate_dataset.py --data data/sanity --min-per-class 1
```

## 4. Commit and push

Use a clean commit message:

```bash
git add .
git commit -m "Improve dataset workflow and README polish"
git pull --rebase origin main
git push origin main
```

## 5. Check GitHub

Refresh the GitHub repo and confirm:

- the README has normal spacing and bullet formatting,
- the demo image is clickable,
- the YouTube link does not have an extra `)` at the end,
- the dataset workflow section shows sanity-data and validation commands,
- `training/train.py` and other Python files are formatted across many lines, not collapsed into one line.
