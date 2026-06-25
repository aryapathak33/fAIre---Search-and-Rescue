# How to put this on GitHub (step by step)

This file is just for you — you can delete it before/after publishing. It walks
through getting this folder onto GitHub and embedding your demo video.

## Before you start
- Make a GitHub account if you don't have one (github.com).
- Have your real `.py` files and trained model ready to copy in.
- Have a short demo video (aim for 30–90s).

## Option A — Upload through the website (easiest, no command line)
1. On github.com, click **+ → New repository**.
2. Name it `fire-search-rescue`, set it **Public**, do NOT add a README
   (this folder already has one). Click **Create repository**.
3. On the new repo page, click **uploading an existing file**.
4. Drag in everything from this folder. Commit.
5. Open each file with `[FILL IN]` placeholders (start with `README.md`) using the
   pencil ✏️ icon, fill in your real details, and commit.

## Option B — Command line (git)
```bash
cd fire-search-rescue
git init
git add .
git commit -m "Initial commit: FIRE search & rescue AI"
git branch -M main
git remote add origin https://github.com/USERNAME/fire-search-rescue.git
git push -u origin main
```

## Adding your real code
- Training code  -> `training/train.py`
- Inference code -> `inference/detect.py`
- Data prep      -> `data/prepare_data.py`
- Arduino sketches -> `hardware/`
- Trained weights -> `models/`  (if the file is >100 MB, don't commit it — upload it
  to a GitHub **Release** or Google Drive and link it in the README instead)

After pasting your code in, come back and I'll help you clean it up, add comments,
and write docstrings so it reads well.

## Embedding the demo video
GitHub plays videos **uploaded directly to GitHub**, not YouTube links, inline.
Two good approaches:

1. **Autoplay GIF at the top (best first impression).** Convert a few seconds of your
   demo to a GIF, name it `demo.gif`, put it in `media/`. The README already points
   to `media/demo.gif`, so it'll show up automatically and autoplay silently.

2. **Full video.** Either:
   - Drag the `.mp4` straight into the README while editing on github.com (works if
     under ~10 MB free / 100 MB Pro) — GitHub hosts it and renders a player, OR
   - Upload to YouTube/Loom and paste the link next to the ▶️ in the README as a
     clickable thumbnail.

## Make it look finished
- Click the ⚙️ next to **About** (top right of the repo) → add a one-line description
  and topics like `computer-vision`, `robotics`, `tensorflow`, `search-and-rescue`.
- **Pin** the repo on your GitHub profile (Customize your pins) so it shows first.
- Make sure the link on your resume points here: `github.com/USERNAME/fire-search-rescue`.
