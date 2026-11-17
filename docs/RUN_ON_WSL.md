# Running the experiment (Windows + WSL Ubuntu + RTX 4050)

Use the **Ubuntu terminal** for the run. VS Code is fine for browsing files and figures afterwards, but a plain terminal keeps the long run independent of the editor.

## 0. Requirements

- Windows 10/11 with WSL2 and Ubuntu 22.04 or 24.04 (`wsl --install -d Ubuntu-24.04` in an admin PowerShell, if not installed).
- A current **Windows** NVIDIA driver. Do not install a driver or CUDA inside Ubuntu.
- About 3 GB of free disk space and an internet connection (PyTorch download, NAB check).

Check the GPU from Ubuntu: `nvidia-smi` must list the *RTX 4050 Laptop GPU*.

## 1. Unpack and set up (once, about 5–10 minutes)

Work in the Linux home folder, which is much faster than `/mnt/c`:

```bash
cd ~
cp /mnt/c/Users/<your-windows-user>/Downloads/atd_project.zip .
unzip atd_project.zip
cd lstm-ae-adaptive-threshold
bash setup_wsl.sh
```

The setup must end with `CUDA available: True | NVIDIA GeForce RTX 4050 Laptop GPU` and nine `ok test_...` lines.

## 2. Run everything

```bash
source .venv/bin/activate
python run_pipeline.py
```

What happens, in order (logs in `logs/<step>.log`):

| Step | What it does | Rough time on an RTX 4050 |
|---|---|---|
| download, select, eda | verifies/downloads the 58 NAB files, applies the selection rules, draws the EDA figures | 2–5 min |
| **estimate** | trains 6 real models, projects the total runtime, **applies the 15-hour rule** (prints the decision) | 2–5 min |
| train | split protocol: 369 LSTM-AE fits | about 1–2.5 h |
| evaluate | validation tuning, test metrics, statistics, selects W\* | 10–20 min |
| train_full | full corpus: 174 fits at W\* | about 0.5–1 h |
| evaluate_full | leave-one-file-out selection, metrics, statistics | 10–30 min |
| figures, dashboards, report | all result figures, `dashboard/index.html`, `docs/RESULTS.md` | 5 min |

The times are my estimates. The **estimate** step prints the real projection for your laptop. Keep the laptop plugged in and stop it from sleeping (Windows Settings → Power → *Never* while plugged in).

**If it stops** (sleep, closed terminal): `python run_pipeline.py --from train`. Finished models are skipped automatically. Use `--from train_full` or `--from evaluate_full` to resume later steps.

**Watch progress** from a second Ubuntu window: `tail -f ~/lstm-ae-adaptive-threshold/logs/train.log`.

**No GPU / CUDA problem:** `python run_pipeline.py --train-args "--device cpu"` (slower; the estimate step will say whether it fits in 15 h).

## 3. Look at the results

```bash
explorer.exe dashboard/index.html      # full-corpus dashboard (dashboard/split.html for the proposal split)
explorer.exe figures                   # all figures (PNG; vector PDFs in figures/full/pdf and figures/split/pdf)
```

`docs/RESULTS.md` lists every number, hypothesis decision and figure. The README results block is filled in automatically.

## 4. Send the results back

```bash
zip -r results_run.zip results figures dashboard docs/RESULTS.md README.md logs -x "results/*/scores/*"   # small: no score files
explorer.exe .
```

Upload `results_run.zip` in the chat. Then push the project to GitHub. The zip already contains the git history, which shows that `PROTOCOL.md` was committed before any result existed. Create an **empty** repository on GitHub, then:

```bash
git add -A && git commit -m "Final experiment run: results, figures, dashboards"
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

The saved reconstruction errors (`results/*/scores/`, roughly 50–100 MB) are pushed too, so figures and statistics can be regenerated without retraining.

## Optional: code-free submission copy

```bash
python scripts/export_submission.py ../submission
```

This copies figures, dashboards, result tables, the protocol and the docs, without code, for an instructor who wants a figures-only repository.
