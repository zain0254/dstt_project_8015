# Running the full pipeline on your Windows laptop (RTX 4050)

The laptop is the recommended way to run: the RTX 4050 trains the 369 small LSTM Autoencoders (41 files × 3 window sizes × 3 seeds) far faster than the 2-vCPU machines allowed on the AWS Free plan.

## 1. One-time setup (≈ 15 min)

1. Install **Python 3.11** (64-bit) from python.org. Tick **"Add python.exe to PATH"**.
2. Install **Git for Windows** (git-scm.com).
3. Make sure the **NVIDIA driver** is recent (GeForce Experience or nvidia.com). No separate CUDA toolkit is needed; PyTorch ships its own.
4. Open **PowerShell** and run:

```powershell
cd $HOME\Documents
git clone https://github.com/<you>/<code-repo>.git atd
cd atd
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scipy scikit-learn matplotlib
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True NVIDIA GeForce RTX 4050 Laptop GPU`.

> If PowerShell refuses to activate the venv, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## 2. Quick check (2 minutes)

```powershell
python tests\test_core.py
python scripts\download_nab.py
python scripts\select_datasets.py
python scripts\train_score.py --limit 2
```

You should see `CUDA available: True` and two finished jobs with their training time. Keep them: they are real LSTM-AE results and the full run continues from them.

## 3. Full run (one command)

```powershell
python run_pipeline.py
```

This runs every step in order: download → selection → EDA → training → evaluation → figures → dashboard. Keep the laptop **plugged in**, and set Windows power mode to *Best performance* so it doesn't sleep.

- If it stops for any reason, rerun: `python run_pipeline.py --from train`. Finished models are skipped.
- Logs are in `logs\`. Training progress is in `logs\train.log`.
- When it finishes, open `dashboard\index.html` in a browser.

## 4. Send the results back

```powershell
git add results figures dashboard logs
git commit -m "Full LSTM-AE run on RTX 4050"
git push
```
