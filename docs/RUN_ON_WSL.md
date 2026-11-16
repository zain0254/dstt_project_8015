# Running the project on Windows with WSL (Ubuntu) and the RTX 4050

WSL2 lets Ubuntu use the laptop's NVIDIA GPU directly. You install no CUDA toolkit and no driver inside Ubuntu, only the normal Windows NVIDIA driver.

## 1. One-time Windows setup

1. Update the **NVIDIA driver** on Windows (GeForce Experience → Drivers, or nvidia.com).
2. If Ubuntu is not installed yet, open **PowerShell as Administrator** and run `wsl --install -d Ubuntu-24.04`, then restart and create your Ubuntu user.
3. Open **Ubuntu** and check that the GPU is visible:
   ```bash
   nvidia-smi
   ```
   It should list *NVIDIA GeForce RTX 4050 Laptop GPU*.

## 2. Unpack the project inside Ubuntu

Work in the Linux home folder, not under `/mnt/c`, because it is much faster. Assuming the zip is in your Windows Downloads folder:

```bash
cd ~
cp /mnt/c/Users/<your-windows-user>/Downloads/atd_project.zip .
unzip atd_project.zip
cd lstm-ae-adaptive-threshold
bash setup_wsl.sh
```

The last lines of the setup must show `CUDA available: True | NVIDIA GeForce RTX 4050 Laptop GPU` and five `ok test_...` lines.

## 3. Quick check (about 2 minutes)

```bash
source .venv/bin/activate
python scripts/train_score.py --limit 2
```

You should see the GPU name and two finished jobs with their training time. Keep these files: the full run continues from them.

## 4. Full run (one command)

```bash
python run_pipeline.py
```

This runs download → dataset selection → EDA → training (369 LSTM-AE fits) → evaluation → figures → dashboard. Keep the laptop plugged in and stop it from sleeping (Windows Settings → Power → Screen and sleep → *Never* while plugged in).

- If it stops, resume with `python run_pipeline.py --from train`. Finished models are skipped.
- Progress: `tail -f logs/train.log` in a second Ubuntu window.
- Results: open the dashboard from Ubuntu with `explorer.exe dashboard/index.html`. Figures are in `figures/`, all numbers in `results/`.

## 5. Send the results back

Zip the outputs and send them, or push them to GitHub:

```bash
zip -r results_run.zip results figures dashboard logs
explorer.exe .
```
