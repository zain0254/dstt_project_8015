"""Run the whole project end to end with one command.

    python run_pipeline.py                 # everything
    python run_pipeline.py --from evaluate # resume from a step
    python run_pipeline.py --only train    # one step

Steps
  download   1. data acquisition: NAB v1.1 from GitHub (pinned commit, checksums)
  select     2. dataset selection rule (PROTOCOL.md §2)
  eda        3. exploratory data analysis figures
  train      4. LSTM-Autoencoder training + reconstruction errors (GPU if available)
  evaluate   5. threshold tuning on validation, test metrics, statistics, hypotheses
  figures    6. result figures
  dashboard  7. interactive dashboard
Every step writes its log to logs/<step>.log.
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("download", ["scripts/download_nab.py"]),
    ("select", ["scripts/select_datasets.py"]),
    ("eda", ["scripts/make_eda.py"]),
    ("train", ["scripts/train_score.py"]),
    ("evaluate", ["scripts/evaluate.py"]),
    ("figures", ["scripts/make_figures.py"]),
    ("dashboard", ["scripts/make_dashboard.py"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS])
ap.add_argument("--only", choices=[s for s, _ in STEPS])
ap.add_argument("--train-args", default="", help='extra args for train, e.g. "--device cpu"')
args = ap.parse_args()

names = [s for s, _ in STEPS]
todo = [args.only] if args.only else names[names.index(args.start):] if args.start else names
(ROOT / "logs").mkdir(exist_ok=True)
t0 = time.time()
for name, cmd in STEPS:
    if name not in todo:
        continue
    extra = args.train_args.split() if name == "train" else []
    print(f"\n=== {name} ===", flush=True)
    ts = time.time()
    with open(ROOT / "logs" / f"{name}.log", "w") as log:
        p = subprocess.Popen([sys.executable, "-u", *cmd, *extra], cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
        p.wait()
    if p.returncode != 0:
        sys.exit(f"step '{name}' failed (see logs/{name}.log)")
    print(f"--- {name} done in {(time.time() - ts) / 60:.1f} min", flush=True)
print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min. Open dashboard/index.html", flush=True)
