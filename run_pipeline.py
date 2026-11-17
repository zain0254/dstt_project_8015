"""Run the whole project end to end with one command.

    python run_pipeline.py                  # everything (asks the runtime estimate first)
    python run_pipeline.py --from evaluate  # resume from a step
    python run_pipeline.py --only figures   # one step

Steps
  download        data acquisition: all 58 NAB v1.1 files (pinned commit, sha256 manifest)
  select          dataset selection rules (PROTOCOL.md §2)
  eda             exploratory data analysis + preprocessing figures
  estimate        times a few real fits on this machine and projects the total runtime;
                  stops if it exceeds the budget (default 15 h)
  train           split protocol: LSTM-AE fits, W in {30,50,100} x 3 seeds (GPU if available)
  evaluate        split protocol: tune on validation, score test, statistics; selects W*
  train_full      full-corpus protocol (primary): LSTM-AE fits for all 58 files at W*
  evaluate_full   full-corpus protocol: leave-one-file-out thresholds, metrics, statistics
  figures_full    figures for the full-corpus protocol (+ method figures)
  figures         figures for the split protocol
  dashboard_full  dashboard/index.html (full corpus)
  dashboard       dashboard/split.html
  report          docs/RESULTS.md and the README results block, read from results/
Every step writes its log to logs/<step>.log. Finished model fits are skipped on re-run.
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("download", ["scripts/download_nab.py"]),
    ("select", ["scripts/select_datasets.py"]),
    ("eda", ["scripts/make_eda.py"]),
    ("estimate", ["scripts/estimate_runtime.py"]),
    ("train", ["scripts/train_score.py", "--protocol", "split"]),
    ("evaluate", ["scripts/evaluate.py"]),
    ("train_full", ["scripts/train_score.py", "--protocol", "full"]),
    ("evaluate_full", ["scripts/evaluate_full.py"]),
    ("figures_full", ["scripts/make_figures.py", "--protocol", "full"]),
    ("figures", ["scripts/make_figures.py", "--protocol", "split"]),
    ("dashboard_full", ["scripts/make_dashboard.py", "--protocol", "full"]),
    ("dashboard", ["scripts/make_dashboard.py", "--protocol", "split"]),
    ("report", ["scripts/make_report.py"]),
]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS])
ap.add_argument("--only", choices=[s for s, _ in STEPS])
ap.add_argument("--train-args", default="", help='extra args for train, e.g. "--device cpu"')
ap.add_argument("--budget-hours", default="15", help="runtime budget checked by the estimate step")
args = ap.parse_args()

names = [s for s, _ in STEPS]
todo = [args.only] if args.only else names[names.index(args.start):] if args.start else names
(ROOT / "logs").mkdir(exist_ok=True)
t0 = time.time()
for name, cmd in STEPS:
    if name not in todo:
        continue
    extra = args.train_args.split() if name in ("train", "train_full") else []
    if name == "estimate":
        extra = ["--budget-hours", args.budget_hours] + args.train_args.split()
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
print(f"\nAll steps finished in {(time.time() - t0) / 60:.1f} min. Open dashboard/index.html (full corpus) and docs/RESULTS.md", flush=True)
