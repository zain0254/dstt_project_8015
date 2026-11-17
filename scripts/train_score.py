"""Step 1 - train the LSTM Autoencoder and save reconstruction errors.

One job = (NAB file, window size W, seed). Jobs are split into balanced
shards so several small machines can run in parallel:

    python scripts/train_score.py --shard 0 --of 4

Finished jobs are skipped, so the command can be re-run after interruption.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from atd import ROOT, load_config, load_datasets
from atd.data import load_labels, load_series, scaled_series, make_windows, training_windows
from atd.pipeline import all_jobs, shard_jobs, score_path, SCORES_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--model", choices=["lstm", "standin"], default="lstm",
                help="'standin' = PCA, for pipeline smoke tests only")
ap.add_argument("--protocol", choices=["split", "full"], default="split",
                help="split = 60/20/20; full = all 58 files, train on first 15%% (needs W* from evaluate.py)")
ap.add_argument("--out", default=None)
ap.add_argument("--window", type=int, default=0, help="full protocol: override W*")
ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs (smoke test)")
ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
args = ap.parse_args()

cfg = load_config()
datasets = load_datasets()
labels = load_labels(cfg["labels_file"])
from atd import protocol
PR = protocol(cfg, args.protocol)
windows = None
if args.protocol == "full":
    if args.window:
        windows = [args.window]
    else:
        import pandas as pd
        ws = pd.read_csv(ROOT / "results/split/window_selection.csv")
        windows = [int(ws[ws.selected].W.iloc[0])]
    print(f"full-corpus protocol: {len(datasets[PR['jobs_key']])} files, W* = {windows[0]}", flush=True)
jobs = shard_jobs(all_jobs(cfg, datasets, PR["jobs_key"], windows), args.shard, args.of)
if args.limit:
    jobs = jobs[: args.limit]
out = Path(args.out or PR["scores"]); out.mkdir(parents=True, exist_ok=True)
m = PR["model"]
if args.model == "lstm":
    from atd.model import train_and_score as fit
else:
    from atd.model import standin_score as fit
if args.model == "lstm":
    import torch
    print("PyTorch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
print(f"shard {args.shard}/{args.of}: {len(jobs)} jobs", flush=True)
t_all = time.time()
n_done = 0
for i, j in enumerate(jobs, 1):
    p = score_path(j["file"], j["W"], j["seed"], out)
    if p.exists():
        old = json.loads(str(np.load(p)["meta"]))
        if old.get("model") == args.model:
            continue          # finished earlier -> resume
        p.unlink()            # never mix stand-in and LSTM scores
    s = load_series(j["file"], cfg["nab_dir"], labels, PR["split"])
    x = scaled_series(s)
    W = j["W"]
    trw = training_windows(x, W, s.a, m["max_train_windows"],
                          s.labels if cfg.get("training_data") == "normal_only" else None)
    allw = make_windows(x, W)
    err_w, meta = fit(trw, allw, seed=j["seed"], hidden=m["hidden_size"], latent=m["latent_size"],
                      lr=m["learning_rate"], batch_size=m["batch_size"], max_epochs=m["max_epochs"],
                      patience=m["early_stopping_patience"], holdout=m["early_stopping_holdout"],
                      threads=m["torch_threads"], **({"device": args.device} if args.model == "lstm" else {}))
    errors = np.full(s.n, np.nan, dtype=np.float32)
    errors[W - 1:] = err_w
    recon = np.full(s.n, np.nan, dtype=np.float32)
    recon[W - 1:] = meta.pop("recon_last")
    extra = {"recon": recon} if j["seed"] == cfg["seeds"][1] else {}   # keep one seed for figures
    meta.update({"file": j["file"], "W": W, "seed": j["seed"], "model": args.model,
                 "n_train_windows": len(trw), "protocol": args.protocol})
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, errors=errors, meta=json.dumps(meta), **extra)
    tmp.rename(p)
    el = time.time() - t_all
    n_done += 1
    eta = el / n_done * (len(jobs) - i)
    print(f"[{i}/{len(jobs)}] {s.name} W={W} seed={j['seed']} epochs={meta.get('epochs_run','-')} "
          f"{meta.get('train_seconds','-')}s  (elapsed {el/60:.1f} min, about {eta/60:.0f} min left)", flush=True)
print("shard done", flush=True)
