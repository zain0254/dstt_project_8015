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
from atd import ROOT, load_config
from atd.data import load_labels, load_series, scaled_series, make_windows, training_windows
from atd.pipeline import all_jobs, shard_jobs, score_path, SCORES_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--model", choices=["lstm", "standin"], default="lstm",
                help="'standin' = PCA, for pipeline smoke tests only")
ap.add_argument("--out", default=str(SCORES_DIR))
ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs (smoke test)")
args = ap.parse_args()

cfg = load_config()
datasets = json.loads((ROOT / "configs/datasets.json").read_text())
labels = load_labels(cfg["labels_file"])
jobs = shard_jobs(all_jobs(cfg, datasets), args.shard, args.of)
if args.limit:
    jobs = jobs[: args.limit]
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
m = cfg["model"]
if args.model == "lstm":
    from atd.model import train_and_score as fit
else:
    from atd.model import standin_score as fit

print(f"shard {args.shard}/{args.of}: {len(jobs)} jobs", flush=True)
t_all = time.time()
for i, j in enumerate(jobs, 1):
    p = score_path(j["file"], j["W"], j["seed"], out)
    if p.exists():
        continue
    s = load_series(j["file"], cfg["nab_dir"], labels)
    x = scaled_series(s)
    W = j["W"]
    trw = training_windows(x, W, s.a, m["max_train_windows"])
    allw = make_windows(x, W)
    err_w, meta = fit(trw, allw, seed=j["seed"], hidden=m["hidden_size"], latent=m["latent_size"],
                      lr=m["learning_rate"], batch_size=m["batch_size"], max_epochs=m["max_epochs"],
                      patience=m["early_stopping_patience"], holdout=m["early_stopping_holdout"],
                      threads=m["torch_threads"])
    errors = np.full(s.n, np.nan, dtype=np.float32)
    errors[W - 1:] = err_w
    meta.update({"file": j["file"], "W": W, "seed": j["seed"], "model": args.model,
                 "n_train_windows": len(trw)})
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, errors=errors, meta=json.dumps(meta))
    tmp.rename(p)
    el = time.time() - t_all
    print(f"[{i}/{len(jobs)}] {s.name} W={W} seed={j['seed']} epochs={meta.get('epochs_run','-')} "
          f"{meta.get('train_seconds','-')}s  (elapsed {el/60:.1f} min)", flush=True)
print("shard done", flush=True)
