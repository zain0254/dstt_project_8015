"""Measure the runtime on THIS machine and decide the evaluation plan.

1. Trains a stratified sample of real jobs (smallest, median and largest
   training sets, several window sizes). They are normal jobs: their score
   files are kept and reused by the training step.
2. Fits seconds-per-unit-cost (cost = training windows x W) and projects the
   training time of every job in both protocols. The full-corpus protocol is
   projected at W = 100 (worst case), because W* is chosen later.
3. Times the threshold search on one file and projects the evaluation time.
4. If the projection is within the budget (default 15 h): plan = full NAB.
   Otherwise: plan = five real-data categories with the files that carry the
   most labelled anomaly windows, kept within the budget
   (configs/datasets.json is rewritten and the choice is logged).

    python scripts/estimate_runtime.py [--budget-hours 15] [--device cuda]
"""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from atd import ROOT, load_config, protocol
from atd.data import load_labels, load_series, scaled_series, make_windows, training_windows
from atd.pipeline import all_jobs, score_path


def n_train_windows(s, W, m, cfg):
    return len(training_windows(scaled_series(s), W, s.a, m["max_train_windows"],
                                s.labels if cfg.get("training_data") == "normal_only" else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=15.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", choices=["lstm", "standin"], default="lstm")
    ap.add_argument("--sample", type=int, default=6)
    args = ap.parse_args()
    cfg = load_config()
    labels = load_labels(cfg["labels_file"])
    from atd import datasets_path
    ds_path = datasets_path()
    ds = json.loads(ds_path.read_text())
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    from atd.model import train_and_score, standin_score
    fit = train_and_score if args.model == "lstm" else standin_score

    # ---- all jobs with their cost
    plans = {}
    for name in ("split", "full"):
        PR = protocol(cfg, name)
        jobs = all_jobs(cfg, ds, PR["jobs_key"], None if name == "split" else [max(cfg["window_sizes"])])
        for j in jobs:
            s = load_series(j["file"], cfg["nab_dir"], labels, PR["split"])
            j["units"] = n_train_windows(s, j["W"], PR["model"], cfg) * j["W"]
            j["protocol"] = name
        plans[name] = jobs

    # ---- time a stratified sample of real split-protocol jobs
    sj = sorted(plans["split"], key=lambda j: j["units"])
    pick = [sj[int(round(q * (len(sj) - 1)))] for q in np.linspace(0, 1, args.sample)]
    PR = protocol(cfg, "split"); m = PR["model"]
    rates = []
    print(f"timing {len(pick)} real fits on this machine ...", flush=True)
    for j in pick:
        s = load_series(j["file"], cfg["nab_dir"], labels, PR["split"])
        x = scaled_series(s)
        trw = training_windows(x, j["W"], s.a, m["max_train_windows"],
                               s.labels if cfg.get("training_data") == "normal_only" else None)
        t = time.time()
        err_w, meta = fit(trw, make_windows(x, j["W"]), seed=j["seed"], hidden=m["hidden_size"],
                          latent=m["latent_size"], lr=m["learning_rate"], batch_size=m["batch_size"],
                          max_epochs=m["max_epochs"], patience=m["early_stopping_patience"],
                          holdout=m["early_stopping_holdout"], threads=m["torch_threads"],
                          **({"device": args.device} if args.model == "lstm" else {}))
        dt = time.time() - t
        rates.append(dt / max(j["units"], 1))
        p = score_path(j["file"], j["W"], j["seed"], PR["scores"])
        if args.model == "lstm" and not p.exists():   # keep the real fit for the training step
            p.parent.mkdir(parents=True, exist_ok=True)
            errors = np.full(s.n, np.nan, dtype=np.float32); errors[j["W"] - 1:] = err_w
            recon = np.full(s.n, np.nan, dtype=np.float32); recon[j["W"] - 1:] = meta.pop("recon_last")
            extra = {"recon": recon} if j["seed"] == cfg["seeds"][1] else {}
            meta.update({"file": j["file"], "W": j["W"], "seed": j["seed"], "model": "lstm",
                         "n_train_windows": len(trw), "protocol": "split"})
            np.savez_compressed(p, errors=errors, meta=json.dumps(meta), **extra)
        print(f"  {s.name} W={j['W']}: {dt:.1f} s, {meta.get('epochs_run', '-')} epochs", flush=True)
    rate = float(np.percentile(rates, 75))          # conservative
    # full-protocol fits use a smaller batch and more epochs -> scale by the configured ratio
    mf = protocol(cfg, "full")["model"]
    full_factor = (m["batch_size"] / mf["batch_size"]) * (mf["max_epochs"] / m["max_epochs"]) ** 0.5
    t_split = sum(j["units"] for j in plans["split"]) * rate
    t_full = sum(j["units"] for j in plans["full"]) * rate * full_factor

    # ---- threshold evaluation: time the candidate search on one file
    from atd.pipeline import _tune_one
    f = sorted(ds["tuning_files"], key=lambda f: load_series(f, cfg["nab_dir"], labels).n)[len(ds["tuning_files"]) // 2]
    tmp = ROOT / "results/_estimate"; tmp.mkdir(parents=True, exist_ok=True)
    s = load_series(f, cfg["nab_dir"], labels)
    np.savez_compressed(score_path(f, 50, cfg["seeds"][0], tmp), errors=np.abs(np.random.default_rng(0).normal(0, 1, s.n)).astype(np.float32),
                        meta=json.dumps({"model": "estimate"}))
    t = time.time(); _tune_one((f, 50, cfg["seeds"][0], "configs/experiment.json", str(tmp))); dt = time.time() - t
    per_point = dt / (s.b - s.a)
    pts_split = sum((load_series(f, cfg["nab_dir"], labels).b - load_series(f, cfg["nab_dir"], labels).a)
                    for f in ds["tuning_files"]) * len(cfg["window_sizes"]) * len(cfg["seeds"])
    pts_split *= 1.6                                   # + test pass
    pts_full = sum(load_series(f, cfg["nab_dir"], labels, protocol(cfg, "full")["split"]).n * 0.85
                   for f in ds["full_corpus_files"]) * len(cfg["seeds"]) * 1.3
    t_eval = (pts_split + pts_full) * per_point / workers
    t_fig = 15 * 60
    total = t_split + t_full + t_eval + t_fig
    h = lambda x: x / 3600
    print(f"\nprojected runtime on this machine ({'GPU' if 'cuda' in str(meta.get('device', '')) else 'CPU'}):")
    print(f"  split protocol training : {h(t_split):5.1f} h  ({len(plans['split'])} fits)")
    print(f"  full-corpus training    : {h(t_full):5.1f} h  ({len(plans['full'])} fits, worst case W=100)")
    print(f"  threshold evaluation    : {h(t_eval):5.1f} h  ({workers} worker processes)")
    print(f"  figures/dashboard/report: {h(t_fig):5.1f} h")
    print(f"  TOTAL                   : {h(total):5.1f} h   (budget {args.budget_hours:.0f} h)")

    est = {"total_hours": h(total), "split_train_hours": h(t_split), "full_train_hours": h(t_full),
           "eval_hours": h(t_eval), "budget_hours": args.budget_hours, "seconds_per_unit": rate}
    if h(total) <= args.budget_hours:
        est["plan"] = "full_nab"
        print("\nDECISION: within budget -> full NAB evaluation (all 58 files).")
    else:
        # fallback: five real-data categories, files with the most scored anomaly windows first
        spare = args.budget_hours * 3600 - t_split - t_eval - t_fig
        if spare <= 0:
            (ROOT / "results/runtime_estimate.json").write_text(json.dumps(est, indent=1))
            sys.exit("DECISION: even the split protocol exceeds the budget on this machine; use a GPU (--device cuda).")
        PRf = protocol(cfg, "full")
        cand = []
        for f in ds["full_corpus_files"]:
            if not f.startswith("real"):
                continue
            s = load_series(f, cfg["nab_dir"], labels, PRf["split"])
            ev = len(s.windows_in(s.a, s.n))
            cost = sum(j["units"] for j in plans["full"] if j["file"] == f) * rate * full_factor
            cand.append((ev, int((s.labels[s.a:] == 0).sum()), f, cost))
        keep, used = [], 0.0
        for ev, nn, f, cost in sorted(cand, key=lambda c: (-c[0], -c[1])):
            if used + cost <= spare:
                keep.append(f); used += cost
        ds["full_corpus_files"] = sorted(keep)
        ds["full_eval_files"] = [f for f in ds["full_eval_files"] if f in keep]
        ds["full_clean_files"] = [f for f in ds["full_clean_files"] if f in keep]
        ds["plan"] = "five_categories"
        ds_path.write_text(json.dumps(ds, indent=1))
        est["plan"] = "five_categories"; est["files"] = keep
        print(f"\nDECISION: over budget -> five real-data categories, {len(keep)} files with the most "
              f"labelled anomaly windows (configs/datasets.json updated).")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/runtime_estimate.json").write_text(json.dumps(est, indent=1))
    import shutil; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
