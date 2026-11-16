"""Step 2 - tune thresholds on validation, evaluate on test, run statistics.

Reads the saved reconstruction errors (results/scores/*.npz) and writes:
  results/tuning_candidates.csv   validation objective of every candidate
  results/tuning_summary.csv      mean over tuning files and seeds
  results/selected_params.csv     chosen parameters per method and W
  results/window_selection.csv    validation ROC-AUC per W (chooses W*)
  results/test_metrics.csv        every test metric, every file/W/seed/method
  results/per_file_metrics.csv    seed-averaged at W* (unit of analysis)
  results/summary_by_method.csv   means over evaluation files at W*
  results/corpus_nab.csv          official-style corpus NAB scores
  results/comparisons.csv         P1 vs every other method, Wilcoxon + Holm
  results/hypotheses.json         H1-H3 decisions (rules from PROTOCOL.md)
  results/traces/*.npz            test thresholds at W* (for figures)
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from atd import ROOT, load_config
from atd.data import file_id
from atd.pipeline import tune, test, SCORES_DIR
from atd.stats import per_file, corpus_nab, comparison_table, decide_hypotheses

ap = argparse.ArgumentParser()
ap.add_argument("--scores", default=str(SCORES_DIR))
ap.add_argument("--out", default=str(ROOT / "results"))
ap.add_argument("--workers", type=int, default=2)
args = ap.parse_args()

cfg = load_config()
datasets = json.loads((ROOT / "configs/datasets.json").read_text())
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

print("tuning on validation partitions ...", flush=True)
cand, agg, selected, wsel, w_star = tune(datasets=datasets, scores_dir=args.scores, workers=args.workers)
cand.to_csv(out / "tuning_candidates.csv", index=False)
agg.to_csv(out / "tuning_summary.csv", index=False)
selected.to_csv(out / "selected_params.csv", index=False)
wsel.assign(selected=wsel.W == w_star).to_csv(out / "window_selection.csv", index=False)
print("W* =", w_star)
print(selected[selected.W == w_star][["method", "params", "val_composite_f1"]].to_string(index=False))

print("evaluating on test partitions ...", flush=True)
tdf, traces = test(selected, datasets=datasets, scores_dir=args.scores, workers=args.workers,
                   keep_traces_for_W=w_star)
tdf.to_csv(out / "test_metrics.csv", index=False)
tr_dir = out / "traces"; tr_dir.mkdir(exist_ok=True)
for (f, sd), tr in traces.items():
    np.savez_compressed(tr_dir / f"{file_id(f)}__s{sd}.npz", **tr)

pf = per_file(tdf, w_star)
pf.to_csv(out / "per_file_metrics.csv", index=False)
metrics = ["composite_f1", "f1", "precision", "recall", "event_recall", "event_precision", "event_f1",
           "fpr", "hv_fpr", "calm_fpr", "nab_standard", "nab_reward_low_fp", "nab_reward_low_fn", "nab_standard_pointwise", "n_alerts",
           "detection_delay_rel", "roc_auc", "pr_auc", "alarm_rate"]
summ = pf.groupby("method")[metrics].mean().reset_index()
corp = corpus_nab(tdf, w_star, "standard")
for prof in ("reward_low_fp", "reward_low_fn", "standard_pointwise"):
    corp = corp.merge(corpus_nab(tdf, w_star, prof), on="method")
corp.to_csv(out / "corpus_nab.csv", index=False)
summ.merge(corp, on="method").to_csv(out / "summary_by_method.csv", index=False)

P = cfg["proposed_method"]
others = [m for m in cfg["thresholds"] if m != P]
comparison_table(pf, P, others, metrics[:14]).to_csv(out / "comparisons.csv", index=False)
hyp = decide_hypotheses(pf, corp, cfg)
(out / "hypotheses.json").write_text(json.dumps({"W_star": w_star, "hypotheses": hyp}, indent=1, default=float))
for h in hyp:
    print(h["hypothesis"], h["role"], "->", h["verdict"])
print(summ.merge(corp, on="method")[["method", "composite_f1", "f1", "event_recall", "fpr", "hv_fpr", "n_alerts",
                                     "corpus_nab_standard", "corpus_nab_standard_pointwise"]].round(3).to_string(index=False))
