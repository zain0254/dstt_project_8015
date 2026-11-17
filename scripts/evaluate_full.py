"""Full-corpus protocol (PRIMARY): all 58 NAB files.

The LSTM-AE was trained on the first 15 % of every file (unlabelled). The
other 85 % is scored. Threshold parameters are chosen leave-one-file-out:
a file is scored with the parameters that were best on the other 57 files.
Uses the window size W* selected in the split protocol. Writes results/full/.
"""
import argparse, json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from atd import ROOT, load_config, protocol, load_datasets
from atd.data import file_id
from atd.pipeline import evaluate_full
from atd.stats import per_file, corpus_nab, comparison_table, decide_hypotheses, base_paper_block


def main():
    cfg = load_config()
    PR = protocol(cfg, "full")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(PR["scores"]))
    ap.add_argument("--out", default=str(PR["results"]))
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--allow-standin", action="store_true", help="pipeline test only")
    args = ap.parse_args()
    models = {json.loads(str(np.load(f)["meta"])).get("model") for f in Path(args.scores).glob("*.npz")}
    if models != {"lstm"} and not args.allow_standin:
        raise SystemExit(f"score files come from {models}; reported results must come from the LSTM-AE only")
    datasets = load_datasets()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    wsel = pd.read_csv(ROOT / "results/split/window_selection.csv") if not args.window else None
    W = args.window or int(wsel[wsel.selected].W.iloc[0])
    print(f"full-corpus protocol: {len(datasets['full_corpus_files'])} files, W* = {W}; leave-one-file-out tuning ...", flush=True)

    cand, chosen, pooled, tdf, traces = evaluate_full(W, PR["split"], args.scores, datasets=datasets, workers=args.workers)
    cand.assign(W=W).to_csv(out / "tuning_candidates.csv", index=False)
    chosen.to_csv(out / "lofo_selected_params.csv", index=False)
    pooled.assign(W=W).to_csv(out / "tuning_summary.csv", index=False)
    # most frequent leave-one-out choice per method (for reporting)
    mode = (chosen.groupby(["method", "params"]).size().rename("n_files").reset_index()
            .sort_values(["method", "n_files"], ascending=[True, False]).groupby("method").head(1))
    mode = mode.merge(pooled, on=["method", "params"], how="left").assign(W=W)
    mode.to_csv(out / "selected_params.csv", index=False)
    pd.DataFrame({"W": [W], "val_roc_auc": [np.nan], "selected": [True]}).to_csv(out / "window_selection.csv", index=False)

    cdf = tdf[tdf.set == "clean"].drop(columns="set")
    tdf = tdf.drop(columns="set")            # corpus NAB uses every file, as NAB does
    edf = tdf[tdf.file.isin(datasets["full_eval_files"])]
    tdf.to_csv(out / "test_metrics.csv", index=False)
    tr_dir = out / "traces"; tr_dir.mkdir(exist_ok=True)
    for (f, sd), tr in traces.items():
        if sd == cfg["seeds"][1]:
            np.savez_compressed(tr_dir / f"{file_id(f)}__s{sd}.npz", **tr)

    pf = per_file(edf, W)
    pf.to_csv(out / "per_file_metrics.csv", index=False)
    cpf = per_file(cdf, W)
    cpf.to_csv(out / "clean_stream_per_file.csv", index=False)
    metrics = ["composite_f1", "f1", "precision", "recall", "event_recall", "event_precision", "event_f1",
               "fpr", "hv_fpr", "calm_fpr", "nab_standard", "nab_reward_low_fp", "nab_reward_low_fn",
               "nab_standard_pointwise", "n_alerts", "alerts_per_1000",
               "detection_delay_rel", "roc_auc", "pr_auc", "alarm_rate"]
    summ = pf.groupby("method")[metrics].mean().reset_index()
    corp = corpus_nab(tdf, W, "standard")
    for prof in ("reward_low_fp", "reward_low_fn", "standard_pointwise"):
        corp = corp.merge(corpus_nab(tdf, W, prof), on="method")
    corp.to_csv(out / "corpus_nab.csv", index=False)
    clean_summ = (cpf.groupby("method")[["fpr", "alerts_per_1000"]].mean()
                  .rename(columns={"fpr": "clean_fpr", "alerts_per_1000": "clean_alerts_per_1000"}).reset_index())
    ev = edf[edf.seed == cfg["seeds"][0]].groupby("method")[["n_events", "events_detected"]].sum().reset_index()
    summ = summ.merge(corp, on="method").merge(clean_summ, on="method", how="left").merge(ev, on="method")
    summ.to_csv(out / "summary_by_method.csv", index=False)

    P = cfg["proposed_method"]
    others = [m for m in cfg["thresholds"] if m != P]
    comp = comparison_table(pf, P, others, metrics[:16])
    if len(cpf):
        cc = comparison_table(cpf, P, others, ["fpr", "alerts_per_1000"]); cc["metric"] = "clean_" + cc["metric"]
        comp = pd.concat([comp, cc])
    comp.to_csv(out / "comparisons.csv", index=False)
    hyp = decide_hypotheses(pf, corp, cfg, clean=cpf)
    (out / "hypotheses.json").write_text(json.dumps({"W_star": W, "protocol": "full", "hypotheses": hyp,
                                                    "base_paper": base_paper_block(pf, cfg)}, indent=1, default=float))
    print(f"events scored: {int(summ.n_events.iloc[0])} in {pf.file.nunique()} files (+{cpf.file.nunique()} anomaly-free files)")
    for h in hyp:
        print(h["hypothesis"], h["role"], "->", h["verdict"])
    print(summ[["method", "composite_f1", "f1", "event_recall", "events_detected", "fpr", "hv_fpr",
                "corpus_nab_standard"]].round(3).to_string(index=False))


if __name__ == "__main__":   # required on Windows
    main()
