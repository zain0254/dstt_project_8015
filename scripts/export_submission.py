"""Copy generated outputs (no code) into the submission repository.

    python scripts/export_submission.py ..\\submission-repo

Copies: paper/ (PDF + LaTeX), figures/, dashboard/, results tables (CSV/JSON,
no raw score files), PROTOCOL.md, data manifest, and writes the submission
README (from submission_template/README.md, with the headline results filled in).
"""
import json, shutil, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
dest = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "submission"
dest.mkdir(parents=True, exist_ok=True)


def copytree(src, dst, pattern="*"):
    if not src.exists():
        return 0
    n = 0
    for f in src.rglob(pattern):
        if f.is_file():
            t = dst / f.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
            n += 1
    return n


counts = {
    "figures": copytree(ROOT / "figures", dest / "figures", "*.png"),
    "dashboard": copytree(ROOT / "dashboard", dest / "dashboard"),
    "paper": copytree(ROOT / "paper", dest / "paper"),
}
res = dest / "results"; res.mkdir(exist_ok=True)
n = 0
for f in (ROOT / "results").glob("*"):
    if f.suffix in (".csv", ".json") and f.is_file():
        shutil.copy2(f, res / f.name); n += 1
counts["result tables"] = n
for f in ("PROTOCOL.md",):
    shutil.copy2(ROOT / f, dest / f)
if (ROOT / "data/nab/MANIFEST.csv").exists():
    (dest / "data").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data/nab/MANIFEST.csv", dest / "data" / "NAB_MANIFEST.csv")
shutil.copy2(ROOT / "configs/experiment.json", dest / "results" / "experiment_config.json")
shutil.copy2(ROOT / "configs/datasets.json", dest / "results" / "datasets.json")

# README with the headline table filled in
tpl = (ROOT / "submission_template/README.md").read_text(encoding="utf-8")
table = "*Results not available yet.*"
if (ROOT / "results/summary_by_method.csv").exists():
    s = pd.read_csv(ROOT / "results/summary_by_method.csv").set_index("method")
    order = ["B1_fixed_percentile", "B2_fixed_mean_ksigma", "B3_rolling_mean_3sigma", "P1_moving_median_mad"]
    names = {"B1_fixed_percentile": "B1 Fixed percentile", "B2_fixed_mean_ksigma": "B2 Fixed mean + kσ",
             "B3_rolling_mean_3sigma": "B3 Rolling mean + 3σ (base paper)", "P1_moving_median_mad": "**P1 Moving median + λ·MAD (proposed)**"}
    rows = ["| Method | Composite F1 ↑ | Point F1 ↑ | Event recall ↑ | FPR ↓ | High-var. FPR ↓ | NAB standard ↑ | NAB low-FP ↑ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k in order:
        r = s.loc[k]
        rows.append(f"| {names[k]} | {r.composite_f1:.3f} | {r.f1:.3f} | {r.event_recall:.3f} | {r.fpr:.3f} | "
                    f"{r.hv_fpr:.3f} | {r.corpus_nab_standard:.1f} | {r.corpus_nab_reward_low_fp:.1f} |")
    hyp = json.loads((ROOT / "results/hypotheses.json").read_text())
    rows.append("")
    rows.append(f"Window size W* = {hyp['W_star']} (chosen on validation). Hypothesis decisions (rules fixed in PROTOCOL.md):")
    rows.append("")
    for h in hyp["hypotheses"]:
        rows.append(f"- **{h['hypothesis']}** ({h['role']}, {h['metric']}): {h['verdict']}")
    table = "\n".join(rows)
(dest / "README.md").write_text(tpl.replace("{{RESULTS}}", table), encoding="utf-8")
print("exported to", dest, counts)
