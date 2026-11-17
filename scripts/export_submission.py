"""Copy generated outputs (no code) into the submission repository.

    python scripts/export_submission.py ..\\submission-repo

Copies: paper/ (PDF + LaTeX), figures/, dashboard/, results tables (CSV/JSON,
no raw score files), PROTOCOL.md, data manifest, and writes the submission
README (from submission_template/README.md, with the headline results filled in).
"""
import json, shutil, sys
from pathlib import Path

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
for f in (ROOT / "results").rglob("*"):
    if f.is_file() and f.suffix in (".csv", ".json") and "scores" not in f.parts:
        t = res / f.relative_to(ROOT / "results"); t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, t); n += 1
counts["result tables"] = n
counts["pdf figures"] = copytree(ROOT / "figures", dest / "figures", "*.pdf")
for f in ("PROTOCOL.md",):
    shutil.copy2(ROOT / f, dest / f)
for f in ("docs/RESULTS.md", "docs/CHANGES.md", "docs/METHODOLOGY.md"):
    if (ROOT / f).exists():
        (dest / "docs").mkdir(exist_ok=True); shutil.copy2(ROOT / f, dest / f)
if (ROOT / "data/nab/MANIFEST.csv").exists():
    (dest / "data").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "data/nab/MANIFEST.csv", dest / "data" / "NAB_MANIFEST.csv")
shutil.copy2(ROOT / "configs/experiment.json", res / "experiment_config.json")
shutil.copy2(ROOT / "configs/datasets.json", res / "datasets.json")

# README: reuse the results block that make_report.py wrote into README.md
tpl = (ROOT / "submission_template/README.md").read_text(encoding="utf-8")
rd = (ROOT / "README.md").read_text(encoding="utf-8")
a, b = rd.find("<!-- RESULTS_START -->"), rd.find("<!-- RESULTS_END -->")
block = rd[a + len("<!-- RESULTS_START -->"):b].strip() if a >= 0 and b > a else "*Results not available yet.*"
(dest / "README.md").write_text(tpl.replace("{{RESULTS}}", block), encoding="utf-8")
print("exported to", dest, counts)
