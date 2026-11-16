"""Step 1 - data acquisition: download NAB v1.1 from Numenta's GitHub.

Downloads labels/combined_windows.json and every labelled real* and
artificialWithAnomaly CSV, pinned to a fixed commit for reproducibility,
and writes data/nab/MANIFEST.csv with row counts and SHA-256 checksums.
Already-downloaded files are verified and skipped.
"""
import csv, hashlib, json, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "ea702d75cc2258d9d7dd35ca8e5e2539d71f3140"   # numenta/NAB master, 3 Dec 2024
BASE = f"https://raw.githubusercontent.com/numenta/NAB/{COMMIT}"
OUT = ROOT / "data" / "nab"


def fetch(rel: str, dest: Path, retries: int = 4) -> bytes:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    for i in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/{rel}", timeout=60) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return data
        except Exception as e:  # network hiccup -> retry
            if i == retries - 1:
                raise RuntimeError(f"could not download {rel}: {e}")
            time.sleep(2 * (i + 1))


def git_fallback() -> Path:
    """Fetch the same pinned commit with git (used if raw HTTP is blocked)."""
    import subprocess
    cache = ROOT / "data" / "_nab_git"
    if not (cache / "labels" / "combined_windows.json").exists():
        cache.mkdir(parents=True, exist_ok=True)
        run = lambda *a: subprocess.run(["git", *a], cwd=cache, check=True, capture_output=True)
        run("init", "-q"); run("remote", "add", "origin", "https://github.com/numenta/NAB.git")
        run("fetch", "-q", "--depth", "1", "origin", COMMIT); run("checkout", "-q", "FETCH_HEAD")
    return cache


def fetch_any(rel, dest, use_git):
    if use_git:
        src = git_fallback() / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(src.read_bytes())
        return dest.read_bytes()
    return fetch(rel, dest)


def main():
    try:
        fetch("labels/combined_windows.json", OUT / "labels" / "combined_windows.json", retries=2)
        use_git = False
    except RuntimeError:
        print("raw download unavailable -> fetching the pinned commit with git", flush=True)
        use_git = True
    labels_raw = fetch_any("labels/combined_windows.json", OUT / "labels" / "combined_windows.json", use_git)
    labels = json.loads(labels_raw)
    files = sorted(f for f in labels if f.startswith("real") or f.startswith("artificialWithAnomaly"))
    rows = []
    for i, f in enumerate(files, 1):
        data = fetch_any(f"data/{f}", OUT / f, use_git)
        rows.append({"file": f, "rows": data.count(b"\n") - 1, "anomaly_windows": len(labels[f]),
                     "sha256": hashlib.sha256(data).hexdigest()})
        print(f"[{i}/{len(files)}] {f}", flush=True)
    with open(OUT / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"NAB commit {COMMIT}: {len(rows)} files, {sum(r['rows'] for r in rows):,} rows -> {OUT}")


if __name__ == "__main__":
    main()
