"""Adaptive thresholding for LSTM-Autoencoder anomaly detection on NAB."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/experiment.json") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with open(p) as f:
        return json.load(f)


PROTOCOLS = ("full", "split")


def datasets_path() -> Path:
    """configs/datasets.json, or the file named by ATD_DATASETS (pipeline smoke tests only)."""
    import os
    return Path(os.environ.get("ATD_DATASETS", ROOT / "configs/datasets.json"))


def load_datasets() -> dict:
    return json.loads(datasets_path().read_text())
_ALIAS = {"A": "split", "B": "full"}


def protocol(cfg: dict, name: str = "full") -> dict:
    """Settings of the two evaluation protocols (PROTOCOL.md).

    full  = PRIMARY. All 58 NAB files; the LSTM-AE learns from the first 15 %
            of each file (NAB probationary period), the other 85 % is scored;
            threshold parameters are chosen leave-one-file-out.
    split = proposal protocol. Chronological 60/20/20 split per file;
            thresholds tuned on validation partitions, scored on test."""
    name = _ALIAS.get(name, name)
    if name == "split":
        sp = cfg["split"]
        return {"name": "split", "split": (sp["train"], sp["validation"], sp["test"]),
                "scores": ROOT / "results/split/scores", "results": ROOT / "results/split",
                "figures": ROOT / "figures/split", "dashboard": ROOT / "dashboard/split.html",
                "model": cfg["model"], "eval_key": "evaluation_files", "jobs_key": "all_files",
                "label": "Split protocol (proposal, 60/20/20)", "segment": "test partition"}
    sp = cfg["full_protocol"]["split"]
    return {"name": "full", "split": (sp["train"], sp["validation"], sp["test"]),
            "scores": ROOT / "results/full/scores", "results": ROOT / "results/full",
            "figures": ROOT / "figures/full", "dashboard": ROOT / "dashboard/index.html",
            "model": {**cfg["model"], **cfg["full_protocol"]["model_overrides"]},
            "eval_key": "full_eval_files", "jobs_key": "full_corpus_files",
            "label": "Full-corpus protocol (primary)", "segment": "scored 85 % of each series"}
