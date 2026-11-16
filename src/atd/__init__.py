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
