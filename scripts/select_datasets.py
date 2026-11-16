"""Step 0 - apply the label-location selection rule (PROTOCOL.md §2)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from atd import ROOT, load_config
from atd.data import load_labels, select_datasets

cfg = load_config()
sel = select_datasets(load_labels(cfg["labels_file"]), cfg["nab_dir"])
(ROOT / "configs/datasets.json").write_text(json.dumps(sel, indent=1))
print({k: len(v) for k, v in sel.items()})
