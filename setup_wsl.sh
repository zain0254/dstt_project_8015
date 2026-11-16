#!/usr/bin/env bash
# One-time setup on Windows WSL2 (Ubuntu 22.04/24.04) with an NVIDIA GPU.
#   bash setup_wsl.sh
set -e
cd "$(dirname "$0")"
echo "== system packages"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip git unzip
echo "== python environment (.venv)"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install numpy pandas scipy scikit-learn matplotlib
echo "== checks"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || echo "WARNING: nvidia-smi not found (update the Windows NVIDIA driver)"
.venv/bin/python -c "import torch; print('PyTorch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
.venv/bin/python tests/test_core.py
echo
echo "Setup done. Next:"
echo "  source .venv/bin/activate"
echo "  python scripts/train_score.py --limit 2     # 2-minute GPU check"
echo "  python run_pipeline.py                      # full run"
