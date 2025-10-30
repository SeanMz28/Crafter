#!/bin/bash
set -euo pipefail

cd "$HOME/rl/CrafterPlus/Crafter"
mkdir -p logs runs

# Conda + env
source ~/.bashrc
conda activate crafter-rl-gpu

# Keep runtime clean & headless
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

# Prefer conda libs; avoid system CUDA
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
unset CUDA_HOME

# If you used pip cu121 wheels for torch, add:
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="$SITE_PKGS/torch/lib:$SITE_PKGS/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# Sanity info
nvidia-smi || true
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY

# Train
python3 train.py --timesteps 1000000 --outdir ./runs/trial11
echo "== DONE =="
