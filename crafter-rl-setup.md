# Crafter RL Cluster Environment Setup
Conda + Stable-Baselines3 1.8.0 + Gym 0.21.0 (CPU) — reproducible setup for the MS cluster.

**Why these pins?**
- `stable-baselines3==1.8.0` depends on the classic Gym API (`gym==0.21.0`).
- `gym==0.21.0` is easiest via **conda-forge** wheels on Python 3.10.
- `numpy==1.26.4` + `matplotlib==3.8.4` avoid ABI mismatches during plotting.
- `PYTHONNOUSERSITE=1` ensures `~/.local` doesn’t pollute your conda env on the cluster.

---

## 0) Create & activate the conda environment
```bash
# If conda is not initialized in your shell:
# echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc && source ~/.bashrc

conda create -y -n crafter-rl python=3.10
conda activate crafter-rl

# Keep installs inside the env (avoid ~/.local shadowing)
export PYTHONNOUSERSITE=1
pip config set global.user false
```

## 1) Install the tricky bits first (conda-forge wheels)
```bash
# Gym 0.21.0: use conda-forge to avoid building from source on Py3.10
conda install -y -c conda-forge "gym==0.21.0"

# Numeric/plot stack (compatible builds that prevent NumPy/Matplotlib ABI crashes)
conda install -y -c conda-forge "numpy==1.26.4" "matplotlib==3.8.4" "pandas==2.2.3"   contourpy cycler fonttools kiwisolver pyparsing packaging pillow
```

## 2) PyTorch (CPU) + RL libraries
```bash
# PyTorch (CPU build)
python -m pip install --no-user --index-url https://download.pytorch.org/whl/cpu torch

# Torch runtime deps (sometimes not pulled automatically)
python -m pip install --no-user "sympy>=1.13.3" "networkx>=2.5.1" "fsspec>=0.8.5" filelock

# Stable-Baselines3 1.8.0 (install without deps so it doesn't try to alter gym)
python -m pip install --no-user --no-deps "stable-baselines3==1.8.0"

# Gymnasium + shimmy + Crafter + helpers
python -m pip install --no-user "gymnasium==1.2.1" "shimmy>=0.2.1" "crafter==1.8.3"   farama-notifications opensimplex ruamel.yaml imageio imageio-ffmpeg tqdm tensorboard

# Headless plotting on the cluster
export MPLBACKEND=Agg
```

## 3) Sanity check
```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import sys, importlib.util as iu
import torch, gym, gymnasium, stable_baselines3 as sb3, numpy as np, pandas as pd
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__, "| pandas:", pd.__version__)
print("torch:", torch.__version__)
print("gym:", gym.__version__, "| gymnasium:", gymnasium.__version__)
print("SB3:", sb3.__version__)
print("crafter present:", bool(iu.find_spec("crafter")))
PY
```

Expected output (versions may show the same or newer micro versions where allowed):
```
Python: 3.10.x
NumPy: 1.26.4 | pandas: 2.2.3
torch: 2.9.0+cpu
gym: 0.21.0 | gymnasium: 1.2.1
SB3: 1.8.0
crafter present: True
```

## 4) Run your training script
```bash
# from your project folder
PYTHONNOUSERSITE=1 python3 train.py --timesteps 1000 --outdir ./test_run --no-eval

# Tip: for a visible learning smoke test, override warmup:
# PYTHONNOUSERSITE=1 python3 train.py --timesteps 1000 --outdir ./test_run --no-eval --learning-starts 0
```

---

## (Optional) SLURM batch script
Create `train_crafter.sbatch`:
```bash
#!/bin/bash
#SBATCH -J crafter-1000
#SBATCH -p bigbatch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o /home-mscluster/$USER/logs/%x.%j.out
#SBATCH -e /home-mscluster/$USER/logs/%x.%j.err
# GPU example (commented):
# SBATCH --gres=gpu:1

source ~/.bashrc
conda activate crafter-rl
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

cd "$SLURM_SUBMIT_DIR"
mkdir -p ./test_run

python3 train.py --timesteps 1000 --outdir ./test_run --no-eval
```

Submit:
```bash
mkdir -p /home-mscluster/$USER/logs
sbatch train_crafter.sbatch
squeue --me
# scancel <JOBID>  # to cancel
```

---

## Notes / Troubleshooting
- **Avoid `~/.local` pollution:** Always keep `export PYTHONNOUSERSITE=1` in your shell and SLURM jobs.  
- **Gym 0.21.0 build error?** Ensure it comes from `conda-forge` (as above).  
- **Plotting crash about NumPy ABI:** Reinstall NumPy & Matplotlib from the same channel (`conda-forge`) with the versions pinned above.  
- **Different hardware/gpu?** If using GPU nodes, install a CUDA build of PyTorch instead of the CPU wheel, and uncomment `--gres=gpu:1`.
