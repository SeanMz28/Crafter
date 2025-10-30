# Crafter RL – GPU Environment Setup (mscluster)

Reproducible instructions to create **`crafter-rl-gpu`** (CUDA) and run **Crafter DQN** on a GPU node with Slurm.

> ✅ These commands avoid common CUDA/NumPy/Matplotlib mismatches on the cluster.  
> ✅ We **do not** export any system CUDA paths (`/usr/local/cuda-*`) when using PyTorch **cu121** wheels.

---

## 0) Create & activate the Conda environment
```bash
conda create -y -n crafter-rl-gpu python=3.10
conda activate crafter-rl-gpu

# Keep installs inside the env (avoid ~/.local pollution)
export PYTHONNOUSERSITE=1
pip config set global.user false
python -m pip install --upgrade --no-user pip
```

## 1) Numeric & plotting stack (stable ABI)
```bash
conda install -y -c conda-forge \
  "numpy<2" "pandas>=2.2,<2.3" "matplotlib=3.8.*" \
  contourpy cycler fonttools kiwisolver pyparsing packaging pillow cloudpickle
```

## 2) PyTorch with GPU (CUDA 12.1 runtime **bundled** in the wheel)
> We rely on PyTorch’s **cu121** wheels so we don’t need system CUDA.
```bash
python -m pip install --no-cache-dir \
  torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## 3) RL stack (Gym 0.21 via conda-forge, SB3 1.8.0 without deps)
```bash
# Gym 0.21.0 must be from conda-forge on Python 3.10
conda install -y -c conda-forge "gym==0.21.0"

# Stable-Baselines3 pinned and installed without deps (prevents changing gym)
python -m pip install --no-user --no-deps "stable-baselines3==1.8.0"

# Gymnasium + Crafter + helpers
python -m pip install --no-user \
  "gymnasium==1.2.1" "shimmy>=0.2.1" "crafter==1.8.3" \
  farama-notifications opensimplex ruamel.yaml imageio imageio-ffmpeg tensorboard tqdm
```

## 4) Ensure Torch runtime deps are in the env
```bash
python -m pip install --no-user filelock "fsspec>=0.8.5" "sympy>=1.13.3" "networkx>=2.5.1" typing-extensions
```

## 5) Avoid CUDA mismatch (IMPORTANT)
**Do not** export `/usr/local/cuda-12.0` (or any system CUDA) in this shell or in your job script.  
Tell the runtime to prefer your **conda** libraries:
```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
```

> If you previously added CUDA 12.0 exports to `~/.bashrc`, remove them (or they’ll shadow cu121):
> ```bash
> sed -i '/\/usr\/local\/cuda-12\.0/d' ~/.bashrc
> source ~/.bashrc
> ```

## 6) Sanity check GPU
```bash
PYTHONNOUSERSITE=1 python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
```

Expected:
```
CUDA available: True
GPU: NVIDIA ...
Torch: 2.4.0+cu121 CUDA: 12.1
```

---

## 7) Train on a GPU node via Slurm

### 7.1 Create a batch script `train_gpu.sbatch`
> Uses the env above, stages to local `/scratch` for speed, writes results to `/gluster`.
```bash
cat > train_gpu.sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=crafter-dqn
#SBATCH --partition=bigbatch          # try bigbatch first (RTX 3090, 24 GB)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home-mscluster/%u/rl_logs/%x-%j.out
#SBATCH --error=/home-mscluster/%u/rl_logs/%x-%j.err

set -euo pipefail

# Avoid ~/.local pollution and headless plotting
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

# Activate env
eval "$(/home-mscluster/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate crafter-rl-gpu

# Prefer conda libs (do not export any /usr/local/cuda-* here)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

# Paths
PROJECT_DIR="/home-mscluster/$USER/rl/CrafterPlus/Crafter"
RUN_ID="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
SCRATCH_RUN="/scratch/$USER/${RUN_ID}"              # fast local
OUTDIR="/gluster/$USER/crafter_runs/${RUN_ID}"      # keep results

mkdir -p "$SCRATCH_RUN" "$OUTDIR" "/home-mscluster/$USER/rl_logs"
rsync -a --delete "$PROJECT_DIR/" "$SCRATCH_RUN/"
cd "$SCRATCH_RUN"

# Info
echo "== Node & GPU =="
nvidia-smi || true
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
PY

# Train (tweak args as needed)
python3 train.py \
  --timesteps 200000 \
  --outdir "$OUTDIR" \
  --no-eval

echo "Outputs in: $OUTDIR"
SBATCH
```

### 7.2 Submit & monitor
```bash
mkdir -p /home-mscluster/$USER/rl_logs
sbatch train_gpu.sbatch
squeue -u $USER
# scancel <JOBID>  # cancel if needed
```

> Tip: for a quick smoke test with visible learning updates, you can pass `--learning-starts 0` to your `train.py` args.

---

## 8) Short interactive GPU check (optional)
```bash
srun -p bigbatch --gres=gpu:1 --time=00:05:00 --cpus-per-task=2 --mem=8G --pty bash
# inside the node:
eval "$(/home-mscluster/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate crafter-rl-gpu
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.cuda)"
exit
```

---

## 9) Alternative (Conda-based CUDA instead of pip cu121)
If you prefer **Conda PyTorch with CUDA 12.1** provided by NVIDIA/PyTorch channels (instead of the pip wheel), use:
```bash
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```
Then keep the same steps for Gym/SB3/Crafter. Still **avoid exporting** `/usr/local/cuda-*` in your job script.
