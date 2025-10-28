# Cluster Setup Guide for DQN Crafter

## Quick Setup

### 1. Initial Setup (One-time)

```bash
# Make setup script executable
chmod +x setup_environment.sh

# Run setup script
./setup_environment.sh
```

This will:
- Create a virtual environment
- Install all required packages
- Set up Crafter environment

### 2. Manual Setup (Alternative)

If the script doesn't work on your cluster:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Required Packages

- **gym** (0.21.0) - RL environment interface
- **stable-baselines3** (2.0.0) - DQN implementation
- **torch** (>=1.13.0) - Neural network backend
- **numpy** - Numerical operations
- **matplotlib** - Plotting (uses non-interactive backend)
- **crafter** - The Crafter environment

## Running on Cluster

### Interactive Session (Testing)

```bash
# Activate environment
source venv/bin/activate

# Quick test run
python train.py --timesteps 20000 --outdir ./test_results

# Full training run (not recommended for interactive)
python train.py --timesteps 500000 --outdir ./results/seed_0
```

### Batch Job (Recommended)

```bash
# Create logs directory
mkdir -p logs

# Make job script executable
chmod +x cluster_job.sh

# Submit job
sbatch cluster_job.sh
```

### Monitor Job

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/dqn_crafter_<JOB_ID>.out

# View errors
tail -f logs/dqn_crafter_<JOB_ID>.err
```

## Cluster-Specific Adjustments

### If Your Cluster Uses Environment Modules

Edit `cluster_job.sh` and uncomment/adjust:
```bash
module load python/3.10
module load cuda/11.8  # If using GPU
```

### If You Have GPU Access

1. In `cluster_job.sh`, uncomment:
   ```bash
   #SBATCH --gres=gpu:1
   #SBATCH --partition=gpu
   ```

2. Install GPU version of PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Memory Issues

If you run out of memory, reduce buffer size in your training command:
```bash
python train.py --timesteps 500000 --buffer-size 50000 --outdir ./results
```

Or adjust in `cluster_job.sh`:
```bash
#SBATCH --mem=32G  # Increase memory allocation
```

## Expected Resource Usage

- **Memory**: ~8-12 GB (depends on buffer size)
- **CPU**: 4 cores recommended
- **Time**: ~4-8 hours for 500K timesteps (CPU)
- **Storage**: ~100-500 MB per run

## Output Files

After training completes, you'll find in your output directory:

```
results/seed_0/
├── dqn_model_final.zip          # Trained model
├── config.json                  # Configuration used
├── stats.jsonl                  # Episode statistics
├── monitor.monitor.csv          # Training metrics
├── training_curves.png          # Visualization
└── eval/
    ├── evaluation_results.json  # Evaluation metrics (JSON)
    ├── evaluation_report.txt    # Evaluation report (TXT)
    └── stats.jsonl             # Evaluation episode data
```

## Multiple Seeds

To run with different random seeds (for statistical significance):

```bash
# Submit multiple jobs with different seeds
sbatch --export=SEED=0 cluster_job.sh
sbatch --export=SEED=1 cluster_job.sh
sbatch --export=SEED=2 cluster_job.sh
```

Then modify `cluster_job.sh` to use the SEED variable:
```bash
python train.py \
    --timesteps 500000 \
    --seed ${SEED:-0} \
    --outdir ./results/seed_${SEED:-0}
```

## Troubleshooting

### Package Installation Fails

If Crafter installation fails:
```bash
# Install build tools first
pip install setuptools wheel
pip install git+https://github.com/danijar/crafter.git
```

### Display/Qt Errors

Already fixed! The code uses non-interactive matplotlib backend (`Agg`), so no display is needed.

### Slow Training

- Ensure you're using enough CPUs: `--cpus-per-task=4` or more
- Consider using GPU if available
- Training 500K timesteps takes several hours - this is normal

### Permission Denied

Make scripts executable:
```bash
chmod +x setup_environment.sh cluster_job.sh train.py evaluate.py
```

## Evaluation After Training

Once training completes:

```bash
# Activate environment
source venv/bin/activate

# Evaluate the model
python evaluate.py ./results/seed_0/dqn_model_final.zip --episodes 50
```

## Contact

If you encounter cluster-specific issues, contact your cluster administrator for:
- Available Python versions
- Module names for PyTorch/CUDA
- Job submission policies
- Storage quotas
