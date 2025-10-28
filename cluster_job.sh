#!/bin/bash
#SBATCH --job-name=dqn_crafter
#SBATCH --output=logs/dqn_crafter_%j.out
#SBATCH --error=logs/dqn_crafter_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# Uncomment if you have GPU access:
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load required modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8  # If using GPU

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Set number of threads for better CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print Python and package info
echo "Python version:"
python --version
echo ""
echo "Installed packages:"
pip list | grep -E "(gym|stable-baselines3|torch|crafter)"
echo ""

# Run training
echo "Starting training..."
echo "=========================================="

python train.py \
    --timesteps 500000 \
    --seed 0 \
    --outdir ./results/seed_0 \
    --eval-episodes 50

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
