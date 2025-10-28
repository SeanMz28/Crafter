# DQN Crafter Training

This directory contains a modular implementation of a DQN agent for the Crafter environment, refactored from the original Jupyter notebook.

## Project Structure

```
Crafter/
├── dqn_crafter/              # Package directory
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration settings
│   ├── wrappers.py          # Environment wrappers
│   ├── env_builder.py       # Environment builder
│   └── callbacks.py         # Training callbacks
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
└── DQN_Crafter_Base_v4.ipynb  # Original notebook
```

## Installation

```bash
# Install required packages
pip install gym==0.21.0
pip install stable-baselines3
pip install git+https://github.com/danijar/crafter.git
pip install matplotlib numpy
```

## Usage

### Training

Basic training:
```bash
python train.py
```

Training with custom parameters:
```bash
python train.py --timesteps 500000 --seed 42 --outdir ./results/run1
```

Available arguments:
- `--timesteps`: Total training timesteps (default: 20,000)
- `--seed`: Random seed (default: 0)
- `--outdir`: Output directory (default: ./logdir/dqn_baseline/seed_0)
- `--eval-episodes`: Number of evaluation episodes (default: 10)
- `--learning-rate`: Learning rate (default: 1e-4)

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --model-path ./logdir/dqn_baseline/seed_0/dqn_crafter_final.zip --episodes 20
```

Available arguments:
- `--model-path`: Path to trained model (required)
- `--episodes`: Number of evaluation episodes (default: 10)
- `--outdir`: Output directory for evaluation (default: ./logdir/eval)
- `--seed`: Random seed (default: 42)
- `--render`: Render the environment during evaluation

## Configuration

Default configuration is in `dqn_crafter/config.py`. Key parameters:

- **Environment**: CrafterReward-v1
- **Total timesteps**: 20,000 (increase to 500,000+ for better results)
- **Learning rate**: 1e-4
- **Buffer size**: 100,000
- **Learning starts**: 50,000
- **Batch size**: 32
- **Frame stack**: 4

## Monitoring

Training progress is logged to:
- Console output (every 10 episodes)
- TensorBoard logs (`<outdir>/tensorboard/`)
- Episode statistics (`<outdir>/monitor.csv`)
- Crafter achievements (`<outdir>/stats.jsonl`)

View TensorBoard logs:
```bash
tensorboard --logdir ./logdir/dqn_baseline/seed_0/tensorboard
```

## Metrics

The implementation tracks:
- **Episode rewards**: Cumulative reward per episode
- **Episode lengths**: Survival time
- **Achievements**: Unlock rates for all Crafter achievements
- **Geometric mean**: Overall achievement score

## Achievements Tracking

The fixed `CrafterInfoWrapper` properly captures achievements from the Crafter Recorder:
1. Extracts achievements from `info['achievements']` at each step
2. Accumulates them throughout the episode
3. Adds them to `info['episode']` for callback access

## Notes

- The notebook version contained duplicated output - this has been cleaned up in the Python scripts
- For proper training, increase `total_timesteps` to at least 500,000 as suggested in the assignment
- Achievement tracking is now working correctly and will be displayed during training and evaluation
- The environment properly handles frame stacking and image preprocessing for CNN-based DQN

## Original Notebook

The original Jupyter notebook (`DQN_Crafter_Base_v4.ipynb`) is preserved in the repository for reference.
