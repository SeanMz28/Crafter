# DQN Crafter Baseline

Modular implementation of a DQN agent for the Crafter environment with proper achievement tracking and evaluation metrics.

## Project Structure

```
.
├── dqn_crafter/              # Main package
│   ├── __init__.py           # Package exports
│   ├── wrappers.py           # Environment wrappers (ToUint8, ClipReward, CrafterInfoWrapper)
│   ├── env_builder.py        # Environment builder and registration
│   ├── callbacks.py          # Training callbacks (CrafterMetricsCallback)
│   ├── evaluation.py         # Evaluation and analysis functions
│   └── config.py             # Default configuration
├── train.py                  # Main training script
├── evaluate.py               # Evaluation script for trained models
└── logdir/                   # Training logs and models (created during training)
```

## Installation

Ensure you have the required dependencies:

```bash
pip install gym stable-baselines3 numpy matplotlib
pip install git+https://github.com/danijar/crafter.git
```

## Usage

### Training

Train with default configuration:
```bash
python train.py
```

Train with custom parameters:
```bash
python train.py --timesteps 500000 --seed 42 --outdir ./my_results
```

Available options:
- `--outdir`: Output directory for logs and models
- `--timesteps`: Total training timesteps
- `--seed`: Random seed
- `--learning-rate`: Learning rate for DQN
- `--buffer-size`: Replay buffer size
- `--clip-reward`: Enable reward clipping to [-1, 1]
- `--eval-episodes`: Number of evaluation episodes
- `--no-eval`: Skip evaluation after training
- `--load-model`: Path to model to continue training

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py ./logdir/dqn_baseline/seed_0/dqn_model_final.zip
```

With custom number of episodes:
```bash
python evaluate.py path/to/model.zip --episodes 20
```

## Configuration

Default configuration is in `dqn_crafter/config.py`:

```python
DEFAULT_CONFIG = {
    "outdir": "./logdir/dqn_baseline/seed_0",
    "env_id": "CrafterReward-v1",
    "clip_reward": False,
    "frame_stack": 4,
    "seed": 0,
    "total_timesteps": 20_000,
    "learning_rate": 1e-4,
    "buffer_size": 100_000,
    "learning_starts": 50_000,
    "batch_size": 32,
    "gamma": 0.99,
    # ... and more
}
```

You can override these via command-line arguments or by modifying the config file.

## Key Features

- **Proper Achievement Tracking**: Fixed wrapper ordering to capture achievements during training
- **Comprehensive Metrics**: Episode rewards, survival time, and achievement unlock rates
- **Geometric Mean Calculation**: Official Crafter metric for overall performance
- **Modular Design**: Easy to extend and modify components
- **Visualization**: Automatic generation of training curves
- **Evaluation**: Separate evaluation script for testing trained models

## Output Files

After training, the following files are created in the output directory:

- `dqn_model_final.zip`: Trained model weights
- `config.json`: Configuration used for training
- `stats.jsonl`: Episode statistics from Crafter Recorder
- `monitor.monitor.csv`: Episode data from Stable Baselines3 Monitor
- `training_curves.png`: Visualization of training progress
- `eval/evaluation_results.json`: Evaluation metrics (if evaluation is run)

## Assignment Notes

This implementation addresses the following assignment requirements:

1. ✅ **Base Algorithm**: DQN with CNN policy
2. ✅ **Environment**: CrafterReward-v1 with standard rewards
3. ✅ **Metrics**: Achievement unlock rates, survival time, cumulative reward
4. ✅ **Geometric Mean**: Official Crafter scoring metric
5. ✅ **Proper Logging**: Comprehensive tracking of all metrics

For full training, consider using `--timesteps 500000` or higher as suggested in the assignment.

## Module Details

### `dqn_crafter/wrappers.py`
- `ToUint8`: Converts float observations to uint8 for CNN processing
- `ClipReward`: Optional reward clipping (Atari-style)
- `CrafterInfoWrapper`: Tracks achievements from Crafter Recorder

### `dqn_crafter/env_builder.py`
- `register_crafter_envs()`: Registers Crafter environments with Gym
- `build_env()`: Creates wrapped environment with proper ordering

### `dqn_crafter/callbacks.py`
- `CrafterMetricsCallback`: Tracks metrics during training and calculates achievement statistics

### `dqn_crafter/evaluation.py`
- `evaluate_agent()`: Evaluates trained model over multiple episodes
- `analyze_training_results()`: Analyzes and visualizes training data
- `load_crafter_stats()`: Loads and displays stats from Crafter Recorder

## Troubleshooting

**No achievements captured during training:**
- Ensure wrappers are in correct order (Recorder before CrafterInfoWrapper)
- Increase `total_timesteps` to allow more episodes to complete
- Check that `save_stats=True` in Crafter Recorder

**Low geometric mean:**
- This is expected early in training when most achievements have 0% unlock rate
- Train for longer (`--timesteps 500000` or more)
- The geometric mean approaches zero when many achievements are never unlocked

**Memory issues:**
- Reduce `buffer_size` 
- Reduce `frame_stack` (but this may hurt performance)
- Train on a machine with more RAM
