"""
train.py                          # Train with default config
    python train.py --timesteps 500000       # Custom training duration
    python train.py --seed 42                # Custom seed
    python train.py --outdir ./my_results    # Custom output directory
"""

import os
import sys
import json
import time
import argparse
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from dqn_crafter import (
    build_env,
    CrafterMetricsCallback,
    analyze_training_results,
    evaluate_agent,
    load_crafter_stats,
    DEFAULT_CONFIG,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train DQN agent on Crafter')
    
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory for logs and models')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Total training timesteps')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate for DQN')
    parser.add_argument('--buffer-size', type=int, default=None,
                        help='Replay buffer size')
    parser.add_argument('--clip-reward', action='store_true',
                        help='Enable reward clipping to [-1, 1]')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip evaluation after training')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to model to continue training')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Detect device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Build configuration from defaults and command line args
    config = DEFAULT_CONFIG.copy()
    
    if args.outdir:
        config['outdir'] = args.outdir
    if args.timesteps:
        config['total_timesteps'] = args.timesteps
    if args.seed is not None:
        config['seed'] = args.seed
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.buffer_size:
        config['buffer_size'] = args.buffer_size
    if args.clip_reward:
        config['clip_reward'] = True
    if args.eval_episodes:
        config['eval_episodes'] = args.eval_episodes
    
    print("="*60)
    print("DQN CRAFTER TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create output directory
    os.makedirs(config["outdir"], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config["outdir"], "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")
    
    # Create vectorized environment
    print("\nCreating environment...")
    vec_env = DummyVecEnv([
        lambda: build_env(
            config["outdir"], 
            config["env_id"], 
            config["clip_reward"],
            config["seed"]
        )
    ])
    
    # Apply image transformations
    vec_env = VecTransposeImage(vec_env)  # HWC -> CHW
    vec_env = VecFrameStack(vec_env, n_stack=config["frame_stack"], channels_order="first")
    
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # Create callback
    metrics_callback = CrafterMetricsCallback(verbose=1)
    
    # Policy network configuration - larger network for complex environment
    policy_kwargs = dict(
        net_arch=[512, 512],  # Larger hidden layers for complex state space
        normalize_images=True,  # Normalize pixel inputs to [0, 1]
    )
    
    # Initialize or load DQN agent
    if args.load_model:
        print(f"\nLoading model from: {args.load_model}")
        model = DQN.load(args.load_model, env=vec_env, device=device)
        print("Model loaded successfully!")
    else:
        print("\nInitializing DQN agent...")
        model = DQN(
            "CnnPolicy",
            vec_env,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            train_freq=config["train_freq"],
            gradient_steps=config["gradient_steps"],
            target_update_interval=config["target_update_interval"],
            exploration_initial_eps=config["exploration_initial_eps"],
            exploration_final_eps=config["exploration_final_eps"],
            exploration_fraction=config["exploration_fraction"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=config["seed"],
            device=device,
        )
    
    # Train the agent
    print(f"\nTraining for {config['total_timesteps']:,} timesteps...")
    print("="*60)
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=metrics_callback,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    elapsed_time = time.time() - start_time
    
    # Save the model
    model_path = os.path.join(config["outdir"], "dqn_model_final.zip")
    model.save(model_path)
    print(f"\n{'='*60}")
    print(f"Model saved to: {model_path}")
    print(f"Training time: {elapsed_time/60:.1f} minutes")
    print("="*60)
    
    vec_env.close()
    
    # Analyze training results
    print("\n" + "="*60)
    print("ANALYZING TRAINING RESULTS")
    print("="*60)
    achievement_stats = analyze_training_results(metrics_callback, config["outdir"])
    
    # Load and display Crafter Recorder stats
    print("\n" + "="*60)
    print("LOADING CRAFTER RECORDER STATS")
    print("="*60)
    crafter_stats = load_crafter_stats(config["outdir"])
    
    # Evaluate the trained model
    if not args.no_eval:
        print("\n" + "="*60)
        print("EVALUATING TRAINED AGENT")
        print("="*60)
        eval_results = evaluate_agent(
            model_path, 
            config, 
            n_eval_episodes=config["eval_episodes"]
        )
    else:
        print("\n⚠️  Skipping evaluation (--no-eval flag set)")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {config['outdir']}")
    print("\nFiles created:")
    print(f"  - {model_path}")
    print(f"  - {config_path}")
    print(f"  - {os.path.join(config['outdir'], 'stats.jsonl')}")
    print(f"  - {os.path.join(config['outdir'], 'training_curves.png')}")
    if not args.no_eval:
        print(f"  - {os.path.join(config['outdir'], 'eval', 'evaluation_results.json')}")
    print()


if __name__ == "__main__":
    main()
