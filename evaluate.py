"""
    python evaluate.py ./logdir/dqn_baseline/seed_0/dqn_model_final.zip
    python evaluate.py path/to/model.zip --episodes 20
"""

import os
import sys
import json
import argparse

from dqn_crafter import evaluate_agent, DEFAULT_CONFIG


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained DQN agent on Crafter')
    
    parser.add_argument('model_path', type=str,
                        help='Path to trained model (.zip file)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config.json (optional, will try to find automatically)')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        sys.exit(1)
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        # Try to find config in same directory as model
        model_dir = os.path.dirname(args.model_path)
        config_path = os.path.join(model_dir, 'config.json')
    
    if os.path.exists(config_path):
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print("Warning: config.json not found, using default configuration")
        config = DEFAULT_CONFIG.copy()
    
    # Override eval episodes
    config['eval_episodes'] = args.episodes
    
    print("="*60)
    print("EVALUATING DQN MODEL")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print("="*60)
    
    # Run evaluation
    results = evaluate_agent(args.model_path, config, n_eval_episodes=args.episodes)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
