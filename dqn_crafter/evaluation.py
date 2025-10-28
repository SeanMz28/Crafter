
import os
import json
import math
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from .env_builder import build_env
from .callbacks import CrafterMetricsCallback


def evaluate_agent(model_path: str, config: Dict, n_eval_episodes: int = 10):
    """Evaluate a trained agent and return detailed metrics."""
    
    print(f"\nEvaluating agent over {n_eval_episodes} episodes...")
    
    # Create evaluation directory
    eval_dir = os.path.join(os.path.dirname(model_path), 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Build evaluation environment
    eval_env = DummyVecEnv([
        lambda: build_env(eval_dir, config["env_id"], clip_reward=False, seed=config["seed"]+1000)
    ])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=config["frame_stack"], channels_order="first")
    
    # Load model
    model = DQN.load(model_path)
    
    # Run evaluation
    episode_rewards = []
    episode_lengths = []
    episode_achievements = []
    
    for ep in range(n_eval_episodes):
        obs = eval_env.reset()
        done = False
        ep_reward = 0
        ep_length = 0
        ep_achievements = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward[0]
            ep_length += 1
            
            if done[0] and 'episode' in info[0]:
                ep_achievements = info[0]['episode'].get('achievements', {})
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        if ep_achievements:
            episode_achievements.append(ep_achievements)
        
        print(f"  Episode {ep+1}/{n_eval_episodes}: Reward={ep_reward:.2f}, Length={ep_length}")
    
    eval_env.close()
    
    # Calculate statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }
    
    # Achievement statistics
    if episode_achievements:
        all_keys = set()
        for achievements in episode_achievements:
            all_keys.update(achievements.keys())
        
        unlock_counts = {key: 0 for key in all_keys}
        for achievements in episode_achievements:
            for key in all_keys:
                if key in achievements and achievements[key] > 0:
                    unlock_counts[key] += 1
        
        unlock_rates = {key: count / n_eval_episodes for key, count in unlock_counts.items()}
        
        eps = 1e-12
        geo_mean = math.exp(sum(math.log(rate + eps) for rate in unlock_rates.values()) / len(unlock_rates))
        
        results['achievement_unlock_rates'] = unlock_rates
        results['geometric_mean'] = geo_mean
    
    # Save JSON results
    results_path = os.path.join(eval_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {
            k: (v.tolist() if isinstance(v, np.ndarray) else 
                float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in results.items()
        }
        json.dump(json_results, f, indent=2)
    
    # Create comprehensive text report
    txt_report_path = os.path.join(eval_dir, 'evaluation_report.txt')
    with open(txt_report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("EVALUATION REPORT - DQN CRAFTER BASELINE\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {os.path.basename(model_path)}\n")
        f.write(f"Number of Episodes: {n_eval_episodes}\n")
        f.write(f"Environment: {config['env_id']}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        # Core metrics
        f.write(f"Average Cumulative Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Average Survival Time:      {results['mean_length']:.1f} ± {results['std_length']:.1f} steps\n")
        
        if 'geometric_mean' in results:
            f.write(f"Geometric Mean Score (GMS): {results['geometric_mean']:.4f}\n")
        
        f.write(f"\nMin Reward:  {np.min(episode_rewards):.2f}\n")
        f.write(f"Max Reward:  {np.max(episode_rewards):.2f}\n")
        f.write(f"Min Survival: {np.min(episode_lengths)} steps\n")
        f.write(f"Max Survival: {np.max(episode_lengths)} steps\n")
        
        # Achievement unlock rates
        if 'achievement_unlock_rates' in results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("ACHIEVEMENT UNLOCK RATES\n")
            f.write("=" * 70 + "\n\n")
            
            unlock_rates = results['achievement_unlock_rates']
            sorted_achievements = sorted(unlock_rates.items(), key=lambda x: x[1], reverse=True)
            
            f.write(f"{'Achievement':<30} {'Unlock Rate':>15} {'Episodes':>15}\n")
            f.write("-" * 70 + "\n")
            
            for achievement, rate in sorted_achievements:
                episodes_unlocked = int(rate * n_eval_episodes)
                f.write(f"{achievement:<30} {rate:>14.3f} {episodes_unlocked:>15}/{n_eval_episodes}\n")
            
            f.write("\n" + "-" * 70 + "\n")
            f.write(f"{'Geometric Mean Score (GMS)':<30} {results['geometric_mean']:>14.4f}\n")
        
        # Per-episode details
        f.write("\n" + "=" * 70 + "\n")
        f.write("PER-EPISODE RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Episode':<10} {'Reward':>12} {'Survival':>12}\n")
        f.write("-" * 70 + "\n")
        
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths), 1):
            f.write(f"{i:<10} {reward:>12.2f} {length:>12}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    # Print summary to console
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Average Cumulative Reward:  {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Average Survival Time:      {results['mean_length']:.1f} ± {results['std_length']:.1f} steps")
    if 'geometric_mean' in results:
        print(f"Geometric Mean Score (GMS): {results['geometric_mean']:.4f}")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  JSON: {results_path}")
    print(f"  TXT:  {txt_report_path}")
    
    return results


def analyze_training_results(callback: CrafterMetricsCallback, outdir: str):
    """Analyze and visualize training results."""
    
    print("="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    # Episode statistics
    if callback.episode_rewards:
        print(f"\nTotal Episodes Completed: {len(callback.episode_rewards)}")
        print(f"Average Episode Reward: {np.mean(callback.episode_rewards):.2f} ± {np.std(callback.episode_rewards):.2f}")
        print(f"Average Survival Time: {np.mean(callback.episode_lengths):.1f} ± {np.std(callback.episode_lengths):.1f}")
        print(f"Max Episode Reward: {np.max(callback.episode_rewards):.2f}")
        print(f"Max Survival Time: {np.max(callback.episode_lengths)}")
    
    # Achievement statistics
    achievement_stats = callback.get_achievement_stats()
    if achievement_stats and achievement_stats['unlock_rates']:
        print(f"\n{'='*60}")
        print("ACHIEVEMENT UNLOCK RATES")
        print("="*60)
        
        unlock_rates = achievement_stats['unlock_rates']
        sorted_achievements = sorted(unlock_rates.items(), key=lambda x: x[1], reverse=True)
        
        for achievement, rate in sorted_achievements:
            bar = '█' * int(rate * 50)
            print(f"{achievement:25s} | {bar:50s} {rate:.3f}")
        
        print(f"\n{'='*60}")
        print(f"Geometric Mean Unlock Rate: {achievement_stats['geometric_mean']:.4f}")
        print(f"Total Episodes Analyzed: {achievement_stats['total_episodes']}")
        print("="*60)
    else:
        print("\n⚠️  No achievement data collected during training.")
        print("   This may indicate the agent didn't complete enough episodes.")
    
    # Plot training curves
    if callback.episode_rewards:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Episode rewards
        axes[0].plot(callback.episode_rewards, alpha=0.3, label='Raw')
        if len(callback.episode_rewards) > 10:
            window = min(50, len(callback.episode_rewards) // 10)
            smoothed = np.convolve(callback.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(callback.episode_rewards)), 
                        smoothed, label=f'Smoothed ({window} ep)', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Progress: Episode Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Survival time
        axes[1].plot(callback.episode_lengths, alpha=0.3, label='Raw')
        if len(callback.episode_lengths) > 10:
            window = min(50, len(callback.episode_lengths) // 10)
            smoothed = np.convolve(callback.episode_lengths, 
                                   np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(callback.episode_lengths)), 
                        smoothed, label=f'Smoothed ({window} ep)', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Survival Time (steps)')
        axes[1].set_title('Training Progress: Survival Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(outdir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close figure instead of showing it
        print(f"\nTraining curves saved to: {plot_path}")
    
    return achievement_stats


def load_crafter_stats(outdir: str):
    """Load and analyze stats from Crafter Recorder (stats.jsonl format)."""
    
    # Try both .jsonl and .json extensions
    stats_paths = [
        os.path.join(outdir, 'stats.jsonl'),
        os.path.join(outdir, 'stats.json'),
    ]
    
    stats_path = None
    for path in stats_paths:
        if os.path.exists(path):
            stats_path = path
            break
    
    if not stats_path:
        print(f"⚠️  No stats file found in {outdir}")
        print("   Checked: stats.jsonl, stats.json")
        print("   This file is created by crafter.Recorder when episodes complete.")
        return None
    
    print(f"Loading Crafter stats from: {stats_path}")
    
    episodes = []
    with open(stats_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
    
    if not episodes:
        print("⚠️  Stats file is empty. No episodes were recorded.")
        return None
    
    print(f"\nLoaded {len(episodes)} episodes from Crafter Recorder")
    
    # Extract episode data
    lengths = [ep.get('length', 0) for ep in episodes]
    rewards = [ep.get('reward', 0.0) for ep in episodes]
    
    # Extract achievements
    all_achievement_keys = set()
    for ep in episodes:
        ach = ep.get('achievements')
        for k in ep.keys():
            if k.startswith("achievement_"):
                all_achievement_keys.add(k[len("achievement_"):])  # store without prefix

    print(f"\n{'='*60}")
    print("CRAFTER RECORDER STATISTICS")
    print("="*60)
    print(f"Total Episodes: {len(episodes)}")
    print(f"Average Survival: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} steps")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Max Survival: {np.max(lengths)} steps")
    print(f"Max Reward: {np.max(rewards):.2f}")

    if all_achievement_keys:
        print(f"\n{'='*60}")
        print("ACHIEVEMENT STATISTICS (from Crafter Recorder)")
        print("="*60)

        # Count per-episode unlocks (value > 0) across both schemas
        unlock_counts = {key: 0 for key in all_achievement_keys}
        for ep in episodes:
            nested = ep.get('achievements') if isinstance(ep.get('achievements'), dict) else {}
            for key in all_achievement_keys:
                # Prefer nested if present, else fall back to flat
                val = nested.get(key, ep.get(f"achievement_{key}", 0))
                if float(val) > 0:
                    unlock_counts[key] += 1

        unlock_rates = {key: unlock_counts[key] / len(episodes) for key in all_achievement_keys}
        sorted_achievements = sorted(unlock_rates.items(), key=lambda x: x[1], reverse=True)

        for achievement, rate in sorted_achievements:
            bar = '█' * int(rate * 50)
            print(f"{achievement:25s} | {bar:50s} {rate:.3f}")

        # Geometric mean of unlock rates (with small epsilon to avoid log(0))
        eps = 1e-12
        geo_mean = math.exp(sum(math.log(rate + eps) for rate in unlock_rates.values()) / len(unlock_rates))

        print(f"\n{'='*60}")
        print(f"Geometric Mean Unlock Rate: {geo_mean:.4f}")
        print("="*60)

        return {
            'episodes': episodes,
            'lengths': lengths,
            'rewards': rewards,
            'unlock_rates': unlock_rates,
            'geometric_mean': geo_mean
        }
    else:
        print("\n⚠️  No achievement data found in stats file.")
        return {
            'episodes': episodes,
            'lengths': lengths,
            'rewards': rewards,
        }
