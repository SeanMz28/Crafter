
import math
from typing import Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CrafterMetricsCallback(BaseCallback):
    """Callback to track Crafter-specific metrics during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.achievements_per_episode = []
        self.all_achievement_keys = set()
    
    def _on_step(self) -> bool:
        # Check if we have episode info
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'episode' in info:
                ep_info = info['episode']
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
                
                achievements = {}
                if 'achievements' in ep_info:
                    achievements = ep_info['achievements']
                else:
                    for key, value in ep_info.items():
                        if key.startswith('achievement_'):
                            achievement_name = key.replace('achievement_', '')
                            achievements[achievement_name] = value
                
                if achievements:
                    self.achievements_per_episode.append(achievements)
                    self.all_achievement_keys.update(achievements.keys())
                
                # Log every 10 episodes
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    recent_lengths = self.episode_lengths[-10:]
                    print(f"\nStep {self.num_timesteps} | Episodes: {len(self.episode_rewards)}")
                    print(f"  Avg Reward (last 10): {np.mean(recent_rewards):.2f}")
                    print(f"  Avg Length (last 10): {np.mean(recent_lengths):.1f}")
        
        return True
    
    def get_achievement_stats(self) -> Dict:
        """Calculate achievement unlock rates."""
        if not self.achievements_per_episode:
            return {}
        
        unlock_counts = {key: 0 for key in self.all_achievement_keys}
        for achievements in self.achievements_per_episode:
            for key in self.all_achievement_keys:
                if key in achievements and achievements[key] > 0:
                    unlock_counts[key] += 1
        
        n_episodes = len(self.achievements_per_episode)
        unlock_rates = {key: count / n_episodes for key, count in unlock_counts.items()}
        
        # Calculate geometric mean
        eps = 1e-12
        if unlock_rates:
            geo_mean = math.exp(sum(math.log(rate + eps) for rate in unlock_rates.values()) / len(unlock_rates))
        else:
            geo_mean = 0.0
        
        return {
            'unlock_rates': unlock_rates,
            'geometric_mean': geo_mean,
            'total_episodes': n_episodes
        }
