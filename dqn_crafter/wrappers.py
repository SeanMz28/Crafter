
import numpy as np
import gym
from gym.spaces import Box


class ToUint8(gym.ObservationWrapper):
    """Convert float32 [0,1] observations to uint8 [0,255] for CNN processing."""
    
    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        assert isinstance(space, Box), "ToUint8 expects Box observation space"
        self.observation_space = Box(low=0, high=255, shape=space.shape, dtype=np.uint8)
    
    def observation(self, obs):
        arr = np.asarray(obs)
        if arr.dtype == np.uint8:
            return arr
        return (arr * 255.0).clip(0, 255).astype(np.uint8)


class ClipReward(gym.RewardWrapper):
    """Optional reward clipping for DQN stability (Atari-style)."""
    
    def __init__(self, env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high
    
    def reward(self, reward):
        return np.clip(reward, self.low, self.high)


class CrafterInfoWrapper(gym.Wrapper):
    """
    Wrapper to track and expose achievement information in the info dict.
    This wrapper should be placed AFTER crafter.Recorder to capture achievements.
    
    Extracts achievements from the top-level info dict where Recorder places them.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_achievements = {}
        self.episode_length = 0
        self.episode_reward = 0.0
    
    def reset(self, **kwargs):
        self.episode_achievements = {}
        self.episode_length = 0
        self.episode_reward = 0.0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_length += 1
        self.episode_reward += reward
        
        # Crafter Recorder puts achievements in the top-level info dict
        # Extract and accumulate them throughout the episode
        if 'achievements' in info and isinstance(info['achievements'], dict):
            for achievement_name, value in info['achievements'].items():
                current_value = self.episode_achievements.get(achievement_name, 0)
                self.episode_achievements[achievement_name] = max(current_value, value)
        
        if done:
            # Create episode info dict for the callback
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length,
            }
            
            # Add achievements to episode info
            if self.episode_achievements:
                info['episode']['achievements'] = self.episode_achievements.copy()
                
                # Also add individual achievement keys for compatibility
                for achievement_name, value in self.episode_achievements.items():
                    info['episode'][f'achievement_{achievement_name}'] = value
        
        return obs, reward, done, info
