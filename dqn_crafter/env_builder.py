
import os
from typing import Optional

import gym
import crafter
from gym.envs.registration import register
from stable_baselines3.common.monitor import Monitor

from .wrappers import ToUint8, ClipReward, CrafterInfoWrapper


def register_crafter_envs():
    """Register Crafter environments if not already registered."""
    envs_to_register = [
        ('CrafterReward-v1', {'reward': True}),
        ('CrafterNoReward-v1', {'reward': False}),
    ]
    
    for env_id, kwargs in envs_to_register:
        try:
            gym.spec(env_id)
        except:
            try:
                register(id=env_id, entry_point=crafter.Env, kwargs=kwargs)
                print(f"Registered {env_id}")
            except Exception as e:
                print(f"Could not register {env_id}: {e}")


def build_env(outdir: str, env_id: str = 'CrafterReward-v1', 
              clip_reward: bool = False, seed: Optional[int] = None):
    """
    Build a single Crafter environment with proper wrappers.
    
    Args:
        outdir: Directory for saving episode statistics
        env_id: Crafter environment ID
        clip_reward: Whether to clip rewards to [-1, 1]
        seed: Random seed
    
    Returns:
        Wrapped Gym environment
    """
    register_crafter_envs()
    
    # Fallback to CrafterReward-v1 if env_id not found
    try:
        gym.spec(env_id)
    except:
        print(f"Environment {env_id} not found, using CrafterReward-v1")
        env_id = 'CrafterReward-v1'
    
    env = gym.make(env_id)
    
    if seed is not None:
        env.seed(seed)
    
    # Convert to uint8 for CNN
    env = ToUint8(env)
    
    # Optional reward clipping
    if clip_reward:
        env = ClipReward(env, -1.0, 1.0)
    
    # Monitor for episode statistics
    env = Monitor(env, filename=os.path.join(outdir, 'monitor'))
    
    # Crafter Recorder for official metrics (save_stats must be True)
    # IMPORTANT: Place Recorder BEFORE CrafterInfoWrapper so achievements are captured
    env = crafter.Recorder(
        env,
        outdir,
        save_stats=True,  # CRITICAL: This creates stats.jsonl
        save_video=False,
        save_episode=False,
    )
    
    # Add achievement tracking - must be AFTER Recorder to capture its info
    env = CrafterInfoWrapper(env)
    
    return env
