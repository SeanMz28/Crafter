
DEFAULT_CONFIG = {
    # Paths & environment
    "outdir": "./logdir/dqn_baseline/seed_0",
    "env_id": "CrafterReward-v1",  # Use standard reward environment
    "clip_reward": True,  # Set to True for Atari-style clipping
    
    # Frame stacking (handled at Vec level)
    "frame_stack": 4,
    
    # Seed & training steps
    "seed": 36,
    "total_timesteps": 1_000_000, 
    
    
    "learning_rate": 2.5e-4,  
    "buffer_size": 1_000_000,  
    "learning_starts": 50_000,  
    "batch_size": 64, 
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 10_000, 
    "exploration_fraction": 0.1,  
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.01,  
    
    # Evaluation
    "eval_episodes": 50,  
}
