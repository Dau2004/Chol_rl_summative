#!/usr/bin/env python3
"""
Training script for DQN using Stable-Baselines3
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from environment.custom_env import WarehouseEnv

def train_dqn():
    """Train DQN agent"""
    print("🚀 Training DQN...")
    
    # Create directories
    os.makedirs("./models/dqn/", exist_ok=True)
    os.makedirs("./results/dqn_logs/", exist_ok=True)
    
    # Create vectorized environments
    env = make_vec_env(lambda: WarehouseEnv(), n_envs=4)
    eval_env = make_vec_env(lambda: WarehouseEnv(), n_envs=1)
    
    # Create DQN model with optimized hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200000,
        learning_starts=5000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=16,
        gradient_steps=8,
        target_update_interval=1000,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./results/dqn_tensorboard/"
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/dqn/",
        log_path="./results/dqn_logs/",
        eval_freq=20000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=400000,
        callback=eval_callback
    )
    
    # Save final model
    model.save("./models/dqn/final_model")
    print("💾 DQN model saved")
    
    return model

if __name__ == "__main__":
    train_dqn()