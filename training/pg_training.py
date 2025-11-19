#!/usr/bin/env python3
"""
Training script for Policy Gradient methods (PPO, A2C, REINFORCE) using Stable-Baselines3
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize
from environment.custom_env import WarehouseEnv

class REINFORCEPolicy(nn.Module):
    """Custom REINFORCE policy network"""
    def __init__(self, state_size, action_size, hidden_size=256):
        super(REINFORCEPolicy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

def train_ppo():
    """Train PPO agent"""
    print("🚀 Training PPO...")
    
    os.makedirs("./models/pg/", exist_ok=True)
    os.makedirs("./results/ppo_logs/", exist_ok=True)
    
    env = make_vec_env(lambda: WarehouseEnv(), n_envs=4)
    eval_env = make_vec_env(lambda: WarehouseEnv(), n_envs=1)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./results/ppo_tensorboard/"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/pg/",
        log_path="./results/ppo_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    model.learn(
        total_timesteps=600000,
        callback=eval_callback
    )
    
    model.save("./models/pg/ppo_final")
    print("💾 PPO model saved")
    return model

def train_a2c():
    """Train A2C agent"""
    print("🚀 Training A2C...")
    
    os.makedirs("./models/pg/", exist_ok=True)
    os.makedirs("./results/a2c_logs/", exist_ok=True)
    
    env = make_vec_env(lambda: WarehouseEnv(), n_envs=8)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    eval_env = make_vec_env(lambda: WarehouseEnv(), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./results/a2c_tensorboard/"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/pg/",
        log_path="./results/a2c_logs/",
        eval_freq=25000,
        deterministic=True,
        render=False
    )
    
    model.learn(
        total_timesteps=600000,
        callback=eval_callback
    )
    
    model.save("./models/pg/a2c_final")
    env.save("./models/pg/a2c_vecnorm.pkl")
    print("💾 A2C model saved")
    return model

def train_reinforce():
    """Train REINFORCE agent"""
    print("🚀 Training REINFORCE...")
    
    os.makedirs("./models/pg/", exist_ok=True)
    
    env = WarehouseEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    policy = REINFORCEPolicy(state_size, action_size, hidden_size=256)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    episodes = 2000
    gamma = 0.99
    
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        for step in range(200):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            
            log_probs.append(dist.log_prob(action))
            
            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient update
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {sum(rewards):.2f}")
    
    torch.save(policy.state_dict(), "./models/pg/reinforce_final.pth")
    print("💾 REINFORCE model saved")
    return policy

def main():
    """Train all policy gradient methods"""
    print("🎯 Training Policy Gradient Methods")
    
    # Train PPO (best performer)
    ppo_model = train_ppo()
    
    # Train A2C
    a2c_model = train_a2c()
    
    # Train REINFORCE
    reinforce_model = train_reinforce()
    
    print("✅ All policy gradient models trained successfully!")

if __name__ == "__main__":
    main()