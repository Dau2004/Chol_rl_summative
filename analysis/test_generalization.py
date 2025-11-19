#!/usr/bin/env python3
"""
Test generalization of trained models on unseen initial states
"""
import os
import sys
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import WarehouseEnv

def test_model(model, env, n_episodes=100, model_name="Model"):
    """Test a model for n episodes and return statistics"""
    rewards = []
    episode_lengths = []
    successes = 0
    
    for episode in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        episode_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if isinstance(obs, tuple):
                obs = obs[0]
            if isinstance(terminated, np.ndarray):
                terminated = terminated[0]
            if isinstance(truncated, np.ndarray):
                truncated = truncated[0]
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Check for success (3 deliveries)
            if hasattr(env, 'envs'):
                deliveries = env.envs[0].deliveries_made
            else:
                deliveries = env.deliveries_made
            
            if deliveries >= 3:
                successes += 1
                break
        
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 20 == 0:
            print(f"  Episode {episode + 1}/{n_episodes} - Avg Reward: {np.mean(rewards[-20:]):.2f}")
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': (successes / n_episodes) * 100,
        'all_rewards': rewards
    }

def main():
    """Test all models for generalization"""
    print("🧪 Testing Model Generalization on Unseen States\n")
    
    results = {}
    
    # Test PPO
    print("Testing PPO...")
    try:
        env = DummyVecEnv([lambda: WarehouseEnv()])
        model = PPO.load("../models/pg/ppo_final", env=env)
        results['PPO'] = test_model(model, env, n_episodes=100, model_name="PPO")
        print(f"✅ PPO - Mean: {results['PPO']['mean_reward']:.2f} ± {results['PPO']['std_reward']:.2f}, Success: {results['PPO']['success_rate']:.1f}%\n")
    except Exception as e:
        print(f"❌ PPO test failed: {e}\n")
    
    # Test A2C
    print("Testing A2C...")
    try:
        env = DummyVecEnv([lambda: WarehouseEnv()])
        env = VecNormalize.load("../models/pg/a2c_vecnorm.pkl", env)
        env.training = False
        env.norm_reward = False
        model = A2C.load("../models/pg/a2c_final", env=env)
        results['A2C'] = test_model(model, env, n_episodes=100, model_name="A2C")
        print(f"✅ A2C - Mean: {results['A2C']['mean_reward']:.2f} ± {results['A2C']['std_reward']:.2f}, Success: {results['A2C']['success_rate']:.1f}%\n")
    except Exception as e:
        print(f"❌ A2C test failed: {e}\n")
    
    # Test DQN
    print("Testing DQN...")
    try:
        env = DummyVecEnv([lambda: WarehouseEnv()])
        model = DQN.load("../models/dqn/best_model", env=env)
        results['DQN'] = test_model(model, env, n_episodes=100, model_name="DQN")
        print(f"✅ DQN - Mean: {results['DQN']['mean_reward']:.2f} ± {results['DQN']['std_reward']:.2f}, Success: {results['DQN']['success_rate']:.1f}%\n")
    except Exception as e:
        print(f"❌ DQN test failed: {e}\n")
    
    # Print summary table
    print("\n" + "="*80)
    print("GENERALIZATION TEST RESULTS SUMMARY")
    print("="*80)
    print(f"{'Algorithm':<12} {'Mean Reward':<15} {'Std Dev':<12} {'Success Rate':<15} {'Avg Length':<12}")
    print("-"*80)
    
    for algo, stats in results.items():
        print(f"{algo:<12} {stats['mean_reward']:>8.2f}      {stats['std_reward']:>8.2f}    {stats['success_rate']:>6.1f}%         {stats['mean_length']:>6.1f}")
    
    print("="*80)
    
    # Save results
    import json
    with open('../results/generalization_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        save_results = {}
        for algo, stats in results.items():
            save_results[algo] = {
                'mean_reward': float(stats['mean_reward']),
                'std_reward': float(stats['std_reward']),
                'mean_length': float(stats['mean_length']),
                'success_rate': float(stats['success_rate']),
                'all_rewards': [float(r) for r in stats['all_rewards']]
            }
        json.dump(save_results, f, indent=2)
    
    print("\n✅ Results saved to ../results/generalization_results.json")

if __name__ == "__main__":
    main()
