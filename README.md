# Autonomous Warehouse Inventory Management - RL Implementation

## Project Overview
This project implements an autonomous warehouse robot using reinforcement learning to efficiently manage inventory operations. The robot navigates a dynamic warehouse environment, picks up items, avoids obstacles, and delivers them to designated locations while optimizing for efficiency and safety.

## Environment Description

### Mission Scenario
An autonomous robot operates in an 8x8 warehouse grid, tasked with:
- Navigating around dynamic obstacles (forklifts, workers)
- Picking up inventory items from shelves (3 items)
- Delivering items to designated drop-off zones (2 targets)
- Maximizing efficiency while maintaining safety

### Action Space
- **0**: Move North
- **1**: Move South  
- **2**: Move East
- **3**: Move West
- **4**: Pick up item
- **5**: Drop item

### Observation Space
Multi-plane encoding with 196 dimensions:
- Robot position (64 dimensions, one-hot)
- Item locations (64 dimensions, one-hot)
- Target locations (64 dimensions, one-hot)
- Scalar features (4 dimensions): carrying capacity, time remaining, items left, deliveries made

### Reward Structure
- **+50/+80**: Item pickup (progressive rewards)
- **+100/+120/+300**: Item delivery (progressive rewards)
- **+1000**: Mission complete bonus
- **-0.05**: Time step penalty
- **-10**: Collision with obstacle
- **+1.0**: Moving closer to objectives

## Algorithms Implemented

### 1. DQN (Value-Based)
- Deep Q-Network with experience replay
- Dueling architecture and target network
- Optimized for sample efficiency

### 2. PPO (Policy Gradient) 
- Proximal Policy Optimization
- Clipped surrogate objective
- **Best performer**: 1781.5 reward, 3/3 deliveries

### 3. A2C (Actor-Critic)
- Advantage Actor-Critic with GAE
- VecNormalize for stable training
- Longer rollout horizons (256 steps)

### 4. REINFORCE (Policy Gradient)
- Monte Carlo policy gradient
- Custom PyTorch implementation
- Baseline subtraction for variance reduction

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   └── rendering.py             # Visualization GUI components 
├── training/
│   ├── dqn_training.py          # DQN training script
│   └── pg_training.py           # Policy gradient training (PPO/A2C/REINFORCE)
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for best model demo
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Installation & Setup

```bash
# Create virtual environment
python3 -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train all models
python3 training/dqn_training.py
python3 training/pg_training.py

# Run best model demonstration
python3 main.py
```

## Usage

### 1. Train Models
```bash
# Train DQN (Value-Based)
python3 training/dqn_training.py

# Train Policy Gradient methods (PPO, A2C, REINFORCE)
python3 training/pg_training.py
```

### 2. Run Demonstration
```bash
# Run best performing model
python3 main.py

# Options:
# 1 - Run Best Model Demo (PPO)
# 2 - Compare All Models
# 3 - Exit
```

## Performance Results

| Algorithm | Avg Reward | Deliveries | Success Rate | Training Time |
|-----------|------------|------------|--------------|---------------|
| **PPO**   | **1781.5** | **3/3**    | **100%**     | ~10 min       |
| A2C       | 1781.5     | 3/3        | ~100%        | ~8 min        |
| REINFORCE | Variable   | 1-2/3      | ~60%         | ~15 min       |
| DQN       | -29.0      | 0/3        | 0%           | ~5 min        |

## Key Features

- **Custom Gymnasium Environment**: Fully compliant warehouse simulation
- **Enhanced Visualization**: Real-time rendering with pygame
- **Progressive Reward Shaping**: Encourages optimal task completion
- **Multi-Algorithm Comparison**: Objective performance evaluation
- **Modular Architecture**: Clean separation of concerns

## Technical Highlights

- **Observation Space Fix**: Multi-plane encoding ensures all items are visible
- **Reward Engineering**: Progressive rewards guide agents to complete all deliveries
- **Hyperparameter Optimization**: Extensive tuning for each algorithm
- **Vectorized Training**: Parallel environments for faster learning

## Best Model Performance

The **PPO (Proximal Policy Optimization)** algorithm achieved optimal performance:
- **Perfect Mission Completion**: 3/3 items delivered consistently
- **High Efficiency**: ~30 steps per episode
- **Maximum Reward**: 1781.5 points
- **100% Success Rate**: Reliable task completion

## Future Enhancements

- Multi-robot coordination
- Dynamic item priorities  
- Real-time obstacle avoidance
- Integration with warehouse management systems
- 3D visualization and advanced graphics

---

**Author**: Chol Daniel Deng 
**Course**: Reinforcement Learning Summative Assignment  
**Date**: 2025
