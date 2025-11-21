# Vaccine Cold-Chain Storage Robot - South Sudan Health Facilities

## Project Overview
This project implements an autonomous vaccine storage robot using reinforcement learning to efficiently manage vaccine distribution in South Sudan's healthcare system. The robot navigates a pharmaceutical cold-chain warehouse, picks up vaccine boxes, avoids dynamic obstacles (staff and equipment), and delivers essential medicines to health facilities across the country, optimizing for speed and safety in a resource-constrained environment.

## Environment Description

### Mission Scenario
An autonomous vaccine storage robot operates in an 8x8 cold-chain warehouse grid in South Sudan, tasked with:
- Navigating around dynamic obstacles (warehouse staff, loading workers, moving trolleys)
- Picking up vaccine boxes from cold storage shelves (3 vaccine boxes)
- Delivering vaccines to designated health facility dispatch zones (2 target locations)
- Maximizing efficiency while maintaining cold-chain integrity and safety
- Addressing medicine shortages and distribution delays common in South Sudan's healthcare system

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
- **+50/+80**: Vaccine box pickup (progressive rewards)
- **+100/+120/+300**: Vaccine delivery to health facilities (progressive rewards)
- **+1000**: Mission complete bonus (all vaccines delivered)
- **-0.05**: Time step penalty (cold-chain timer)
- **-10**: Collision with obstacle (staff/equipment)
- **+1.0**: Moving closer to objectives (efficient routing)

## Algorithms Implemented

### 1. DQN (Value-Based)
- Deep Q-Network with experience replay
- Dueling architecture and target network
- Optimized for sample efficiency

### 2. PPO (Policy Gradient) 
- Proximal Policy Optimization
- Clipped surrogate objective
- **Best performer**: 1781.5 reward, 3/3 vaccine deliveries to health facilities

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
│   ├── custom_env.py            # Custom Gymnasium cold-chain warehouse environment
│   └── rendering.py             # Vaccine robot visualization GUI
├── training/
│   ├── dqn_training.py          # DQN training script
│   └── pg_training.py           # Policy gradient training (PPO/A2C/REINFORCE)
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for vaccine robot demo
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
# Run best performing vaccine robot model
python3 main.py

# Options:
# 1 - Run Best Model Demo (PPO - Vaccine Delivery)
# 2 - Compare All Models
# 3 - Exit
```

## Performance Results

| Algorithm | Avg Reward | Vaccine Deliveries | Success Rate | Training Time |
|-----------|------------|-------------------|--------------|---------------|
| **PPO**   | **1781.5** | **3/3**           | **100%**     | ~10 min       |
| A2C       | 1781.5     | 3/3               | ~100%        | ~8 min        |
| REINFORCE | Variable   | 1-2/3             | ~60%         | ~15 min       |
| DQN       | -29.0      | 0/3               | 0%           | ~5 min        |

## Key Features

- **Custom Gymnasium Environment**: Fully compliant cold-chain warehouse simulation
- **Enhanced Visualization**: Real-time rendering with pygame (vaccine robot, health facilities)
- **Progressive Reward Shaping**: Encourages optimal vaccine delivery completion
- **Multi-Algorithm Comparison**: Objective performance evaluation
- **Modular Architecture**: Clean separation of concerns
- **Real-World Context**: Addresses South Sudan's medicine distribution challenges

## Technical Highlights

- **Observation Space Fix**: Multi-plane encoding ensures all items are visible
- **Reward Engineering**: Progressive rewards guide agents to complete all deliveries
- **Hyperparameter Optimization**: Extensive tuning for each algorithm
- **Vectorized Training**: Parallel environments for faster learning

## Assignment Requirements Fulfilled

✅ **Four RL Algorithms**: DQN, PPO, A2C, REINFORCE  
✅ **Stable-Baselines3**: Professional RL library implementation  
✅ **Same Environment**: All algorithms use identical cold-chain warehouse environment  
✅ **Objective Comparison**: Standardized evaluation metrics  
✅ **Custom Environment**: Gymnasium-compliant implementation  
✅ **Visualization**: Real-time GUI demonstration with South Sudan health context  
✅ **Real-World Application**: Addresses vaccine distribution challenges in South Sudan
## Best Model Performance

The **PPO (Proximal Policy Optimization)** algorithm achieved optimal performance:
- **Perfect Mission Completion**: 3/3 vaccine boxes delivered to health facilities consistently
- **High Efficiency**: ~30 steps per episode (critical for cold-chain integrity)
- **Maximum Reward**: 1781.5 points
- **100% Success Rate**: Reliable vaccine delivery to South Sudan health facilities

## Future Enhancements

- Multi-robot coordination for larger facilities
- Dynamic vaccine priority based on urgency (outbreak response)
- Real-time obstacle avoidance with sensor integration
- Integration with South Sudan's health management information systems
- Temperature monitoring and cold-chain breach detection
- 3D visualization and advanced graphics

---

**Author**: Chol Daniel Deng  
**Course**: Reinforcement Learning Summative Assignment  
**Date**: 2025
