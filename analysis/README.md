# Analysis Scripts for Assignment Report

This directory contains scripts to generate visualizations and extract metrics for the assignment report.

## Files

- `generate_report_data.py` - Main script to generate all plots and extract hyperparameter results

## Generated Outputs

### Plots (saved to `../results/plots/`)
1. **cumulative_rewards_comparison.png** - 2x2 subplot showing reward curves for all 4 algorithms
2. **training_stability.png** - Loss curves and entropy analysis for training stability
3. **convergence_analysis.png** - Comparison of convergence speed across algorithms

### Data Files (saved to `../results/`)
- **hyperparameter_results.json** - Complete hyperparameter tuning results for all algorithms

## Usage

```bash
cd analysis
python3 generate_report_data.py
```

## Requirements

All dependencies are in the main `requirements.txt`:
- tensorboard
- matplotlib
- numpy
- stable-baselines3

## Notes

- The script automatically loads data from tensorboard logs in `../results/`
- Plots are saved at 300 DPI for high-quality report inclusion
- Hyperparameter results include 10 runs per algorithm with various configurations
