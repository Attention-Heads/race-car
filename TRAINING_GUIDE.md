# DQN Training Guide

## 🚀 Quick Start

### 1. Test Setup
```bash
python demo_dqn.py                    # 5 episodes demo (~2 min)
python test_dqn.py                    # Verify all components work
```

### 2. Measure Baseline
```bash
python training/evaluation.py --baseline-only --episodes 20
```

### 3. Start Training
```bash
# Fast training (recommended first)
python training/train_dqn.py --config fast --episodes 1000

# Full training 
python training/train_dqn.py --config default
```

## 📊 Training Configurations

| Config | Episodes | Time | Purpose |
|--------|----------|------|---------|
| `debug` | 100 | 30 min | Testing/debugging with game window |
| `fast` | 1000 | 2-4 hours | Quick validation of approach |  
| `default` | 10000 | 10-20 hours | Full competitive training |

## 🎯 Training Options

```bash
# Basic training
python training/train_dqn.py

# With options
python training/train_dqn.py --config fast --episodes 2000 --verbose

# Custom episodes
python training/train_dqn.py --episodes 5000
```

## 📈 Monitor Progress

```bash
# Real-time logs
tail -f logs/training_log.jsonl

# Training plots created automatically at:
# logs/training_progress.png
```

## 🔧 Evaluate Models

```bash
# Test specific checkpoint
python training/evaluation.py --model models/checkpoints/dqn_episode_5000.pth

# Compare vs baseline (20 episodes each)
python training/evaluation.py --model models/checkpoints/dqn_latest.pth --episodes 20
```

## 📋 Expected Results

### Training Progress
- **Episodes 1-1000**: Learning basic controls, high crashes (~70%)
- **Episodes 1000-3000**: Crash rate drops to ~30%, basic lane switching  
- **Episodes 3000-6000**: Strategic decisions, distance improving
- **Episodes 6000-10000**: Consistent performance, crash rate <20%

### Success Metrics  
- **Distance**: 20-50% improvement over baseline (~800-1200)
- **Crash Rate**: <20% (vs baseline ~10-30%)
- **Lane Changes**: Purposeful, not random

## 🛠️ Troubleshooting

### High Crash Rate (>70% after 2000 episodes)
```python
# Edit training/hyperparameters.py
REWARD_CONFIG = {
    'crash_penalty': -2000.0,  # Increase penalty
    'survival_weight': 1.0,    # Increase survival reward
}
```

### Too Conservative (Never changes lanes)
```python
# Increase exploration
DQN_CONFIG = {
    'epsilon_decay': 0.9995,   # Slower decay
}
```

### Training Too Slow
```bash
# Use fast config with smaller buffer
python training/train_dqn.py --config fast
```

## 🎮 Deploy Trained Model

### 1. Start DQN API
```bash
# Ensure model exists
ls models/checkpoints/dqn_latest.pth

# Start API
python api_dqn.py
```

### 2. Test API
```bash
python test_api_compatibility.py
```

### 3. Connect to Validation
- API runs on `localhost:9052`
- Compatible with validation backend
- Automatically falls back to rule-based if model fails

## 📁 File Structure

```
logs/                          # Training logs and plots
├── training_log.jsonl        # Episode statistics
├── evaluation_log.jsonl     # Evaluation results  
└── training_progress.png     # Progress plots

models/checkpoints/           # Saved models
├── dqn_episode_1000.pth     # Periodic saves
├── dqn_latest.pth           # Most recent
└── dqn_final.pth            # End of training

training/                     # Training scripts
├── train_dqn.py             # Main training
├── evaluation.py            # Model evaluation
└── hyperparameters.py       # Configuration
```

## ⚡ Hardware Requirements

- **CPU**: ~15-25 hours full training
- **GPU**: ~6-12 hours full training  
- **RAM**: 8GB+ recommended
- **Storage**: ~1GB for logs/models

## 🎯 Recommended Workflow

1. **Test** (`demo_dqn.py`) → 2 min
2. **Baseline** (`evaluation.py --baseline-only`) → 30 min  
3. **Fast Training** (`--config fast`) → 2-4 hours
4. **Evaluate Fast Results** → 15 min
5. **Full Training** (`--config default`) → 10-20 hours
6. **Final Evaluation** → 30 min
7. **Deploy** (`api_dqn.py`) → Ready for validation

**Start with fast training to validate approach before full training!**