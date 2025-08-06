# PPO Race Car Implementation

## Overview
This implementation uses Proximal Policy Optimization (PPO) to train an AI agent for the race car challenge. PPO is a policy gradient method that's more stable and sample-efficient than DQN for this type of sequential decision-making task.

## Files Structure
- `ppo_model.py` - PPO actor-critic network and training agent
- `train_ppo.py` - PPO training loop with rollout collection
- `example_ppo.py` - PPO-based action selection for API
- `api_ppo.py` - FastAPI server using PPO model
- `dqn_environment.py` - Environment wrapper (reused from DQN)
- `training_simulator.py` - Headless game simulator (reused)

## Quick Start

### 1. Install Dependencies
```bash
pip install torch numpy matplotlib pygame fastapi uvicorn pydantic
```

### 2. Train PPO Model
```bash
# Train with default settings (1000 episodes, 2048 rollout length)
python train_ppo.py

# Custom training
python train_ppo.py --episodes 2000 --rollout 4096

# Resume training
python train_ppo.py --resume
```

### 3. Evaluate Trained Model
```bash
python train_ppo.py --evaluate --eval_episodes 10
```

### 4. Run API Server with PPO
```bash
python api_ppo.py
```

### 5. Test Locally with Pygame
```bash
python example_ppo.py
```

## PPO Algorithm Details

### Actor-Critic Architecture
```
Shared Layers: Input(12) → Dense(128) → Dense(128)
Actor Head: → Dense(64) → Softmax(5) [action probabilities]  
Critic Head: → Dense(64) → Linear(1) [state value]
```

### Key PPO Features
- **Clipped Policy Loss**: Prevents large policy updates
- **Generalized Advantage Estimation (GAE)**: Better advantage calculation
- **Rollout Collection**: Collects batches of experiences before updates
- **Multiple PPO Epochs**: 4 update epochs per rollout for sample efficiency

### Hyperparameters
- Learning rate: 3e-4
- Gamma (discount): 0.99
- PPO clip ratio: 0.2
- GAE lambda: 0.95
- Entropy coefficient: 0.01
- Value loss coefficient: 0.5
- PPO epochs per update: 4
- Rollout length: 2048 steps

### State Space (12 dimensions)
- 8 sensor readings (0-1000px → normalized to 0-1)
- Velocity X, Y (normalized)
- Distance traveled (normalized)
- Elapsed time (normalized)

### Action Space (5 discrete actions)
- NOTHING
- ACCELERATE
- DECELERATE  
- STEER_LEFT
- STEER_RIGHT

### Reward Function
- **Distance progress**: +0.01 per unit distance
- **Velocity bonus**: Small bonus for maintaining speed
- **Crash penalty**: -100 for crashes
- **Action penalty**: Small penalty for unnecessary actions
- **Time penalty**: Encourages efficient completion

## Training Process

PPO uses **rollout-based training**:

1. **Collect Rollout**: Gather 2048 steps of experience using current policy
2. **Compute Advantages**: Use GAE to estimate advantage of each action
3. **Policy Update**: Run 4 PPO epochs on the collected rollout
4. **Repeat**: Clear memory and collect new rollout

This is more sample-efficient than DQN's experience replay approach.

## Expected Performance

### Training Time
- **CPU (8+ cores)**: ~1-2 hours for 1000 episodes
- **GPU**: ~30-60 minutes for 1000 episodes
- **Convergence**: Usually within 500-1000 episodes

### Memory Usage
- **During Training**: ~500MB (rollout buffer)
- **Model Size**: ~1MB (actor-critic network)

### Sample Efficiency
PPO typically converges **2-3x faster** than DQN due to:
- On-policy learning (no stale data)
- Better exploration through policy entropy
- More stable gradient updates

## Monitoring Training

Training automatically saves:
- `ppo_training_progress.png` - 6 training plots (rewards, distances, losses, etc.)
- `models/ppo_racecar.pth` - Model checkpoints
- Periodic models: `ppo_model_TIMESTAMP_EPISODE.pth`

## Competition Usage

1. **Train**: `python train_ppo.py --episodes 1000`
2. **Start server**: `python api_ppo.py`
3. **Test endpoint**: Visit http://localhost:9052
4. **Submit**: Use the API endpoint for competition

## Advantages of PPO over DQN

### For This Racing Task:
1. **Better for sequential decisions**: Racing requires smooth action sequences
2. **More stable learning**: PPO's clipped updates prevent policy collapse  
3. **Sample efficiency**: On-policy learning with rollouts
4. **Natural exploration**: Policy entropy encourages diverse actions
5. **Value function**: Critic helps with credit assignment over long episodes

### When to Use Each:
- **PPO**: Complex sequential tasks, continuous control, stable training needed
- **DQN**: Simple decision problems, when off-policy learning is beneficial

## Troubleshooting

### Slow Training
- Reduce `--rollout` length (e.g., 1024 instead of 2048)
- Train on GPU if available
- Reduce `ppo_epochs` from 4 to 2

### Poor Performance
- Increase rollout length for more diverse experiences
- Adjust learning rate (try 1e-4 or 1e-3)
- Tune reward function in `dqn_environment.py`

### Memory Issues
- Reduce batch_size in PPOAgent constructor
- Use smaller rollout lengths

## Advanced Tuning

```python
# Custom PPO agent
agent = PPOAgent(
    lr=1e-4,           # Lower learning rate
    epsilon=0.1,       # Smaller clip ratio
    entropy_coef=0.02, # More exploration
    rollout_length=4096 # Longer rollouts
)
```