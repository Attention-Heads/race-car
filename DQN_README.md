# DQN Lane Switching Agent

🚗 **Deep Q-Network agent for intelligent lane switching in a racing environment**

This implementation provides a complete DQN-based lane switching controller that learns to maximize distance traveled while avoiding crashes through strategic lane changes.

## 🏗️ Architecture Overview

### Core Components

1. **DQN Agent** (`src/agents/dqn_agent.py`)
   - Main controller integrating all components
   - Handles action selection, experience collection, and training
   - Enforces safety mechanisms (3-second cooldown, boundary validation)

2. **State Processor** (`src/utils/state_processor.py`)
   - Temporal stacking with geometric backoff: [k, k-1, k-2, k-4, k-8, k-16, k-32]
   - Normalized 117-feature state vector
   - Lane position tracking and validation

3. **Neural Network** (`src/agents/neural_network.py`)
   - PyTorch implementation with Double DQN and Dueling DQN variants
   - Architecture: 117 → 256 → 128 → 64 → 3 (Q-values)
   - Target network with periodic updates

4. **Experience Replay** (`src/agents/replay_buffer.py`)
   - Standard and prioritized experience replay
   - Efficient circular buffer implementation
   - Proper sampling and batching

5. **Reward System** (`src/utils/reward_calculator.py`)
   - Comprehensive reward function balancing safety and performance
   - Adaptive weights based on training progress
   - Strategic bonuses for overtaking and positioning

## 📋 Features

### ✅ **Safety Mechanisms**
- **3-second cooldown** between lane changes (180 ticks at 60 FPS)
- **Boundary validation** prevents wall collisions  
- **Emergency override** for imminent collision scenarios
- **Cruise control integration** maintains safe following distances
- **Full 60-second episodes** matching competition specifications

### ✅ **Advanced DQN Techniques**
- **Double DQN** reduces overestimation bias
- **Target networks** for training stability
- **Experience replay** breaks correlation in training data
- **Gradient clipping** prevents exploding gradients
- **Epsilon decay** for exploration/exploitation balance

### ✅ **State Representation**
- **Temporal stacking** provides time-aware decision making
- **Smart normalization** handles sensor data, velocities, and positions
- **117-dimensional** feature vector with geometric backoff sampling

### ✅ **Reward Engineering**
- **Distance maximization** primary objective
- **Speed optimization** encourages faster driving
- **Safety penalties** for risky maneuvers
- **Strategic rewards** for overtaking and positioning
- **Adaptive scaling** based on training progress

## 🚀 Getting Started

### Prerequisites

```bash
# Install dependencies
uv pip install torch torchvision numpy matplotlib tensorboard pygame-ce fastapi uvicorn pydantic
```

### Quick Test

```bash
# Test all components
python test_dqn.py

# Run quick demo (5 episodes with game window)
python demo_dqn.py
```

### Training

```bash
# Full training (10,000 episodes)
python training/train_dqn.py

# Debug mode (100 episodes, verbose)
python training/train_dqn.py --config debug --verbose

# Fast training (1,000 episodes)
python training/train_dqn.py --config fast
```

### Evaluation

```bash
# Compare trained agent vs baseline
python training/evaluation.py --model models/checkpoints/dqn_latest.pth

# Evaluate baseline only
python training/evaluation.py --baseline-only
```

## 📊 Training Configuration

### Default Hyperparameters
```python
DQN_CONFIG = {
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 100000,
    'batch_size': 32,
    'target_update_frequency': 1000,
    'cooldown_ticks': 180  # 3 seconds at 60 FPS
}

# Environment matches game specifications exactly
ENV_CONFIG = {
    'screen_width': 1600,  # SCREEN_WIDTH from core.py
    'screen_height': 1200,  # SCREEN_HEIGHT from core.py  
    'lane_count': 5,  # LANE_COUNT from core.py
    'fps': 60,  # Game runs at 60 FPS
    'max_ticks': 3600  # 60 seconds * 60 FPS (MAX_TICKS)
}
```

### Training Phases
1. **Stage 1 (Episodes 1-3000)**: High exploration (ε: 1.0 → 0.1)
2. **Stage 2 (Episodes 3001-7000)**: Balanced (ε: 0.1 → 0.05)  
3. **Stage 3 (Episodes 7001-10000)**: Fine-tuning (ε: 0.05 → 0.01)

## 🎯 Action Space

The agent outputs one of three discrete actions:
- **0**: Change to left lane
- **1**: Change to right lane  
- **2**: Do nothing (cruise control active)

## 📈 State Space

**117-dimensional feature vector:**
- **112 features**: 16 sensors × 7 timesteps (temporal stacking)
- **5 additional features**: lane position, cooldown timer, velocity components, distance

**Normalization:**
- Sensors: [0, 1] (distance/1000)
- Lane position: [-1, 1] (centered on middle lane)
- Velocities: tanh normalization (handles outliers gracefully)
- Cooldown: [0, 1] (normalized by 180 ticks)

## 🏆 Performance Metrics

### Success Criteria
- **Distance improvement**: > 20% vs rule-based agent
- **Crash rate**: < 10%
- **Successful lane changes**: > 70%
- **Training stability**: Converging loss and Q-values

### Evaluation Metrics
- Average distance traveled
- Survival time
- Crash frequency
- Lane change efficiency
- Overtaking success rate

## 📁 Project Structure

```
race-car/
├── src/
│   ├── agents/
│   │   ├── dqn_agent.py          # Main DQN agent
│   │   ├── neural_network.py     # DQN network architectures
│   │   ├── replay_buffer.py      # Experience replay
│   │   └── training_env.py       # Training environment wrapper
│   ├── utils/
│   │   ├── state_processor.py    # State processing & temporal stacking
│   │   └── reward_calculator.py  # Reward function
│   ├── game/
│   │   ├── core.py              # Enhanced game loop with DQN support
│   │   └── agent.py             # Rule-based cruise controller
│   └── elements/                # Car, road, sensor components
├── training/
│   ├── train_dqn.py            # Main training script
│   ├── evaluation.py           # Performance evaluation
│   └── hyperparameters.py      # Configuration management
├── models/checkpoints/          # Saved model weights
├── logs/                       # Training logs
└── evaluation_results/         # Evaluation outputs
```

## 🔧 Configuration Options

### Training Modes
- **Default**: Full 10,000 episode training
- **Debug**: 100 episodes with visualization
- **Fast**: 1,000 episodes for quick testing

### Network Variants  
- **Standard DQN**: Basic implementation
- **Double DQN**: Reduced overestimation bias
- **Dueling DQN**: Separate value/advantage streams

### Replay Buffer Options
- **Standard**: Uniform sampling
- **Prioritized**: TD-error based sampling
- **Circular**: Memory-efficient implementation

## 📋 Training Logs

Training progress is automatically logged:
- **TensorBoard**: Real-time metrics visualization
- **JSON logs**: Episode statistics and hyperparameters  
- **Model checkpoints**: Periodic model saving
- **Training plots**: Reward/distance curves

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not available**: Training will use CPU (slower but functional)
2. **Memory errors**: Reduce buffer_size or batch_size
3. **Slow convergence**: Check reward function and exploration schedule
4. **Training instability**: Enable gradient clipping and reduce learning rate

### Debug Mode
```bash
python training/train_dqn.py --config debug --verbose
```
This shows the game window and detailed logging for debugging.

## 🎮 Integration with Existing Code

The DQN agent integrates seamlessly with the existing race car simulation:
- **Cruise control**: Rule-based agent handles longitudinal control
- **Lane switching**: DQN agent makes strategic lane change decisions
- **Safety**: Built-in collision avoidance and boundary checking
- **Compatibility**: Falls back gracefully if DQN components unavailable

## 🚀 Advanced Usage

### Custom Reward Functions
```python
from src.utils.reward_calculator import RewardCalculator

class CustomRewardCalculator(RewardCalculator):
    def calculate_reward(self, *args, **kwargs):
        # Your custom reward logic
        return super().calculate_reward(*args, **kwargs) + custom_bonus
```

### Hyperparameter Tuning
```python
from training.hyperparameters import get_default_config

config = get_default_config()
config['dqn']['learning_rate'] = 0.0005  # Custom learning rate
```

### Model Ensembles
```python
from src.agents.dqn_agent import EnsembleDQNAgent

ensemble_agent = EnsembleDQNAgent(num_agents=3)
```

## 📊 Expected Results

After successful training, expect:
- **20-50% distance improvement** over rule-based agent
- **Intelligent lane switching** based on traffic patterns
- **Smooth driving behavior** with strategic overtaking
- **Low crash rate** (< 10%) with safety-conscious decisions

---

**Ready to train your intelligent lane switching agent! 🏁**