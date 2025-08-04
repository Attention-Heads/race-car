# Race Car PPO Environment

This directory contains a custom Gymnasium environment for training a race car agent using PPO (Proximal Policy Optimization). The environment integrates with your existing race car simulation and trained model.

## Files Overview

- **`race_car_env.py`** - Custom Gymnasium environment implementation
- **`train_ppo.py`** - PPO training script using Stable-Baselines3
- **`test_environment.py`** - Test script to validate environment functionality
- **`requirements_ppo.txt`** - Python dependencies for PPO training
- **`README_PPO.md`** - This documentation file

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ppo.txt
```

### 2. Test the Environment

Before training, test that the environment works correctly:

```bash
python test_environment.py
```

This will run comprehensive tests to ensure:
- DTO creation and processing works
- State flattening matches your training data format
- Reward calculation functions properly
- Complete episodes can be run

### 3. Train PPO Agent

Start training with default settings:

```bash
python train_ppo.py --mode train
```

Or with custom parameters:

```bash
python train_ppo.py --mode train --timesteps 50000
```

### 4. Evaluate Trained Model

After training, evaluate the model:

```bash
python train_ppo.py --mode eval --model-path ./models/final_model.zip
```

## Environment Details

### Observation Space

The environment uses a **flat vector** of 18 features that matches your training data:

1. **Velocity (2 features)**: `velocity_x`, `velocity_y` (scaled using `velocity_scaler.pkl`)
2. **Sensors (16 features)**: All sensor readings normalized by dividing by 1000.0

Feature order (critical for consistency):
```python
['velocity_x', 'velocity_y', 'sensor_back', 'sensor_back_left_back', 
 'sensor_back_right_back', 'sensor_front', 'sensor_front_left_front', 
 'sensor_front_right_front', 'sensor_left_back', 'sensor_left_front', 
 'sensor_left_side', 'sensor_left_side_back', 'sensor_left_side_front', 
 'sensor_right_back', 'sensor_right_front', 'sensor_right_side', 
 'sensor_right_side_back', 'sensor_right_side_front']
```

### Action Space

Discrete action space with 5 actions:
- `0`: NOTHING
- `1`: ACCELERATE  
- `2`: DECELERATE
- `3`: STEER_LEFT
- `4`: STEER_RIGHT

### Reward Function

Multi-component reward system:

1. **Distance Progress** (+1.0 per unit): Reward for making forward progress
2. **Crash Penalty** (-100.0): Large penalty for crashing
3. **Time Penalty** (-0.1 per step): Small penalty to encourage efficiency
4. **Speed Bonus** (+0.1 × velocity_magnitude): Reward for maintaining good speed
5. **Proximity Penalty** (-0.5 when close): Penalty for getting too close to obstacles

## Integration with Your Game

### API Integration

The environment expects your game simulation to provide an API endpoint that:

1. **Accepts POST requests** with:
   ```json
   {
     "action": "ACCELERATE",
     "game_session_id": "session_123456"
   }
   ```

2. **Returns responses** matching `RaceCarPredictRequestDto`:
   ```json
   {
     "did_crash": false,
     "elapsed_ticks": 100,
     "distance": 150.5,
     "velocity": {"x": 5.2, "y": -1.3},
     "sensors": {
       "back": 500.0,
       "front": 200.0,
       "left_side": 120.0,
       ...
     }
   }
   ```

### Mock Mode

If your API isn't ready, the environment runs in **mock mode** with simulated data for testing.

## Configuration

### Environment Configuration

```python
env_config = {
    'api_endpoint': 'http://localhost:8000/predict',
    'max_steps': 1000,
    'reward_config': {
        'distance_progress': 1.0,
        'crash_penalty': -100.0,
        'time_penalty': -0.1,
        'speed_bonus': 0.1,
        'proximity_penalty': -0.5
    }
}
```

### PPO Hyperparameters

```python
ppo_config = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}
```

## Training Pipeline

### 1. Data Preprocessing

Ensure your `velocity_scaler.pkl` file is present. This should be the same scaler used when training your supervised model.

### 2. Environment Testing

Always run the test script before training:
```bash
python test_environment.py
```

### 3. Training

Start with shorter training runs to validate everything works:
```bash
python train_ppo.py --mode train --timesteps 10000
```

### 4. Monitoring

Training progress is logged to:
- **Console**: Real-time training information
- **TensorBoard**: `./tensorboard_logs/` (run `tensorboard --logdir=tensorboard_logs`)
- **Models**: Saved to `./models/`

### 5. Evaluation

Evaluate trained models:
```bash
python train_ppo.py --mode eval --model-path ./models/best_model.zip
```

## Advanced Usage

### Parallel Training

Train with multiple parallel environments:

```python
training_config = {
    'n_envs': 8,              # Number of parallel environments
    'use_subprocess': True,   # Use subprocesses for better parallelization
}
```

### Custom Reward Functions

Modify reward weights in the configuration:

```python
reward_config = {
    'distance_progress': 2.0,    # Emphasize distance more
    'crash_penalty': -200.0,     # Harsher crash penalty
    'time_penalty': -0.05,       # Less time pressure
    'speed_bonus': 0.2,          # Reward speed more
    'proximity_penalty': -1.0    # Stronger proximity penalty
}
```

### Fine-tuning from Pretrained Models

Load and continue training from a previous model:

```bash
python train_ppo.py --mode train --model-path ./models/previous_model.zip
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements_ppo.txt`

2. **Shape Mismatches**: The environment observation must match your trained model's input. Run tests to verify.

3. **API Connection**: Check that your game simulation API is running and accessible.

4. **Scaler Missing**: Ensure `velocity_scaler.pkl` exists and is the same one used for training.

### Debug Mode

Run with verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Individual Components

Test specific parts:

```python
from race_car_env import make_race_car_env
from dtos import RaceCarPredictRequestDto

# Test environment creation
env = make_race_car_env()

# Test state processing
test_dto = RaceCarPredictRequestDto(...)
processed = env._flatten_and_process_state(test_dto)
print(f"Processed shape: {processed.shape}")
```

## Performance Tips

1. **Vectorization**: Use multiple parallel environments (`n_envs > 1`)
2. **Batch Size**: Tune batch size based on your hardware
3. **Learning Rate**: Start with 3e-4, adjust based on training curves
4. **Episode Length**: Longer episodes provide more learning signal but slower training

## Next Steps

1. **Test Environment**: Run `test_environment.py` successfully
2. **Short Training**: Train for 10k steps to verify everything works
3. **Hyperparameter Tuning**: Adjust reward weights and PPO parameters
4. **Long Training**: Train for 100k+ steps for good performance
5. **Evaluation**: Compare against your supervised learning baseline

## Integration Notes

This environment is designed to work with your existing:
- **DTOs** (`dtos.py`)
- **Velocity Scaler** (`velocity_scaler.pkl`)
- **Data Format** (from `expert_training_data.csv`)
- **Action Space** (5 discrete actions)

The key is ensuring the observation preprocessing in the environment exactly matches what your supervised model was trained on.
