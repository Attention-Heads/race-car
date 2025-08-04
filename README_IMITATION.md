# Imitation Learning for Race Car Environment

This repository implements a complete pipeline for training a race car AI using imitation learning followed by PPO fine-tuning. The system learns from expert demonstrations and can be further improved through reinforcement learning.

## Overview

The training pipeline consists of:

1. **Data Preparation**: Clean and preprocess expert demonstration data
2. **Behavioral Cloning**: Train a neural network to mimic expert behavior
3. **PPO Initialization**: Initialize PPO agent with behavioral cloning weights
4. **Fine-tuning**: Improve the policy using PPO reinforcement learning

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ppo.txt
```

### 2. Prepare Expert Data (Optional but Recommended)

Analyze and clean your expert data:

```bash
python prepare_expert_data.py --input expert_training_data.csv --output expert_data_clean.csv --analyze --filter --balance
```

This will:
- Create analysis plots showing data distribution
- Filter out low-quality games
- Balance the action distribution
- Save cleaned data

### 3. Run Complete Training Pipeline

Train both behavioral cloning and PPO in one command:

```bash
python train_integrated.py --expert_data expert_training_data.csv --bc_epochs 50 --ppo_timesteps 50000
```

### 4. Or Train Step by Step

#### Step 1: Train Behavioral Cloning Model

```bash
python imitation_learning.py
```

#### Step 2: Fine-tune with PPO

```bash
python train_ppo.py --mode train --model-path models/ppo_initialized_with_bc.zip
```

### 5. Evaluate Trained Models

```bash
python train_ppo.py --mode eval --model-path models/final_model.zip
```

## Architecture

### Neural Network Architecture

The system uses a shared neural network architecture for both behavioral cloning and PPO:

```
Input (18 features) -> Dense(128) -> ReLU -> Dropout(0.1) 
                   -> Dense(256) -> ReLU -> Dropout(0.1)
                   -> Dense(128) -> ReLU 
                   -> Dense(256) -> ReLU
                   -> Output (5 actions)
```

### Feature Processing

- **Velocity features** (2): `velocity_x`, `velocity_y` - scaled using StandardScaler
- **Sensor features** (16): Distance readings from various sensors - normalized by dividing by 1000

### Action Space

The model predicts one of 5 discrete actions:
- `NOTHING` (0): No action
- `ACCELERATE` (1): Increase speed
- `DECELERATE` (2): Decrease speed  
- `STEER_LEFT` (3): Turn left
- `STEER_RIGHT` (4): Turn right

## Command Line Options

### train_integrated.py

Complete training pipeline with both behavioral cloning and PPO:

```bash
python train_integrated.py [OPTIONS]

Options:
  --expert_data PATH          Path to expert training data CSV
  --model_dir PATH           Directory to save models
  --log_dir PATH             Directory for logs and evaluation
  
  # Behavioral Cloning
  --bc_epochs INT            Number of BC training epochs (default: 100)
  --bc_batch_size INT        BC batch size (default: 512)
  --bc_learning_rate FLOAT   BC learning rate (default: 1e-3)
  --bc_patience INT          Early stopping patience (default: 10)
  
  # PPO Fine-tuning
  --ppo_timesteps INT        PPO training timesteps (default: 100,000)
  --ppo_learning_rate FLOAT  PPO learning rate (default: 3e-4)
  --ppo_n_steps INT          Steps per update (default: 2048)
  --ppo_batch_size INT       PPO batch size (default: 64)
  --eval_freq INT            Evaluation frequency (default: 5000)
```

### prepare_expert_data.py

Data preparation and analysis:

```bash
python prepare_expert_data.py [OPTIONS]

Options:
  --input PATH               Input expert data CSV
  --output PATH              Output processed CSV
  --analyze                  Create analysis plots
  --filter                   Apply data filtering
  --balance                  Balance action distribution
  --remove-crashes           Remove crashed states
  --min-distance FLOAT       Minimum game distance threshold
  --max-crash-ratio FLOAT    Max ratio of crash games to keep
  --max-samples-per-action INT  Max samples per action class
```

### train_ppo.py

PPO training with optional pre-trained model loading:

```bash
python train_ppo.py [OPTIONS]

Options:
  --mode {train,eval}        Training or evaluation mode
  --model-path PATH          Path to pretrained model
  --timesteps INT            Override training timesteps
```

## File Structure

```
race-car/
├── imitation_learning.py      # Main imitation learning implementation
├── train_integrated.py        # Complete training pipeline
├── prepare_expert_data.py     # Data preparation utilities
├── train_ppo.py              # PPO training (enhanced)
├── race_car_env.py           # Environment implementation
├── expert_training_data.csv   # Expert demonstration data
├── velocity_scaler.pkl       # Pre-trained velocity scaler
├── models/                   # Saved models directory
│   ├── behavioral_cloning_final.pth
│   ├── ppo_initialized_with_bc.zip
│   └── ppo_final_model.zip
├── logs/                     # Training logs
└── tensorboard_logs/         # TensorBoard logs
```

## Key Features

### 1. Robust Data Processing
- Handles missing sensor values
- Applies consistent normalization
- Supports velocity scaling
- Filters low-quality demonstrations

### 2. Transfer Learning
- Shared architecture between BC and PPO
- Weight transfer from BC to PPO
- Preserves learned features during fine-tuning

### 3. Comprehensive Evaluation
- Action accuracy metrics for BC
- Episode performance metrics for PPO
- Crash rate analysis
- Training curve visualization

### 4. Flexible Configuration
- Configurable hyperparameters
- Multiple training modes
- Extensible architecture

## Training Tips

### 1. Data Quality
- Use `prepare_expert_data.py` to analyze your data first
- Consider filtering out very short games or excessive crashes
- Balance action distribution if one action dominates

### 2. Behavioral Cloning
- Start with 50-100 epochs, use early stopping
- Monitor both training loss and test accuracy
- Higher accuracy doesn't always mean better RL performance

### 3. PPO Fine-tuning
- Start with pre-trained BC model for faster convergence
- Use evaluation callbacks to monitor progress
- Adjust reward function based on your objectives

### 4. Hyperparameter Tuning
- BC learning rate: 1e-3 to 1e-4
- PPO learning rate: 3e-4 to 1e-4
- PPO timesteps: Start with 50K-100K
- Evaluation frequency: Every 5K-10K steps

## Example Workflow

```bash
# 1. Analyze raw data
python prepare_expert_data.py --input expert_training_data.csv --analyze

# 2. Clean and balance data
python prepare_expert_data.py --input expert_training_data.csv --output clean_data.csv --filter --balance --min-distance 100

# 3. Train complete pipeline
python train_integrated.py --expert_data clean_data.csv --bc_epochs 50 --ppo_timesteps 100000

# 4. Evaluate final model
python train_ppo.py --mode eval --model-path models/ppo_final_model.zip
```

## Monitoring Training

### TensorBoard
Monitor training progress with TensorBoard:

```bash
tensorboard --logdir tensorboard_logs/
```

### Log Files
Check training logs in the `logs/` directory for detailed metrics and evaluation results.

## Troubleshooting

### Common Issues

1. **Low BC Accuracy**: 
   - Check data quality and distribution
   - Try balancing actions or filtering bad demonstrations
   - Increase model capacity or training epochs

2. **PPO Not Improving**:
   - Verify BC model is loaded correctly
   - Check reward function design
   - Adjust PPO hyperparameters (learning rate, steps)

3. **Memory Issues**:
   - Reduce batch size for BC training
   - Use fewer parallel environments for PPO
   - Process data in chunks for large datasets

4. **Environment Issues**:
   - Ensure `velocity_scaler.pkl` exists or disable velocity scaling
   - Check that action mappings match between BC and environment
   - Verify sensor normalization consistency

## Advanced Usage

### Custom Reward Functions
Modify the reward function in `race_car_env.py` to optimize for specific behaviors.

### Different Architectures
Extend `RaceCarFeaturesExtractor` in `imitation_learning.py` to experiment with different network architectures.

### Data Augmentation
Add data augmentation techniques in the `ExpertDataset` class for better generalization.

### Multi-Modal Learning
Combine behavioral cloning with other learning paradigms like inverse reinforcement learning.

## Contributing

Feel free to submit issues and enhancement requests. The codebase is designed to be modular and extensible for research and development purposes.
