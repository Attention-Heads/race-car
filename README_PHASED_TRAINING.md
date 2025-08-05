# Phased Training for Race Car PPO

This implementation adds **phased training** capabilities to the PPO race car agent, allowing for gradual transition from conservative exploration around a BC (Behavioral Cloning) policy to more aggressive optimization.

## 🎯 Overview

Phased training helps bridge the gap between imitation learning and reinforcement learning by:

1. **Starting conservatively** - Low learning rates and high entropy to gently explore around the BC policy
2. **Gradually increasing exploration** - Progressively adjusting hyperparameters to encourage more exploration
3. **Ending with optimization** - Using standard PPO parameters for final performance optimization

## 🔧 Key Features

- **Automatic hyperparameter scheduling** - Learning rate, entropy coefficient, clip range, and epoch count adjust automatically
- **Flexible phase configuration** - Define custom phases with different durations and hyperparameters
- **Enhanced visualizations** - Track hyperparameter changes and phase transitions
- **BC model compatibility** - Start from a pretrained behavioral cloning model
- **Easy to use** - Command-line interface with predefined configurations

## 🚀 Quick Start

### Basic Usage

```bash
# Train with default phased training
python train_ppo.py --mode train

# Disable phased training (standard PPO)
python train_ppo.py --mode train --disable-phased-training

# Load a pretrained BC model for fine-tuning
python train_ppo.py --mode train --model-path ./models/bc_model.zip
```

### Interactive Demo

```bash
# Run the interactive demo with preset configurations
python run_phased_training_demo.py
```

The demo offers:
- **Conservative** (4 phases, gentle progression)
- **Aggressive** (3 phases, faster learning)
- **Experimental** (6 phases, research-oriented)
- **Standard PPO** (no phasing)

## 📊 Phase Configuration

### Default Conservative Configuration

```python
phases = [
    # Phase 1: Ultra-gentle exploration
    {
        'phase_name': 'gentle_exploration',
        'duration_timesteps': 30_000,
        'learning_rate': 1e-4,     # Low learning rate
        'ent_coef': 0.05,          # High entropy for exploration
        'clip_range': 0.1,         # Conservative clipping
        'n_epochs': 5,             # Fewer epochs to avoid overfitting
    },
    # Phase 2: Moderate learning
    {
        'phase_name': 'moderate_learning', 
        'duration_timesteps': 40_000,
        'learning_rate': 2e-4,     # Increased learning rate
        'ent_coef': 0.02,          # Medium entropy
        'clip_range': 0.15,        # Medium clipping
        'n_epochs': 8,             # More epochs
    },
    # Phase 3: Aggressive optimization
    {
        'phase_name': 'aggressive_optimization',
        'duration_timesteps': 30_000,
        'learning_rate': 3e-4,     # Standard learning rate
        'ent_coef': 0.01,          # Low entropy for exploitation
        'clip_range': 0.2,         # Standard clipping
        'n_epochs': 10,            # Full epochs
    }
]
```

## 📈 Hyperparameter Evolution

The phased training system automatically adjusts:

- **Learning Rate**: Gradually increases for faster learning
- **Entropy Coefficient**: Decreases to transition from exploration to exploitation
- **Clip Range**: Adjusts PPO clipping for different learning phases
- **Training Epochs**: Increases for more thorough optimization

## 🎨 Visualizations

Phased training includes enhanced visualizations:

### Standard Training Plots
- Episode rewards over time
- Training stability metrics
- Policy and value losses
- Learning rate evolution

### Phased Training Specific Plots
- **Hyperparameter evolution** - Track how learning rate, entropy, and clip range change
- **Phase transition markers** - Visual indicators of when phases change
- **Performance by phase** - Compare average rewards across different phases
- **Learning progression** - Smoothed rewards with phase transitions highlighted

## 🔧 Custom Configuration

Create your own phased training configuration:

```python
def create_custom_config():
    config = create_training_config()
    
    config['phased_training'] = {
        'enable_phased_training': True,
        'phases': [
            {
                'phase_name': 'your_phase_name',
                'duration_timesteps': 20_000,
                'learning_rate': 1e-4,
                'ent_coef': 0.03,
                'clip_range': 0.12,
                'n_epochs': 6,
                'description': 'Your phase description'
            },
            # Add more phases...
        ]
    }
    
    return config
```

## 🎯 Best Practices

### For BC Fine-tuning
1. **Start very conservatively** - Use low learning rates (1e-5 to 1e-4)
2. **High initial entropy** - 0.05-0.1 for gentle exploration
3. **Small clip ranges** - 0.05-0.1 to avoid large policy changes
4. **Fewer epochs initially** - 3-5 epochs to prevent overfitting

### For Faster Learning
1. **Moderate initial parameters** - Don't start too conservatively
2. **Shorter phase durations** - 15k-25k timesteps per phase
3. **Aggressive final phase** - Higher learning rates and more epochs

### For Research/Experimentation
1. **Many short phases** - 5-10k timesteps each for fine-grained control
2. **Gradual transitions** - Small changes between consecutive phases
3. **Detailed logging** - Enable all visualizations for analysis

## 📝 Configuration Parameters

### Phase Parameters
- `phase_name`: Human-readable name for the phase
- `duration_timesteps`: Number of timesteps for this phase
- `learning_rate`: PPO learning rate for this phase
- `ent_coef`: Entropy coefficient for exploration/exploitation balance
- `clip_range`: PPO clip range for policy updates
- `n_epochs`: Number of training epochs per update
- `description`: Optional description of the phase purpose

### Training Parameters
- `enable_phased_training`: Boolean to enable/disable phasing
- `phases`: List of phase configurations
- `total_timesteps`: Automatically calculated from phase durations

## 🎮 Command Line Options

```bash
# Basic options
python train_ppo.py --mode train                    # Standard training
python train_ppo.py --mode eval --model-path MODEL  # Evaluation

# Phased training options
python train_ppo.py --disable-phased-training       # Use standard PPO
python train_ppo.py --model-path BC_MODEL           # Fine-tune from BC model

# Visualization options
python train_ppo.py --disable-plots                 # No visualizations
python train_ppo.py --viz-dir ./custom_viz          # Custom viz directory

# Training options
python train_ppo.py --timesteps 150000              # Override total timesteps
```

## 🔬 Monitoring Training

During training, you'll see:

```
🔄 PHASE TRANSITION AT TIMESTEP 30,000
Entering Phase 2: moderate_learning
Description: Balanced exploration and exploitation
Hyperparameter changes:
  Learning Rate: 0.000100 → 0.000200
  Entropy Coef:  0.050000 → 0.020000
  Clip Range:    0.100000 → 0.150000
  N Epochs:      5 → 8

📊 Phase Progress: moderate_learning - 25.3% complete (29,847 timesteps remaining)
```

## 📊 Results Analysis

After training, check the visualization directory for:

- `training_curves/` - Standard training metrics
- `evaluation_metrics/` - Evaluation performance over time
- `model_analysis/` - Phased training specific analysis
- `performance_summary/` - Overall training summary

The phased training analysis includes:
- Hyperparameter evolution plots
- Performance comparison by phase
- Learning progression with phase markers
- Phase transition timeline

## 🤝 Integration with Existing Code

Phased training is designed to be backwards compatible:

- **Existing configs work unchanged** - Just disable phased training
- **Same model interface** - Models can be loaded/saved normally
- **Standard evaluation** - No changes needed for model evaluation
- **BC model compatibility** - Load BC models as pretrained starting points

## 🐛 Troubleshooting

### Common Issues

1. **Phase transitions not visible**
   - Ensure `enable_phased_training: True` in config
   - Check that phases have different hyperparameters
   - Verify visualization is enabled

2. **Training unstable**
   - Try more conservative initial parameters
   - Reduce learning rate changes between phases
   - Use smaller clip ranges

3. **Poor performance**
   - Ensure final phase has good optimization parameters
   - Consider longer phase durations
   - Check BC model quality if fine-tuning

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show detailed phase transition information and parameter updates.

## 📚 References

The phased training approach is inspired by curriculum learning and gradual unfreezing techniques commonly used in deep learning, adapted specifically for the transition from imitation learning to reinforcement learning in continuous control tasks.
