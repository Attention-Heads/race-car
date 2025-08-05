# PPO Training Visualizations

This enhanced version of `train_ppo.py` includes comprehensive visualization capabilities that automatically generate training diagrams organized in timestamped folders.

## 🎯 Features

### Automatic Plot Generation
- **Training Curves**: Episode rewards, lengths, crash rates, learning rates, and losses
- **Evaluation Metrics**: Mean rewards with error bars, success vs crash rates, distance traveled
- **Performance Summary**: Final statistics, reward distributions, learning progress, and convergence analysis
- **Raw Data Export**: All metrics saved as JSON and CSV files for further analysis

### Organized Directory Structure
All visualizations are automatically organized in timestamped folders:

```
visualizations/
└── training_session_20250805_143022/
    ├── training_curves/
    │   └── training_curves_20250805_143022.png
    ├── evaluation_metrics/
    │   └── evaluation_metrics_20250805_143022.png
    ├── model_analysis/
    ├── episode_analysis/
    ├── performance_summary/
    │   └── performance_summary_20250805_143022.png
    ├── training_metrics_20250805_143022.json
    ├── training_data_20250805_143022.csv
    └── evaluation_data_20250805_143022.csv
```

## 🚀 Quick Start

### 1. Run Training with Visualizations (Default)
```bash
python train_ppo.py --mode train --timesteps 50000
```

### 2. Run Training Without Visualizations
```bash
python train_ppo.py --mode train --disable-plots
```

### 3. Custom Visualization Directory
```bash
python train_ppo.py --mode train --viz-dir ./my_training_results
```

### 4. Demo the Visualization System
```bash
python demo_visualizations.py
```

## 📊 Generated Visualizations

### Training Curves (`training_curves/`)
- **Episode Rewards**: Shows learning progress with moving average
- **Episode Lengths**: Track how long episodes last over time
- **Crash Rates**: Monitor safety improvements
- **Learning Rate**: Shows learning rate schedule
- **Policy & Value Losses**: PPO-specific loss curves
- **Explained Variance**: Model's ability to predict values

### Evaluation Metrics (`evaluation_metrics/`)
- **Mean Reward ± Std**: Evaluation performance with confidence intervals
- **Success vs Crash Rates**: Safety and performance metrics
- **Average Distance**: How far the agent travels
- **Performance Improvement Rate**: Rate of learning over time

### Performance Summary (`performance_summary/`)
- **Training Statistics**: Duration, timesteps, final performance
- **Reward Distribution**: Histogram of episode rewards
- **Learning Progress**: Percentage improvement from start
- **Training Stability**: Rolling standard deviation
- **Model Convergence**: Recent loss trends

## 📈 Understanding the Plots

### Training Curves
- **Upward trending rewards** = Good learning
- **Decreasing crash rates** = Improved safety
- **Stable or decreasing losses** = Model convergence
- **High explained variance** = Good value function

### Evaluation Metrics
- **Increasing mean rewards** = Consistent improvement
- **Decreasing error bars** = More stable performance
- **Low crash rates** = Safe driving behavior
- **Increasing distances** = Better endurance

### Performance Summary
- **Positive learning progress** = Successful training
- **Tight reward distribution** = Consistent performance
- **Decreasing rolling std** = Stable learning
- **Converging loss trends** = Model has learned

## 🔧 Configuration

### Visualization Settings
You can customize visualization settings in `create_training_config()`:

```python
'visualization_config': {
    'enable_plots': True,          # Enable/disable all visualizations
    'plot_freq': 5000,             # How often to update plots (timesteps)
    'save_raw_data': True,         # Save metrics to CSV/JSON
    'create_summary': True,        # Create final summary plots
}
```

### Plot Styling
The visualizations use:
- **Seaborn styling** for professional appearance
- **High DPI (300)** for publication-quality images
- **Husl color palette** for distinct, colorful plots
- **Grid lines** for easy reading
- **Moving averages** for trend visualization

## 📂 File Organization

### Automatic Timestamping
Each training session creates a unique timestamped directory:
- Format: `training_session_YYYYMMDD_HHMMSS`
- Prevents overwriting previous results
- Easy chronological organization

### Data Formats
- **PNG Images**: High-quality plots (300 DPI)
- **JSON Files**: Complete metrics with metadata
- **CSV Files**: Tabular data for external analysis

## 🛠️ Advanced Usage

### Custom Analysis
Access raw data for custom analysis:

```python
import json
import pandas as pd

# Load training metrics
with open('visualizations/training_session_*/training_metrics_*.json', 'r') as f:
    data = json.load(f)

# Load as DataFrame
df = pd.read_csv('visualizations/training_session_*/training_data_*.csv')
```

### Integration with TensorBoard
Visualizations complement TensorBoard logs:
- **TensorBoard**: Real-time monitoring during training
- **Our visualizations**: Comprehensive post-training analysis

### Comparison Across Runs
Compare multiple training sessions:

```bash
# Train with different hyperparameters
python train_ppo.py --timesteps 25000 --viz-dir ./experiment_1
python train_ppo.py --timesteps 50000 --viz-dir ./experiment_2
```

## 🎨 Customization

### Adding New Metrics
To track additional metrics, modify the `TrainingVisualizer` class:

1. Add new metric to storage dictionaries
2. Update logging methods
3. Create new plot functions
4. Add to visualization pipeline

### Custom Plot Styles
Modify the matplotlib style settings in the visualizer initialization:

```python
plt.style.use('your_custom_style')
sns.set_palette("your_color_palette")
```

## 🐛 Troubleshooting

### Common Issues

**No plots generated**:
- Check that `enable_plots: True` in config
- Ensure matplotlib and seaborn are installed
- Verify write permissions to visualization directory

**Empty plots**:
- Training may be too short to collect meaningful data
- Check that metrics are being logged correctly
- Increase `plot_freq` for more frequent updates

**Memory issues**:
- Reduce `plot_freq` to save memory
- Disable raw data saving if not needed
- Clear old visualization directories periodically

### Dependencies
Ensure these packages are installed:
```bash
pip install matplotlib seaborn pandas
```

## 📚 Examples

### Basic Training Session
```bash
python train_ppo.py --mode train --timesteps 100000
```
Generates comprehensive visualizations in `./visualizations/training_session_*/`

### Quick Testing
```bash
python train_ppo.py --mode train --timesteps 5000 --viz-dir ./quick_test
```
Fast training with visualizations in custom directory

### Production Training
```bash
python train_ppo.py --mode train --timesteps 500000 --viz-dir ./production_models
```
Long training session with detailed analysis

## 🔮 Future Enhancements

Planned visualization features:
- **Model Analysis**: Weight distributions, activation patterns
- **Episode Analysis**: Individual episode trajectories
- **Action Analysis**: Action distribution over time
- **Comparative Analysis**: Multi-run comparisons
- **Interactive Plots**: Web-based dashboard
- **Video Generation**: Animated training progress

## 📞 Support

For visualization-related issues:
1. Check this README for common solutions
2. Run `demo_visualizations.py` to test the system
3. Verify all dependencies are installed
4. Check file permissions for the visualization directory

---

**Happy Training! 🚗💨**
