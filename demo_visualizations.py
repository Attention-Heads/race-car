#!/usr/bin/env python3
"""
Demo script to showcase the PPO training visualization capabilities.
This script runs a short training session to demonstrate the visualization features.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add the current directory to the path to import local modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from train_ppo import TrainingVisualizer, create_training_config


def demo_visualization_system():
    """
    Demonstrate the visualization system with mock data.
    """
    print("=" * 60)
    print("PPO Training Visualization Demo")
    print("=" * 60)
    
    # Create a visualizer instance
    visualizer = TrainingVisualizer(base_dir="./demo_visualizations")
    
    print(f"📁 Demo visualizations will be saved to: {visualizer.session_dir}")
    
    # Generate mock training data
    print("📊 Generating mock training data...")
    timesteps = range(0, 10000, 100)
    
    for i, timestep in enumerate(timesteps):
        # Simulate improving performance with some noise
        base_reward = 50 + i * 2 + np.random.normal(0, 10)
        episode_length = 100 + i * 5 + np.random.normal(0, 20)
        crash_rate = max(0, 50 - i * 0.5 + np.random.normal(0, 5))
        
        metrics = {
            'episode_rewards': base_reward,
            'episode_lengths': max(10, episode_length),
            'crash_rates': max(0, min(100, crash_rate)),
            'learning_rates': 3e-4 * (0.99 ** (i // 10)),
            'policy_losses': 0.1 * np.exp(-i * 0.01) + np.random.normal(0, 0.01),
            'value_losses': 0.2 * np.exp(-i * 0.01) + np.random.normal(0, 0.02),
            'explained_variances': min(1.0, 0.1 + i * 0.01 + np.random.normal(0, 0.05))
        }
        
        visualizer.log_training_step(timestep, metrics)
        
        # Add evaluation data at regular intervals
        if i % 10 == 0 and i > 0:
            eval_metrics = {
                'mean_rewards': base_reward + np.random.normal(0, 5),
                'std_rewards': max(1, 15 - i * 0.1),
                'crash_rates': max(0, crash_rate),
                'success_rates': min(100, 100 - crash_rate),
                'average_distances': 50 + i * 3 + np.random.normal(0, 10)
            }
            visualizer.log_evaluation_step(timestep, eval_metrics)
    
    # Create all visualization plots
    print("📈 Creating training curves...")
    visualizer.create_training_curves()
    
    print("📊 Creating evaluation plots...")
    visualizer.create_evaluation_plots()
    
    # Create performance summary with mock final stats
    final_stats = {
        'training_duration': '0:15:30',
        'total_timesteps': 10000,
        'final_mean_reward': 75.3,
        'best_mean_reward': 82.1,
        'final_crash_rate': 15.2,
        'best_success_rate': 91.5,
    }
    
    print("📋 Creating performance summary...")
    visualizer.create_performance_summary(final_stats)
    
    print("💾 Saving metrics data...")
    visualizer.save_metrics_data()
    
    # Print summary
    print("\n" + "=" * 60)
    print("DEMO VISUALIZATION SUMMARY")
    print("=" * 60)
    print(f"📁 Base directory: {visualizer.session_dir}")
    print(f"📊 Training curves: {visualizer.dirs['training_curves']}")
    print(f"📈 Evaluation metrics: {visualizer.dirs['evaluation_metrics']}")
    print(f"📋 Performance summary: {visualizer.dirs['performance_summary']}")
    print(f"💾 Raw data: {visualizer.session_dir}")
    print("=" * 60)
    
    return visualizer.session_dir


def show_directory_structure():
    """
    Show the expected directory structure for visualizations.
    """
    config = create_training_config()
    viz_dir = config['paths']['visualization_dir']
    
    print("\n" + "=" * 60)
    print("VISUALIZATION DIRECTORY STRUCTURE")
    print("=" * 60)
    print(f"""
When you run PPO training, visualizations will be organized as follows:

{viz_dir}/
└── training_session_YYYYMMDD_HHMMSS/
    ├── training_curves/
    │   └── training_curves_YYYYMMDD_HHMMSS.png
    ├── evaluation_metrics/
    │   └── evaluation_metrics_YYYYMMDD_HHMMSS.png  
    ├── model_analysis/
    │   └── (future: model weight analysis, etc.)
    ├── episode_analysis/
    │   └── (future: episode-specific analysis)
    ├── performance_summary/
    │   └── performance_summary_YYYYMMDD_HHMMSS.png
    ├── training_metrics_YYYYMMDD_HHMMSS.json
    ├── training_data_YYYYMMDD_HHMMSS.csv
    └── evaluation_data_YYYYMMDD_HHMMSS.csv

Each training session gets its own timestamped folder for easy organization!
    """)
    print("=" * 60)


def main():
    """
    Main demo function.
    """
    print("🚗 Race Car PPO Training Visualization Demo")
    print("This demo shows the visualization capabilities without running actual training.\n")
    
    # Show directory structure
    show_directory_structure()
    
    # Run the demo
    demo_dir = demo_visualization_system()
    
    print(f"\n✅ Demo completed successfully!")
    print(f"🔍 Check the generated visualizations in: {demo_dir}")
    print("\n💡 To run actual PPO training with visualizations:")
    print("   python train_ppo.py --mode train --timesteps 10000")
    print("\n💡 To disable visualizations during training:")
    print("   python train_ppo.py --mode train --disable-plots")
    print("\n💡 To specify a custom visualization directory:")
    print("   python train_ppo.py --mode train --viz-dir ./my_custom_viz")


if __name__ == "__main__":
    main()
