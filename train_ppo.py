"""
PPO Training Script for Race Car Environment

This script demonstrates how to use the custom RaceCarEnv with Stable-Baselines3 PPO
for fine-tuning your race car agent.
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from race_car_env import make_race_car_env
from imitation_learning import RaceCarFeaturesExtractor
import argparse
from typing import Dict, Any, List, Tuple
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import json
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrainingVisualizer:
    """
    Handles creation and organization of training visualization diagrams.
    """
    
    def __init__(self, base_dir: str = "./visualizations"):
        """
        Initialize the visualizer with timestamped directories.
        
        Args:
            base_dir: Base directory for all visualizations
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir)
        
        # Create organized directory structure
        self.session_dir = self.base_dir / f"training_session_{self.timestamp}"
        self.dirs = {
            'training_curves': self.session_dir / "training_curves",
            'evaluation_metrics': self.session_dir / "evaluation_metrics", 
            'model_analysis': self.session_dir / "model_analysis",
            'episode_analysis': self.session_dir / "episode_analysis",
            'performance_summary': self.session_dir / "performance_summary"
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Storage for tracking metrics
        self.training_metrics = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'crash_rates': [],
            'learning_rates': [],
            'policy_losses': [],
            'value_losses': [],
            'explained_variances': []
        }
        
        self.evaluation_metrics = {
            'eval_timesteps': [],
            'mean_rewards': [],
            'std_rewards': [],
            'crash_rates': [],
            'success_rates': [],
            'average_distances': []
        }
        
        logger.info(f"Visualizations will be saved to: {self.session_dir}")
    
    def log_training_step(self, timestep: int, metrics: Dict[str, float]):
        """Log metrics from a training step."""
        self.training_metrics['timesteps'].append(timestep)
        for key, value in metrics.items():
            if key in self.training_metrics:
                self.training_metrics[key].append(value)
    
    def log_evaluation_step(self, timestep: int, eval_results: Dict[str, float]):
        """Log metrics from an evaluation step."""
        self.evaluation_metrics['eval_timesteps'].append(timestep)
        for key, value in eval_results.items():
            if key in self.evaluation_metrics:
                self.evaluation_metrics[key].append(value)
    
    def create_training_curves(self):
        """Create comprehensive training curve visualizations."""
        if not self.training_metrics['timesteps']:
            logger.warning("No training metrics to plot")
            return
            
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'PPO Training Curves - {self.timestamp}', fontsize=16, fontweight='bold')
        
        # Episode Rewards
        if self.training_metrics['episode_rewards']:
            axes[0, 0].plot(self.training_metrics['timesteps'], self.training_metrics['episode_rewards'], 
                           color='blue', alpha=0.7, linewidth=1.5)
            axes[0, 0].set_title('Episode Rewards Over Time')
            axes[0, 0].set_xlabel('Training Timesteps')
            axes[0, 0].set_ylabel('Episode Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add moving average
            if len(self.training_metrics['episode_rewards']) > 10:
                window = min(50, len(self.training_metrics['episode_rewards']) // 10)
                rewards_df = pd.Series(self.training_metrics['episode_rewards'])
                moving_avg = rewards_df.rolling(window=window, center=True).mean()
                axes[0, 0].plot(self.training_metrics['timesteps'], moving_avg, 
                               color='red', linewidth=2, label=f'Moving Avg ({window})')
                axes[0, 0].legend()
        
        # Episode Lengths
        if self.training_metrics['episode_lengths']:
            axes[0, 1].plot(self.training_metrics['timesteps'], self.training_metrics['episode_lengths'], 
                           color='green', alpha=0.7, linewidth=1.5)
            axes[0, 1].set_title('Episode Lengths Over Time')
            axes[0, 1].set_xlabel('Training Timesteps')
            axes[0, 1].set_ylabel('Episode Length')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Crash Rates
        if self.training_metrics['crash_rates']:
            axes[0, 2].plot(self.training_metrics['timesteps'], self.training_metrics['crash_rates'], 
                           color='red', alpha=0.7, linewidth=1.5)
            axes[0, 2].set_title('Crash Rate Over Time')
            axes[0, 2].set_xlabel('Training Timesteps')
            axes[0, 2].set_ylabel('Crash Rate (%)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Learning Rate
        if self.training_metrics['learning_rates']:
            axes[1, 0].plot(self.training_metrics['timesteps'], self.training_metrics['learning_rates'], 
                           color='purple', alpha=0.7, linewidth=1.5)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Training Timesteps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # Policy and Value Losses
        if self.training_metrics['policy_losses'] and self.training_metrics['value_losses']:
            axes[1, 1].plot(self.training_metrics['timesteps'], self.training_metrics['policy_losses'], 
                           color='orange', alpha=0.7, linewidth=1.5, label='Policy Loss')
            axes[1, 1].plot(self.training_metrics['timesteps'], self.training_metrics['value_losses'], 
                           color='cyan', alpha=0.7, linewidth=1.5, label='Value Loss')
            axes[1, 1].set_title('Policy and Value Losses')
            axes[1, 1].set_xlabel('Training Timesteps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Explained Variance
        if self.training_metrics['explained_variances']:
            axes[1, 2].plot(self.training_metrics['timesteps'], self.training_metrics['explained_variances'], 
                           color='brown', alpha=0.7, linewidth=1.5)
            axes[1, 2].set_title('Explained Variance')
            axes[1, 2].set_xlabel('Training Timesteps')
            axes[1, 2].set_ylabel('Explained Variance')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.dirs['training_curves'] / f"training_curves_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to: {save_path}")
    
    def create_evaluation_plots(self):
        """Create evaluation performance plots."""
        if not self.evaluation_metrics['eval_timesteps']:
            logger.warning("No evaluation metrics to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'PPO Evaluation Metrics - {self.timestamp}', fontsize=16, fontweight='bold')
        
        # Mean Rewards with Error Bars
        if self.evaluation_metrics['mean_rewards'] and self.evaluation_metrics['std_rewards']:
            axes[0, 0].errorbar(self.evaluation_metrics['eval_timesteps'], 
                               self.evaluation_metrics['mean_rewards'],
                               yerr=self.evaluation_metrics['std_rewards'],
                               color='blue', capsize=5, capthick=2, linewidth=2)
            axes[0, 0].set_title('Evaluation Mean Reward ± Std')
            axes[0, 0].set_xlabel('Training Timesteps')
            axes[0, 0].set_ylabel('Mean Episode Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Success vs Crash Rates
        if self.evaluation_metrics['crash_rates'] and self.evaluation_metrics['success_rates']:
            axes[0, 1].plot(self.evaluation_metrics['eval_timesteps'], 
                           self.evaluation_metrics['crash_rates'], 
                           color='red', linewidth=2, label='Crash Rate', marker='o')
            axes[0, 1].plot(self.evaluation_metrics['eval_timesteps'], 
                           self.evaluation_metrics['success_rates'], 
                           color='green', linewidth=2, label='Success Rate', marker='s')
            axes[0, 1].set_title('Success vs Crash Rates')
            axes[0, 1].set_xlabel('Training Timesteps')
            axes[0, 1].set_ylabel('Rate (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Average Distance Traveled
        if self.evaluation_metrics['average_distances']:
            axes[1, 0].plot(self.evaluation_metrics['eval_timesteps'], 
                           self.evaluation_metrics['average_distances'], 
                           color='purple', linewidth=2, marker='d')
            axes[1, 0].set_title('Average Distance Traveled')
            axes[1, 0].set_xlabel('Training Timesteps')
            axes[1, 0].set_ylabel('Average Distance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Performance Improvement Rate
        if len(self.evaluation_metrics['mean_rewards']) > 1:
            rewards = np.array(self.evaluation_metrics['mean_rewards'])
            improvement_rate = np.diff(rewards) / rewards[:-1] * 100
            axes[1, 1].plot(self.evaluation_metrics['eval_timesteps'][1:], 
                           improvement_rate, 
                           color='orange', linewidth=2, marker='^')
            axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Performance Improvement Rate')
            axes[1, 1].set_xlabel('Training Timesteps')
            axes[1, 1].set_ylabel('Improvement Rate (%)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.dirs['evaluation_metrics'] / f"evaluation_metrics_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Evaluation plots saved to: {save_path}")
    
    def create_performance_summary(self, final_stats: Dict[str, Any]):
        """Create a comprehensive performance summary visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'PPO Training Summary - {self.timestamp}', fontsize=16, fontweight='bold')
        
        # Final performance statistics
        def format_number(value, format_str):
            """Safely format a number or return 'N/A'"""
            if value == 'N/A' or value is None:
                return 'N/A'
            try:
                return format_str.format(value)
            except (ValueError, TypeError):
                return 'N/A'
        
        total_timesteps = final_stats.get('total_timesteps', 'N/A')
        total_timesteps_str = f"{total_timesteps:,}" if total_timesteps != 'N/A' else 'N/A'
        
        stats_text = f"""
        Training Duration: {final_stats.get('training_duration', 'N/A')}
        Total Timesteps: {total_timesteps_str}
        Final Mean Reward: {format_number(final_stats.get('final_mean_reward', 'N/A'), '{:.2f}')}
        Best Mean Reward: {format_number(final_stats.get('best_mean_reward', 'N/A'), '{:.2f}')}
        Final Crash Rate: {format_number(final_stats.get('final_crash_rate', 'N/A'), '{:.1f}')}%
        Best Success Rate: {format_number(final_stats.get('best_success_rate', 'N/A'), '{:.1f}')}%
        """
        
        axes[0, 0].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[0, 0].set_title('Training Statistics')
        axes[0, 0].axis('off')
        
        # Reward distribution histogram (if we have episode data)
        if self.training_metrics['episode_rewards']:
            axes[0, 1].hist(self.training_metrics['episode_rewards'], bins=30, 
                           alpha=0.7, color='blue', edgecolor='black')
            axes[0, 1].set_title('Episode Reward Distribution')
            axes[0, 1].set_xlabel('Episode Reward')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning progress (reward improvement over time)
        if self.evaluation_metrics['mean_rewards']:
            reward_progress = np.array(self.evaluation_metrics['mean_rewards'])
            if len(reward_progress) > 1:
                normalized_progress = (reward_progress - reward_progress[0]) / np.abs(reward_progress[0]) * 100
                axes[0, 2].plot(self.evaluation_metrics['eval_timesteps'], normalized_progress, 
                               color='green', linewidth=3, marker='o')
                axes[0, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[0, 2].set_title('Learning Progress (% Improvement)')
                axes[0, 2].set_xlabel('Training Timesteps')
                axes[0, 2].set_ylabel('Improvement from Start (%)')
                axes[0, 2].grid(True, alpha=0.3)
        
        # Training stability (rolling standard deviation)
        if len(self.training_metrics['episode_rewards']) > 20:
            rewards_series = pd.Series(self.training_metrics['episode_rewards'])
            rolling_std = rewards_series.rolling(window=20).std()
            axes[1, 0].plot(self.training_metrics['timesteps'], rolling_std, 
                           color='red', linewidth=2)
            axes[1, 0].set_title('Training Stability (Rolling Std)')
            axes[1, 0].set_xlabel('Training Timesteps')
            axes[1, 0].set_ylabel('Reward Standard Deviation')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Action distribution (if available)
        # This would require logging action data, which we'd need to implement
        axes[1, 1].text(0.5, 0.5, 'Action Distribution\n(Feature coming soon)', 
                        ha='center', va='center', fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].axis('off')
        
        # Model convergence indicator
        if len(self.training_metrics['policy_losses']) > 10:
            recent_losses = self.training_metrics['policy_losses'][-10:]
            convergence_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            convergence_status = "Converging" if convergence_trend < 0 else "Still Learning"
            
            axes[1, 2].plot(self.training_metrics['timesteps'][-10:], recent_losses, 
                           color='purple', linewidth=3, marker='o')
            axes[1, 2].set_title(f'Recent Convergence: {convergence_status}')
            axes[1, 2].set_xlabel('Recent Timesteps')
            axes[1, 2].set_ylabel('Policy Loss')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.dirs['performance_summary'] / f"performance_summary_{self.timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance summary saved to: {save_path}")
    
    def save_metrics_data(self):
        """Save all collected metrics to JSON and CSV files."""
        # Save as JSON
        all_metrics = {
            'training_metrics': self.training_metrics,
            'evaluation_metrics': self.evaluation_metrics,
            'timestamp': self.timestamp
        }
        
        json_path = self.session_dir / f"training_metrics_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Save as CSV
        if self.training_metrics['timesteps']:
            training_df = pd.DataFrame(self.training_metrics)
            csv_path = self.session_dir / f"training_data_{self.timestamp}.csv"
            training_df.to_csv(csv_path, index=False)
        
        if self.evaluation_metrics['eval_timesteps']:
            eval_df = pd.DataFrame(self.evaluation_metrics)
            csv_path = self.session_dir / f"evaluation_data_{self.timestamp}.csv"
            eval_df.to_csv(csv_path, index=False)
        
        logger.info(f"Metrics data saved to: {self.session_dir}")

class VisualizationCallback:
    """
    Custom callback to collect metrics and create visualizations during training.
    """
    
    def __init__(self, visualizer: TrainingVisualizer, log_freq: int = 1000):
        self.visualizer = visualizer
        self.log_freq = log_freq
        self.last_log_timestep = 0
    
    def on_step(self, model, timestep: int) -> bool:
        """Called during training steps to collect metrics."""
        # Log metrics at specified frequency
        if timestep - self.last_log_timestep >= self.log_freq:
            # Extract metrics from model (this would need to be implemented based on SB3 internals)
            # For now, we'll create a placeholder structure
            metrics = {
                'episode_rewards': getattr(model, 'episode_reward', 0),
                'episode_lengths': getattr(model, 'episode_length', 0),
                'learning_rates': float(model.learning_rate) if hasattr(model, 'learning_rate') else 0,
            }
            
            self.visualizer.log_training_step(timestep, metrics)
            self.last_log_timestep = timestep
        
        return True

def create_training_config() -> Dict[str, Any]:
    """
    Create default training configuration.
    Adjust these parameters based on your specific requirements.
    """
    return {
        # Environment configuration
        'env_config': {
            'api_endpoint': 'http://localhost:8000/predict',
            'max_steps': 1000,
            'reward_config': {
                'distance_progress': 1.0,      # Primary reward: 1 point per unit distance
                'crash_penalty': -1000.0,      # Strong crash penalty (equivalent to losing 1000 distance)
                'time_penalty': -0.01,         # Very small time penalty to encourage efficiency
                # Removed: speed_bonus, proximity_penalty (let imitation learning handle these)
            }
        },
        
        # PPO hyperparameters
        'ppo_config': {
            'learning_rate': 3e-4,
            'n_steps': 2048,               # Steps to collect before update
            'batch_size': 64,              # Batch size for training
            'n_epochs': 10,                # Number of epochs per update
            'gamma': 0.99,                 # Discount factor
            'gae_lambda': 0.95,            # GAE parameter
            'clip_range': 0.2,             # PPO clip range
            'clip_range_vf': None,         # Value function clip range
            'ent_coef': 0.01,              # Entropy coefficient
            'vf_coef': 0.5,                # Value function coefficient
            'max_grad_norm': 0.5,          # Gradient clipping
            'target_kl': None,             # Target KL divergence
            'device': 'cpu',               # Use CPU (adjust if you have CUDA)
        },
        
        # Training configuration
        'training_config': {
            'total_timesteps': 100_000,    # Total training timesteps
            'n_eval_episodes': 10,         # Episodes per evaluation
            'eval_freq': 5000,             # Evaluation frequency
            'save_freq': 10000,            # Model save frequency
            'n_envs': 4,                  # Number of parallel environments
            'use_subprocess': False,       # Use subprocess for parallel envs
        },
        
        # Visualization configuration
        'visualization_config': {
            'enable_plots': True,          # Enable visualization creation
            'plot_freq': 5000,             # Frequency to update plots
            'save_raw_data': True,         # Save metrics to CSV/JSON
            'create_summary': True,        # Create final summary plots
        },
        
        # Directory paths
        'paths': {
            'log_dir': './logs/',
            'model_dir': './models/',
            'tensorboard_log': './tensorboard_logs/',
            'visualization_dir': './visualizations/',
        }
    }

def make_env(env_config: Dict[str, Any], rank: int = 0):
    """
    Create a single environment instance.
    
    Args:
        env_config: Environment configuration
        rank: Environment rank for parallel training
    
    Returns:
        Environment creation function
    """
    def _init():
        env = make_race_car_env(env_config)
        # Wrap with Monitor for logging
        env = Monitor(env, filename=None)
        return env
    return _init

def create_vectorized_env(config: Dict[str, Any]) -> DummyVecEnv:
    """
    Create vectorized environment for parallel training.
    
    Args:
        config: Training configuration
        
    Returns:
        Vectorized environment
    """
    env_config = config['env_config']
    training_config = config['training_config']
    
    n_envs = training_config['n_envs']
    use_subprocess = training_config['use_subprocess']
    
    # Create environment functions
    env_fns = [make_env(env_config, i) for i in range(n_envs)]
    
    # Create vectorized environment
    if use_subprocess and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    
    logger.info(f"Created vectorized environment with {n_envs} parallel environments")
    return vec_env

def create_callbacks(config: Dict[str, Any], eval_env, visualizer: TrainingVisualizer = None):
    """
    Create training callbacks.
    
    Args:
        config: Training configuration
        eval_env: Evaluation environment
        visualizer: Training visualizer instance
        
    Returns:
        List of callbacks
    """
    training_config = config['training_config']
    paths = config['paths']
    
    callbacks = []
    
    # Evaluation callback with custom logging for visualization
    class VisualizationEvalCallback(EvalCallback):
        def __init__(self, *args, visualizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.visualizer = visualizer
            
        def _on_step(self) -> bool:
            result = super()._on_step()
            
            # Log evaluation metrics to visualizer
            if self.visualizer and hasattr(self, 'last_mean_reward'):
                eval_metrics = {
                    'mean_rewards': self.last_mean_reward,
                    'std_rewards': getattr(self, 'last_std_reward', 0.0),
                    # Calculate crash rate and success rate from episode info
                    'crash_rates': 0.0,  # Would need to be calculated from episode info
                    'success_rates': 100.0,  # Would need to be calculated from episode info
                    'average_distances': 0.0,  # Would need to be calculated from episode info
                }
                
                self.visualizer.log_evaluation_step(self.num_timesteps, eval_metrics)
            
            return result
    
    eval_callback = VisualizationEvalCallback(
        eval_env,
        best_model_save_path=os.path.join(paths['model_dir'], 'best_model'),
        log_path=os.path.join(paths['log_dir'], 'eval'),
        eval_freq=training_config['eval_freq'],
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False,
        visualizer=visualizer
    )
    callbacks.append(eval_callback)
    
    # Optional: Stop training when reaching reward threshold
    # stop_callback = StopTrainingOnRewardThreshold(
    #     reward_threshold=200.0,  # Adjust based on your reward scale
    #     verbose=1
    # )
    # callbacks.append(stop_callback)
    
    return callbacks

def train_ppo_agent(config: Dict[str, Any], pretrained_model_path: str = None):
    """
    Train PPO agent on the race car environment.
    
    Args:
        config: Training configuration
        pretrained_model_path: Path to pretrained model (for fine-tuning)
    """
    # Force CPU usage
    torch.set_default_device('cpu')
    
    # Initialize visualizer if enabled
    visualizer = None
    viz_config = config.get('visualization_config', {})
    if viz_config.get('enable_plots', True):
        viz_dir = config['paths'].get('visualization_dir', './visualizations/')
        visualizer = TrainingVisualizer(base_dir=viz_dir)
        logger.info("Training visualizations enabled")
    
    # Record training start time
    training_start_time = datetime.now()
    
    # Create directories
    paths = config['paths']
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Create environments
    train_env = create_vectorized_env(config)
    eval_env = DummyVecEnv([make_env(config['env_config'])])
    
    # Create callbacks with visualizer
    callbacks = create_callbacks(config, eval_env, visualizer)
    
    # Initialize or load PPO model
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        logger.info(f"Loading pretrained model from {pretrained_model_path}")
        model = PPO.load(
            pretrained_model_path,
            env=train_env,
            tensorboard_log=paths['tensorboard_log']
        )
        # Update hyperparameters if needed
        model.learning_rate = config['ppo_config']['learning_rate']
    else:
        logger.info("Creating new PPO model")
        
        # Custom policy class that uses our feature extractor
        class CustomActorCriticPolicy(ActorCriticPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs,
                               features_extractor_class=RaceCarFeaturesExtractor,
                               features_extractor_kwargs=dict(features_dim=256))
        
        model = PPO(
            CustomActorCriticPolicy,
            train_env,
            verbose=1,
            tensorboard_log=paths['tensorboard_log'],
            **config['ppo_config']
        )
    
    # Train the model
    total_timesteps = config['training_config']['total_timesteps']
    logger.info(f"Starting training for {total_timesteps} timesteps")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Calculate training duration
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    
    # Save final model
    final_model_path = os.path.join(paths['model_dir'], 'final_model')
    model.save(final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
    # Create final visualizations if enabled
    if visualizer and viz_config.get('create_summary', True):
        logger.info("Creating final training visualizations...")
        
        # Prepare final statistics
        final_stats = {
            'training_duration': str(training_duration),
            'total_timesteps': total_timesteps,
            'final_mean_reward': 0,  # Would be populated from evaluation
            'best_mean_reward': 0,   # Would be populated from evaluation
            'final_crash_rate': 0,   # Would be populated from evaluation
            'best_success_rate': 100, # Would be populated from evaluation
        }
        
        # Generate all visualization plots
        visualizer.create_training_curves()
        visualizer.create_evaluation_plots() 
        visualizer.create_performance_summary(final_stats)
        
        # Save metrics data if enabled
        if viz_config.get('save_raw_data', True):
            visualizer.save_metrics_data()
        
        logger.info(f"Training visualizations saved to: {visualizer.session_dir}")
    
    # Cleanup
    train_env.close()
    eval_env.close()
    
    return model

def evaluate_model(model_path: str, config: Dict[str, Any], n_episodes: int = 10):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to the trained model
        config: Environment configuration
        n_episodes: Number of episodes to evaluate
    """
    # Force CPU usage
    torch.set_default_device('cpu')
    
    # Create evaluation environment
    env = make_race_car_env(config['env_config'])
    
    # Load model
    model = PPO.load(model_path)
    
    logger.info(f"Evaluating model for {n_episodes} episodes")
    
    episode_rewards = []
    episode_lengths = []
    crash_count = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                if info.get('crashed', False):
                    crash_count += 1
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        logger.info(f"Episode {episode + 1}: Reward={total_reward:.2f}, "
                   f"Length={episode_length}, Distance={info.get('distance', 0):.2f}")
    
    # Print evaluation summary
    logger.info("\n=== EVALUATION SUMMARY ===")
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    logger.info(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    logger.info(f"Crash Rate: {crash_count}/{n_episodes} ({100*crash_count/n_episodes:.1f}%)")
    logger.info(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    
    env.close()

def main():
    """Main training script with enhanced visualization support."""
    # Force CPU usage globally
    torch.set_default_device('cpu')
    
    parser = argparse.ArgumentParser(description='Train PPO agent for race car environment with visualization')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                       help='Mode to run the script in')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pretrained model (for fine-tuning or evaluation)')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Override total training timesteps')
    parser.add_argument('--disable-plots', action='store_true',
                       help='Disable visualization plot generation')
    parser.add_argument('--viz-dir', type=str, default='./visualizations',
                       help='Directory for saving visualizations')
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_training_config()
    
    # Override configuration based on arguments
    if args.timesteps:
        config['training_config']['total_timesteps'] = args.timesteps
    
    if args.disable_plots:
        config['visualization_config']['enable_plots'] = False
    
    if args.viz_dir:
        config['paths']['visualization_dir'] = args.viz_dir
    
    print("=" * 60)
    print("PPO Training with Advanced Visualization")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Total timesteps: {config['training_config']['total_timesteps']:,}")
    print(f"Visualization enabled: {config['visualization_config']['enable_plots']}")
    if config['visualization_config']['enable_plots']:
        print(f"Visualization directory: {config['paths']['visualization_dir']}")
    print("=" * 60)
    
    if args.mode == 'train':
        logger.info("Starting PPO training with visualization...")
        model = train_ppo_agent(config, args.model_path)
        
        # Evaluate the trained model
        final_model_path = os.path.join(config['paths']['model_dir'], 'final_model')
        logger.info("Evaluating final trained model...")
        evaluate_model(final_model_path, config, n_episodes=5)
        
        # Print final visualization information
        if config['visualization_config']['enable_plots']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            viz_dir = Path(config['paths']['visualization_dir']) / f"training_session_{timestamp}"
            print("\n" + "=" * 60)
            print("TRAINING VISUALIZATION SUMMARY")
            print("=" * 60)
            print(f"📊 Training curves: {viz_dir / 'training_curves'}")
            print(f"📈 Evaluation metrics: {viz_dir / 'evaluation_metrics'}")
            print(f"📋 Performance summary: {viz_dir / 'performance_summary'}")
            print(f"💾 Raw data files: {viz_dir}")
            print("=" * 60)
        
    elif args.mode == 'eval':
        if not args.model_path:
            raise ValueError("Model path required for evaluation mode")
        
        logger.info("Starting model evaluation...")
        evaluate_model(args.model_path, config, n_episodes=20)

if __name__ == "__main__":
    main()
