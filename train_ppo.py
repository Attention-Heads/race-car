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
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                'distance_progress': 1.0,      # Reward for making progress
                'crash_penalty': -100.0,       # Penalty for crashing
                'time_penalty': -0.1,          # Small penalty per step
                'speed_bonus': 0.1,            # Bonus for maintaining speed
                'proximity_penalty': -0.5      # Penalty for getting close to obstacles
            }
        },
        
        # PPO hyperparameters
        'ppo_config': {
            'learning_rate': 3e-4,
            'n_steps': 2048,               # Steps to collect before update
            'batch_size': 64,              # Batch size for training
            'n_epochs': 10,                # Number of epochs per update
            'gamma': 0.99,                 # Discount factor
            'gae_lambda': 0.95,            # GAE lambda
            'clip_range': 0.2,             # PPO clip range
            'clip_range_vf': None,         # Value function clip range
            'ent_coef': 0.01,              # Entropy coefficient
            'vf_coef': 0.5,                # Value function coefficient
            'max_grad_norm': 0.5,          # Gradient clipping
            'target_kl': None,             # Target KL divergence
            'device': 'cpu',               # Force CPU usage
        },
        
        # Training configuration
        'training_config': {
            'total_timesteps': 100_000,     # Total training steps
            'n_eval_episodes': 10,         # Episodes for evaluation
            'eval_freq': 5000,             # Evaluation frequency
            'save_freq': 10000,            # Model save frequency
            'n_envs': 16,                  # Number of parallel environments
            'use_subprocess': False,       # Use subprocess for parallel envs
        },
        
        # Paths and logging
        'paths': {
            'log_dir': './logs/',
            'model_dir': './models/',
            'tensorboard_log': './tensorboard_logs/',
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

def create_callbacks(config: Dict[str, Any], eval_env):
    """
    Create training callbacks.
    
    Args:
        config: Training configuration
        eval_env: Evaluation environment
        
    Returns:
        List of callbacks
    """
    training_config = config['training_config']
    paths = config['paths']
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(paths['model_dir'], 'best_model'),
        log_path=os.path.join(paths['log_dir'], 'eval'),
        eval_freq=training_config['eval_freq'],
        n_eval_episodes=training_config['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    # Optional: Stop training when reaching reward threshold
    # stop_callback = StopTrainingOnRewardThreshold(
    #     reward_threshold=200.0,  # Adjust based on your reward scale
    #     verbose=1
    # )
    
    return [eval_callback]

def train_ppo_agent(config: Dict[str, Any], pretrained_model_path: str = None):
    """
    Train PPO agent on the race car environment.
    
    Args:
        config: Training configuration
        pretrained_model_path: Path to pretrained model (for fine-tuning)
    """
    # Force CPU usage
    torch.set_default_device('cpu')
    
    # Create directories
    paths = config['paths']
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Create environments
    train_env = create_vectorized_env(config)
    eval_env = DummyVecEnv([make_env(config['env_config'])])
    
    # Create callbacks
    callbacks = create_callbacks(config, eval_env)
    
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
                               features_extractor_kwargs=dict(features_dim=256),
                               net_arch=[256])  # Match the feature extractor output size
        
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
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(paths['model_dir'], 'final_model')
    model.save(final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    
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
    """Main training script."""
    # Force CPU usage globally
    torch.set_default_device('cpu')
    
    parser = argparse.ArgumentParser(description='Train PPO agent for race car environment')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train',
                       help='Mode to run the script in')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to pretrained model (for fine-tuning or evaluation)')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Override total training timesteps')
    
    args = parser.parse_args()
    
    # Load configuration
    config = create_training_config()
    
    # Override timesteps if provided
    if args.timesteps:
        config['training_config']['total_timesteps'] = args.timesteps
    
    if args.mode == 'train':
        logger.info("Starting PPO training...")
        model = train_ppo_agent(config, args.model_path)
        
        # Evaluate the trained model
        final_model_path = os.path.join(config['paths']['model_dir'], 'final_model')
        evaluate_model(final_model_path, config, n_episodes=5)
        
    elif args.mode == 'eval':
        if not args.model_path:
            raise ValueError("Model path required for evaluation mode")
        
        logger.info("Starting model evaluation...")
        evaluate_model(args.model_path, config, n_episodes=20)

if __name__ == "__main__":
    main()
