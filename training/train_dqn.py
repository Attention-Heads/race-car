"""
Main training script for DQN lane switching agent.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import json
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agents.dqn_agent import DQNAgent
from src.agents.training_env import TrainingEnvironment
from training.hyperparameters import get_default_config, get_debug_config, get_fast_config
import pygame


class DQNTrainer:
    """
    Main trainer class for DQN lane switching agent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DQN trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.training_config = config['training']
        self.dqn_config = config['dqn']
        self.logging_config = config['logging']
        
        # Create directories
        self._create_directories()
        
        # Initialize environment
        self.env = TrainingEnvironment(config)
        
        # Initialize DQN agent
        self.agent = DQNAgent(**self.dqn_config)
        
        # Connect agent to environment
        self.env.set_dqn_agent(self.agent)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_crashes = []
        self.episode_steps = []
        self.training_losses = []
        
        # Best performance tracking
        self.best_distance = 0.0
        self.best_episode = 0
        
        # Training state
        self.start_time = time.time()
        self.episodes_trained = 0
        
    def _create_directories(self):
        """Create necessary directories for logging and model saving."""
        dirs = [
            self.logging_config['log_dir'],
            self.logging_config['model_dir'],
            self.logging_config['tensorboard_dir']
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def train(self):
        """
        Main training loop.
        """
        print("🚀 Starting DQN Lane Switching Agent Training")
        print(f"📋 Configuration:")
        print(f"   • Episodes: {self.training_config['max_episodes']}")
        print(f"   • Max steps per episode: {self.training_config['max_episode_steps']}")
        print(f"   • Buffer size: {self.dqn_config['buffer_size']}")
        print(f"   • Learning rate: {self.dqn_config['learning_rate']}")
        print(f"   • Target update freq: {self.dqn_config['target_update_frequency']}")
        print()
        
        try:
            for episode in range(self.training_config['max_episodes']):
                episode_start_time = time.time()
                
                # Run episode
                episode_stats = self._run_episode(episode)
                
                # Update statistics
                self._update_statistics(episode_stats)
                
                # Logging
                if episode % self.training_config['log_frequency'] == 0:
                    self._log_progress(episode, episode_start_time)
                
                # Save model
                if episode % self.training_config['save_frequency'] == 0:
                    self._save_model(episode)
                
                # Evaluation
                if episode % self.training_config['eval_frequency'] == 0:
                    self._evaluate_agent(episode)
                
                self.episodes_trained = episode + 1
                
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            raise
        finally:
            self._cleanup()
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Args:
            episode: Episode number
            
        Returns:
            Episode statistics dictionary
        """
        # Reset environment
        state = self.env.reset(seed_value=None)
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < self.training_config['max_episode_steps']:
            # Get action from agent
            sensor_data = self.env._get_sensor_data()
            
            action = self.agent.get_action(
                sensor_data=sensor_data,
                velocity=self.env.game_state.ego.velocity,
                distance=self.env.game_state.distance,
                current_tick=episode_steps,
                is_performing_maneuver=self.env.rule_based_agent.is_performing_maneuver,
                training=True
            )
            
            # Execute step
            next_state, reward, done, info = self.env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Early termination on crash
            if info['crashed']:
                done = True
        
        # End episode
        episode_info = self.env.end_episode()
        
        return {
            'episode': episode,
            'reward': episode_reward,
            'distance': info['distance'],
            'steps': episode_steps,
            'crashed': info['crashed'],
            'final_velocity': info['velocity'],
            'epsilon': self.agent.epsilon
        }
    
    def _update_statistics(self, stats: Dict[str, Any]):
        """Update training statistics."""
        self.episode_rewards.append(stats['reward'])
        self.episode_distances.append(stats['distance'])
        self.episode_crashes.append(1 if stats['crashed'] else 0)
        self.episode_steps.append(stats['steps'])
        
        # Track best performance
        if stats['distance'] > self.best_distance:
            self.best_distance = stats['distance']
            self.best_episode = stats['episode']
    
    def _log_progress(self, episode: int, episode_start_time: float):
        """Log training progress."""
        episode_time = time.time() - episode_start_time
        total_time = time.time() - self.start_time
        
        # Calculate averages over last 100 episodes
        recent_rewards = self.episode_rewards[-100:]
        recent_distances = self.episode_distances[-100:]
        recent_crashes = self.episode_crashes[-100:]
        
        avg_reward = np.mean(recent_rewards)
        avg_distance = np.mean(recent_distances)
        crash_rate = np.mean(recent_crashes)
        
        print(f"Episode {episode:5d} | "
              f"Reward: {avg_reward:8.1f} | "
              f"Distance: {avg_distance:8.1f} | "
              f"Crashes: {crash_rate:5.1%} | "
              f"ε: {self.agent.epsilon:.3f} | "
              f"Buffer: {self.agent.replay_buffer.size():6d} | "
              f"Time: {episode_time:.2f}s")
        
        # Log to file
        log_entry = {
            'episode': episode,
            'avg_reward': float(avg_reward),
            'avg_distance': float(avg_distance),
            'crash_rate': float(crash_rate),
            'epsilon': float(self.agent.epsilon),
            'buffer_size': self.agent.replay_buffer.size(),
            'best_distance': float(self.best_distance),
            'total_time': float(total_time)
        }
        
        log_path = os.path.join(self.logging_config['log_dir'], 'training_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _save_model(self, episode: int):
        """Save model checkpoint."""
        model_path = os.path.join(
            self.logging_config['model_dir'], 
            f'dqn_episode_{episode}.pth'
        )
        self.agent.save_model(model_path)
        
        # Save latest model
        latest_path = os.path.join(
            self.logging_config['model_dir'], 
            'dqn_latest.pth'
        )
        self.agent.save_model(latest_path)
        
        print(f"💾 Model saved: {model_path}")
    
    def _evaluate_agent(self, episode: int):
        """Evaluate agent performance."""
        print(f"🔍 Evaluating agent at episode {episode}...")
        
        # Set agent to evaluation mode
        self.agent.set_eval_mode()
        
        eval_rewards = []
        eval_distances = []
        eval_crashes = []
        
        for eval_episode in range(self.config['evaluation']['num_eval_episodes']):
            # Run evaluation episode
            state = self.env.reset(seed_value=f"eval_{eval_episode}")
            
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.training_config['max_episode_steps']:
                sensor_data = self.env._get_sensor_data()
                
                action = self.agent.get_action(
                    sensor_data=sensor_data,
                    velocity=self.env.game_state.ego.velocity,
                    distance=self.env.game_state.distance,
                    current_tick=episode_steps,
                    is_performing_maneuver=self.env.rule_based_agent.is_performing_maneuver,
                    training=False  # No exploration during evaluation
                )
                
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1
                
                if info['crashed']:
                    done = True
            
            eval_rewards.append(episode_reward)
            eval_distances.append(info['distance'])
            eval_crashes.append(1 if info['crashed'] else 0)
        
        # Calculate evaluation metrics
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_distance = np.mean(eval_distances)
        eval_crash_rate = np.mean(eval_crashes)
        
        print(f"📊 Evaluation Results:")
        print(f"   • Average Reward: {avg_eval_reward:.1f}")
        print(f"   • Average Distance: {avg_eval_distance:.1f}")
        print(f"   • Crash Rate: {eval_crash_rate:.1%}")
        print()
        
        # Set agent back to training mode
        self.agent.set_train_mode()
        
        # Log evaluation results
        eval_log = {
            'episode': episode,
            'eval_avg_reward': float(avg_eval_reward),
            'eval_avg_distance': float(avg_eval_distance),
            'eval_crash_rate': float(eval_crash_rate)
        }
        
        eval_path = os.path.join(self.logging_config['log_dir'], 'evaluation_log.jsonl')
        with open(eval_path, 'a') as f:
            f.write(json.dumps(eval_log) + '\n')
    
    def _cleanup(self):
        """Cleanup resources and save final results."""
        print(f"\n🏁 Training completed!")
        print(f"   • Episodes trained: {self.episodes_trained}")
        print(f"   • Best distance: {self.best_distance:.1f} (episode {self.best_episode})")
        print(f"   • Total training time: {(time.time() - self.start_time) / 3600:.2f} hours")
        
        # Save final model
        final_model_path = os.path.join(
            self.logging_config['model_dir'], 
            'dqn_final.pth'
        )
        self.agent.save_model(final_model_path)
        
        # Create training plots
        self._create_training_plots()
        
        # Close environment
        self.env.close()
    
    def _create_training_plots(self):
        """Create training progress plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode distances
        axes[0, 1].plot(self.episode_distances)
        axes[0, 1].set_title('Episode Distances')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Distance')
        
        # Crash rate (rolling average)
        window_size = 100
        if len(self.episode_crashes) >= window_size:
            crash_rate = np.convolve(self.episode_crashes, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(crash_rate)
        axes[1, 0].set_title('Crash Rate (100-episode rolling avg)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Crash Rate')
        
        # Episode steps
        axes[1, 1].plot(self.episode_steps)
        axes[1, 1].set_title('Episode Steps')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        
        plt.tight_layout()
        plot_path = os.path.join(self.logging_config['log_dir'], 'training_progress.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"📈 Training plots saved: {plot_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train DQN Lane Switching Agent')
    parser.add_argument('--config', choices=['default', 'debug', 'fast'], 
                       default='default', help='Configuration preset')
    parser.add_argument('--episodes', type=int, help='Number of episodes to train')
    parser.add_argument('--verbose', action='store_true', help='Show pygame window')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'debug':
        config = get_debug_config()
    elif args.config == 'fast':
        config = get_fast_config()
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.episodes:
        config['training']['max_episodes'] = args.episodes
    if args.verbose:
        config['environment']['verbose'] = True
    
    # Initialize pygame
    pygame.init()
    
    try:
        # Create and run trainer
        trainer = DQNTrainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()