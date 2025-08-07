#!/usr/bin/env python3
"""
Demo script to test DQN agent training with a few episodes.
"""

import sys
import os
import pygame

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from training.train_dqn import DQNTrainer
from training.hyperparameters import get_debug_config

def main():
    """Run a quick demo training."""
    print("🚀 DQN Lane Switching Agent - Quick Demo")
    print("="*50)
    
    # Use debug configuration for quick testing
    config = get_debug_config()
    
    # Override for even quicker demo
    config['training']['max_episodes'] = 5  # Just 5 episodes
    config['training']['max_episode_steps'] = 3600  # 60 seconds per episode
    config['dqn']['buffer_size'] = 1000  # Small buffer
    config['environment']['verbose'] = True  # Show game window
    
    print("📋 Demo Configuration:")
    print(f"   • Episodes: {config['training']['max_episodes']}")
    print(f"   • Max steps per episode: {config['training']['max_episode_steps']}")
    print(f"   • Verbose (game window): {config['environment']['verbose']}")
    print()
    
    # Initialize pygame
    pygame.init()
    
    try:
        # Create trainer
        trainer = DQNTrainer(config)
        
        print("🎮 Starting demo training...")
        print("   • Press Ctrl+C to stop early")
        print("   • Game window will show training progress")
        print()
        
        # Run training
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        raise
    finally:
        pygame.quit()
    
    print("\n✅ Demo completed!")
    print("   • Check logs/ directory for training logs")
    print("   • Check models/checkpoints/ for saved models")

if __name__ == "__main__":
    main()