#!/usr/bin/env python3
"""
Test script to verify both CPU and GPU training work correctly.
"""

import sys
import os
import torch
import traceback

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_device_compatibility():
    """Test DQN agent on both CPU and GPU if available."""
    print("🧪 Testing Device Compatibility")
    print("=" * 50)
    
    # Check available devices
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    print()
    
    devices_to_test = ['cpu']
    if torch.cuda.is_available():
        devices_to_test.append('cuda')
    
    success_count = 0
    
    for device in devices_to_test:
        print(f"🔧 Testing device: {device}")
        
        try:
            from src.agents.dqn_agent import DQNAgent
            from src.mathematics.vector import Vector
            
            # Create agent with specific device
            agent = DQNAgent(
                state_size=117,
                action_size=3,
                learning_rate=0.001,
                buffer_size=1000,
                batch_size=16,
                device=device
            )
            
            print(f"   ✅ Agent created on {device}")
            print(f"   • Network device: {next(agent.trainer.q_network.parameters()).device}")
            print(f"   • Target network device: {next(agent.trainer.target_network.parameters()).device}")
            
            # Test action selection
            dummy_sensors = {
                'front': 500.0, 'left_side': 800.0, 'right_side': 900.0, 'back': 600.0,
                'left_front': 700.0, 'right_front': 750.0, 'left_back': 850.0, 'right_back': 880.0,
                'left_side_front': 920.0, 'right_side_front': 970.0, 'front_left_front': 650.0,
                'front_right_front': 720.0, 'left_side_back': 870.0, 'right_side_back': 940.0,
                'back_left_back': 830.0, 'back_right_back': 860.0
            }
            
            # Get action (this tests tensor device placement)
            action = agent.get_action(
                sensor_data=dummy_sensors,
                velocity=Vector(10.0, 0.0),
                distance=100.0,
                current_tick=1,
                training=True
            )
            
            print(f"   ✅ Action selection working: {action}")
            
            # Test training step (add some experiences first)
            for i in range(5):
                agent.step(
                    sensor_data=dummy_sensors,
                    velocity=Vector(10.0, 0.0),
                    distance=100.0 + i * 10,
                    current_tick=i + 1,
                    crashed=False
                )
            
            print(f"   ✅ Training steps working")
            print(f"   • Buffer size: {agent.replay_buffer.size()}")
            print(f"   • Epsilon: {agent.epsilon:.3f}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed on {device}: {e}")
            print(f"   Stack trace:")
            traceback.print_exc()
        
        print()
    
    print("=" * 50)
    print(f"📊 Device Tests: {success_count}/{len(devices_to_test)} passed")
    
    if success_count == len(devices_to_test):
        print("🎉 All devices working correctly!")
        return True
    else:
        print("⚠️  Some device tests failed")
        return False

def test_training_loop_short():
    """Test a very short training loop on available device."""
    print("\n🏃 Testing Short Training Loop")
    print("=" * 30)
    
    try:
        from training.train_dqn import DQNTrainer
        from training.hyperparameters import get_debug_config
        import pygame
        
        # Create minimal config
        config = get_debug_config()
        config['training']['max_episodes'] = 2  # Just 2 episodes
        config['training']['max_episode_steps'] = 60  # 1 second each
        config['environment']['verbose'] = False  # No window
        config['dqn']['buffer_size'] = 100  # Tiny buffer
        
        print(f"   • Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"   • Episodes: {config['training']['max_episodes']}")
        print(f"   • Steps per episode: {config['training']['max_episode_steps']}")
        
        pygame.init()
        
        try:
            trainer = DQNTrainer(config)
            print("   ✅ Trainer created")
            
            # Run just one episode
            episode_stats = trainer._run_episode(0)
            print(f"   ✅ Episode completed: {episode_stats}")
            print(f"   • Distance: {episode_stats['distance']:.1f}")
            print(f"   • Crashed: {episode_stats['crashed']}")
            print(f"   • Steps: {episode_stats['steps']}")
            
        finally:
            pygame.quit()
        
        print("🎉 Short training loop successful!")
        return True
        
    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all device compatibility tests."""
    print("🚀 Device Compatibility Test Suite")
    
    tests_passed = 0
    total_tests = 2
    
    if test_device_compatibility():
        tests_passed += 1
    
    if test_training_loop_short():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Final Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All device compatibility tests passed!")
        print("   → Training should work on both CPU and GPU")
        return True
    else:
        print("⚠️  Some device tests failed")
        print("   → Check issues before running full training")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)