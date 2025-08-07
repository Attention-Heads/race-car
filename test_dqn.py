#!/usr/bin/env python3
"""
Test script to verify DQN agent functionality and fix import issues.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test all DQN-related imports."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        print("✅ numpy imported successfully")
        
        import torch
        print(f"✅ PyTorch imported successfully (version: {torch.__version__})")
        print(f"   • CUDA available: {torch.cuda.is_available()}")
        
        # Test project imports
        from src.mathematics.vector import Vector
        print("✅ Vector class imported")
        
        from src.utils.state_processor import StateProcessor, LaneTracker
        print("✅ StateProcessor and LaneTracker imported")
        
        from src.utils.reward_calculator import RewardCalculator
        print("✅ RewardCalculator imported")
        
        from src.agents.replay_buffer import ReplayBuffer
        print("✅ ReplayBuffer imported")
        
        from src.agents.neural_network import DQNNetwork, DQNTrainer
        print("✅ DQN neural network classes imported")
        
        from src.agents.dqn_agent import DQNAgent
        print("✅ DQNAgent imported")
        
        from training.hyperparameters import get_default_config
        print("✅ Hyperparameters imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of DQN components."""
    print("\n🔬 Testing basic functionality...")
    
    try:
        from src.utils.state_processor import StateProcessor
        from src.utils.reward_calculator import RewardCalculator
        from src.agents.dqn_agent import DQNAgent
        from src.mathematics.vector import Vector
        
        # Test StateProcessor
        state_processor = StateProcessor()
        print(f"✅ StateProcessor created (state size: {state_processor.state_size})")
        
        # Test sensor data processing
        dummy_sensors = {
            'front': 500.0,
            'left_side': 800.0,
            'right_side': 900.0,
            'back': 600.0
        }
        
        state_processor.add_sensor_reading(dummy_sensors)
        print("✅ Sensor data added to StateProcessor")
        
        # Test RewardCalculator
        reward_calc = RewardCalculator()
        reward = reward_calc.calculate_reward(
            current_distance=100.0,
            velocity=Vector(10.0, 0.0),
            action=2,  # Do nothing
            crashed=False,
            sensors=dummy_sensors,
            lane=2,
            time_since_lane_change=200,
            is_performing_maneuver=False
        )
        print(f"✅ RewardCalculator working (sample reward: {reward:.2f})")
        
        # Test DQNAgent creation
        agent = DQNAgent(
            state_size=117,
            action_size=3,
            learning_rate=0.001,
            buffer_size=1000,  # Small buffer for testing
            batch_size=16
        )
        print("✅ DQNAgent created successfully")
        print(f"   • Epsilon: {agent.epsilon}")
        print(f"   • Buffer size: {agent.replay_buffer.size()}")
        
        # Test action selection (without enough sensor history)
        action = agent.get_action(
            sensor_data=dummy_sensors,
            velocity=Vector(10.0, 0.0),
            distance=100.0,
            current_tick=1,
            training=False
        )
        print(f"✅ Action selection working (action: {action})")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training environment setup."""
    print("\n🏃 Testing training setup...")
    
    try:
        from training.hyperparameters import get_debug_config
        
        # Load debug configuration
        config = get_debug_config()
        print("✅ Debug configuration loaded")
        print(f"   • Max episodes: {config['training']['max_episodes']}")
        print(f"   • Buffer size: {config['dqn']['buffer_size']}")
        
        # Test that core game components are importable
        from src.game.core import initialize_game_state, GameState
        print("✅ Game core components accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 DQN Agent Testing Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_imports():
        tests_passed += 1
    
    if test_basic_functionality():
        tests_passed += 1
    
    if test_training_setup():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! DQN agent is ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)