"""
Test script for the Race Car Gymnasium Environment

This script validates that the environment works correctly and provides
debugging information to help with integration.
"""

import numpy as np
import sys
import traceback
from race_car_env import make_race_car_env, RaceCarEnv
from dtos import RaceCarPredictRequestDto
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dto_creation():
    """Test that we can create and process DTOs correctly."""
    logger.info("Testing DTO creation and processing...")
    
    # Create a sample DTO
    sample_dto = RaceCarPredictRequestDto(
        did_crash=False,
        elapsed_ticks=100,
        distance=150.5,
        velocity={'x': 5.2, 'y': -1.3},
        sensors={
            'back': 500.0,
            'back_left_back': 300.0,
            'back_right_back': 450.0,
            'front': 200.0,
            'front_left_front': 180.0,
            'front_right_front': 220.0,
            'left_back': 400.0,
            'left_front': 150.0,
            'left_side': 120.0,
            'left_side_back': 350.0,
            'left_side_front': 100.0,
            'right_back': 380.0,
            'right_front': 160.0,
            'right_side': 130.0,
            'right_side_back': 340.0,
            'right_side_front': 110.0
        }
    )
    
    logger.info(f"Created DTO: {sample_dto}")
    return sample_dto

def test_environment_creation():
    """Test basic environment creation and configuration."""
    logger.info("Testing environment creation...")
    
    config = {
        'api_endpoint': 'http://localhost:8000/predict',
        'max_steps': 100,
        'reward_config': {
            'distance_progress': 1.0,
            'crash_penalty': -50.0,
            'time_penalty': -0.05,
            'speed_bonus': 0.1,
            'proximity_penalty': -0.2
        }
    }
    
    env = make_race_car_env(config)
    
    logger.info(f"Environment created successfully!")
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")
    logger.info(f"Feature order: {env.feature_order}")
    logger.info(f"Action mapping: {env.action_mapping}")
    
    return env

def test_state_processing(env: RaceCarEnv):
    """Test state flattening and processing."""
    logger.info("Testing state processing...")
    
    # Create test DTO
    test_dto = RaceCarPredictRequestDto(
        did_crash=False,
        elapsed_ticks=50,
        distance=75.0,
        velocity={'x': 3.0, 'y': -0.5},
        sensors={
            'back': 800.0,
            'back_left_back': 600.0,
            'back_right_back': 750.0,
            'front': 300.0,
            'front_left_front': 280.0,
            'front_right_front': 320.0,
            'left_back': 500.0,
            'left_front': 250.0,
            'left_side': 200.0,
            'left_side_back': 450.0,
            'left_side_front': 180.0,
            'right_back': 480.0,
            'right_front': 260.0,
            'right_side': 210.0,
            'right_side_back': 440.0,
            'right_side_front': 190.0
        }
    )
    
    # Process state
    processed_state = env._flatten_and_process_state(test_dto)
    
    logger.info(f"Original velocity: {test_dto.velocity}")
    logger.info(f"Original sensors (sample): front={test_dto.sensors['front']}, left_side={test_dto.sensors['left_side']}")
    logger.info(f"Processed state shape: {processed_state.shape}")
    logger.info(f"Processed state (first 5): {processed_state[:5]}")
    logger.info(f"Processed state (last 5): {processed_state[-5:]}")
    
    # Validate shape
    expected_shape = (len(env.feature_order),)
    assert processed_state.shape == expected_shape, f"Expected shape {expected_shape}, got {processed_state.shape}"
    
    logger.info("State processing test passed!")
    return processed_state

def test_reward_calculation(env: RaceCarEnv):
    """Test reward calculation logic."""
    logger.info("Testing reward calculation...")
    
    # Set up initial state
    env.previous_distance = 100.0
    
    # Test normal progress
    normal_dto = RaceCarPredictRequestDto(
        did_crash=False,
        elapsed_ticks=50,
        distance=110.0,  # 10 units progress
        velocity={'x': 5.0, 'y': 0.0},
        sensors={'front': 500.0, 'left_side': 400.0, 'right_side': 450.0}
    )
    
    reward_normal = env._calculate_reward(normal_dto, action=1)  # ACCELERATE
    logger.info(f"Normal progress reward: {reward_normal:.3f}")
    
    # Test crash scenario
    crash_dto = RaceCarPredictRequestDto(
        did_crash=True,
        elapsed_ticks=51,
        distance=110.0,
        velocity={'x': 0.0, 'y': 0.0},
        sensors={'front': 0.0, 'left_side': 0.0, 'right_side': 0.0}
    )
    
    reward_crash = env._calculate_reward(crash_dto, action=1)
    logger.info(f"Crash reward: {reward_crash:.3f}")
    
    # Test proximity penalty
    close_dto = RaceCarPredictRequestDto(
        did_crash=False,
        elapsed_ticks=52,
        distance=115.0,
        velocity={'x': 2.0, 'y': 0.0},
        sensors={'front': 50.0, 'left_side': 400.0, 'right_side': 450.0}  # Close to obstacle
    )
    
    reward_close = env._calculate_reward(close_dto, action=1)
    logger.info(f"Close to obstacle reward: {reward_close:.3f}")
    
    logger.info("Reward calculation test completed!")

def test_environment_episode():
    """Test a complete episode."""
    logger.info("Testing complete environment episode...")
    
    config = {
        'max_steps': 20,  # Short episode for testing
        'reward_config': {
            'distance_progress': 1.0,
            'crash_penalty': -100.0,
            'time_penalty': -0.1,
            'speed_bonus': 0.1,
            'proximity_penalty': -0.5
        }
    }
    
    env = make_race_car_env(config)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial info: {info}")
    
    total_reward = 0
    step_count = 0
    
    # Run episode
    while True:
        # Random action for testing
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        logger.info(f"Step {step_count}: Action={env.action_mapping[action]}, "
                   f"Reward={reward:.3f}, Total={total_reward:.3f}, "
                   f"Distance={info.get('distance', 0):.2f}")
        
        if terminated or truncated:
            logger.info(f"Episode ended: Terminated={terminated}, Truncated={truncated}")
            logger.info(f"Final info: {info}")
            break
    
    env.close()
    logger.info(f"Episode completed! Total steps: {step_count}, Total reward: {total_reward:.3f}")

def test_action_space_coverage():
    """Test that all actions are properly mapped."""
    logger.info("Testing action space coverage...")
    
    env = make_race_car_env()
    
    logger.info(f"Action space size: {env.action_space.n}")
    logger.info("Action mappings:")
    for action_id, action_name in env.action_mapping.items():
        logger.info(f"  {action_id}: {action_name}")
    
    # Test that all actions are valid
    for action_id in range(env.action_space.n):
        assert action_id in env.action_mapping, f"Action {action_id} not in mapping"
    
    logger.info("Action space test passed!")

def run_all_tests():
    """Run all test functions."""
    logger.info("=" * 50)
    logger.info("STARTING RACE CAR ENVIRONMENT TESTS")
    logger.info("=" * 50)
    
    tests = [
        ("DTO Creation", test_dto_creation),
        ("Environment Creation", test_environment_creation),
        ("Action Space Coverage", test_action_space_coverage),
    ]
    
    results = {}
    env = None
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n--- Running {test_name} ---")
            result = test_func()
            if test_name == "Environment Creation":
                env = result
            results[test_name] = "PASSED"
            logger.info(f"✓ {test_name} PASSED")
        except Exception as e:
            results[test_name] = f"FAILED: {str(e)}"
            logger.error(f"✗ {test_name} FAILED: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Tests that require an environment
    if env is not None:
        env_tests = [
            ("State Processing", lambda: test_state_processing(env)),
            ("Reward Calculation", lambda: test_reward_calculation(env)),
        ]
        
        for test_name, test_func in env_tests:
            try:
                logger.info(f"\n--- Running {test_name} ---")
                test_func()
                results[test_name] = "PASSED"
                logger.info(f"✓ {test_name} PASSED")
            except Exception as e:
                results[test_name] = f"FAILED: {str(e)}"
                logger.error(f"✗ {test_name} FAILED: {str(e)}")
                logger.error(traceback.format_exc())
    
    # Final integration test
    try:
        logger.info(f"\n--- Running Episode Integration Test ---")
        test_environment_episode()
        results["Episode Integration"] = "PASSED"
        logger.info(f"✓ Episode Integration Test PASSED")
    except Exception as e:
        results["Episode Integration"] = f"FAILED: {str(e)}"
        logger.error(f"✗ Episode Integration Test FAILED: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status} {test_name}: {result}")
        if result == "PASSED":
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed + failed}, Passed: {passed}, Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Environment is ready for PPO training.")
    else:
        logger.error(f"⚠️  {failed} test(s) failed. Please fix issues before training.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
