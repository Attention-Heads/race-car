#!/usr/bin/env python3
"""
Test script to verify API compatibility with validation backend.
"""

import sys
import os
import json
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def test_original_api():
    """Test the original API endpoint."""
    print("🧪 Testing original API...")
    
    # Import original API
    from api import app
    client = TestClient(app)
    
    # Sample request matching validation backend format
    request_data = {
        "did_crash": False,
        "elapsed_ticks": 100,
        "distance": 500.0,
        "velocity": {"x": 10.0, "y": 0.0},
        "sensors": {
            "front": 600.0,
            "back": 800.0,
            "left_side": 900.0,
            "right_side": 950.0,
            "left_front": 700.0,
            "right_front": 750.0,
            "left_back": 850.0,
            "right_back": 880.0,
            "left_side_front": 920.0,
            "right_side_front": 970.0,
            "front_left_front": 650.0,
            "front_right_front": 720.0,
            "left_side_back": 870.0,
            "right_side_back": 940.0,
            "back_left_back": 830.0,
            "back_right_back": 860.0
        }
    }
    
    try:
        response = client.post("/predict", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Original API working")
            print(f"   Response: {result}")
            print(f"   Actions type: {type(result.get('actions'))}")
            print(f"   Actions: {result.get('actions')}")
            return True
        else:
            print(f"❌ Original API failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Original API error: {e}")
        return False

def test_dqn_api():
    """Test the DQN API endpoint."""
    print("\n🤖 Testing DQN API...")
    
    # Check if DQN API file exists
    if not os.path.exists('api_dqn.py'):
        print("❌ DQN API file not found")
        return False
    
    try:
        # Import DQN API
        from api_dqn import app
        client = TestClient(app)
        
        # Same request as original API
        request_data = {
            "did_crash": False,
            "elapsed_ticks": 100,
            "distance": 500.0,
            "velocity": {"x": 10.0, "y": 0.0},
            "sensors": {
                "front": 600.0,
                "back": 800.0,
                "left_side": 900.0,
                "right_side": 950.0,
                "left_front": 700.0,
                "right_front": 750.0,
                "left_back": 850.0,
                "right_back": 880.0,
                "left_side_front": 920.0,
                "right_side_front": 970.0,
                "front_left_front": 650.0,
                "front_right_front": 720.0,
                "left_side_back": 870.0,
                "right_side_back": 940.0,
                "back_left_back": 830.0,
                "back_right_back": 860.0
            }
        }
        
        response = client.post("/predict", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ DQN API working")
            print(f"   Response: {result}")
            print(f"   Actions type: {type(result.get('actions'))}")
            print(f"   Actions: {result.get('actions')}")
            
            # Verify response format matches original
            if 'actions' in result and isinstance(result['actions'], list):
                if all(isinstance(action, str) for action in result['actions']):
                    print("✅ Response format compatible with original")
                    return True
                else:
                    print("❌ Action types not all strings")
                    return False
            else:
                print("❌ Response format incompatible")
                return False
                
        else:
            print(f"❌ DQN API failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ DQN API error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_action_compatibility():
    """Test that actions are in the correct format."""
    print("\n🎯 Testing action compatibility...")
    
    valid_actions = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
    
    try:
        from src.game.agent import get_action_from_rule_based_agent
        
        # Test with various sensor scenarios
        test_cases = [
            {"front": 400},  # Should DECELERATE
            {"front": 1000, "back": 300},  # Should ACCELERATE
            {"front": 1000, "back": 600},  # Should DECELERATE
            {},  # Empty sensors
        ]
        
        for i, sensors in enumerate(test_cases):
            actions = get_action_from_rule_based_agent(sensors)
            
            print(f"   Test case {i+1}: {sensors} → {actions}")
            
            # Verify format
            if not isinstance(actions, list):
                print(f"❌ Actions not a list: {type(actions)}")
                return False
            
            for action in actions:
                if not isinstance(action, str):
                    print(f"❌ Action not a string: {type(action)}")
                    return False
                
                if action not in valid_actions:
                    print(f"❌ Invalid action: {action}")
                    return False
        
        print("✅ All actions are valid and compatible")
        return True
        
    except Exception as e:
        print(f"❌ Action compatibility test failed: {e}")
        return False

def main():
    """Run all compatibility tests."""
    print("🚀 API Compatibility Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_original_api():
        tests_passed += 1
    
    if test_dqn_api():
        tests_passed += 1
    
    if test_action_compatibility():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All compatibility tests passed!")
        print("   → DQN API is compatible with validation backend")
        return True
    else:
        print("⚠️  Some compatibility tests failed")
        print("   → Fix issues before connecting to validation backend")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)