#!/usr/bin/env python3
"""
Test script to verify the PPO API is working correctly.
Run this to test your API before running the full game demo.
"""

import requests
import json
import time
import subprocess
import sys

def test_api():
    """Test the API with sample data."""
    api_url = "http://localhost:9052/predict"
    
    # Sample request data (similar to what the game would send)
    test_data = {
        "did_crash": False,
        "elapsed_ticks": 100,
        "distance": 500.0,
        "velocity": {"x": 10.0, "y": 0.0},
        "sensors": {
            "front": 800.0,
            "right_front": 1000.0,
            "right_side": 1000.0,
            "right_back": 1000.0,
            "back": 1000.0,
            "left_back": 1000.0,
            "left_side": 1000.0,
            "left_front": 1000.0,
            "left_side_front": 1000.0,
            "front_left_front": 1000.0,
            "front_right_front": 1000.0,
            "right_side_front": 1000.0,
            "right_side_back": 1000.0,
            "back_right_back": 1000.0,
            "back_left_back": 1000.0,
            "left_side_back": 1000.0
        }
    }
    
    try:
        print("🧪 Testing API with sample data...")
        response = requests.post(api_url, json=test_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Check if we got a valid action
            if 'actions' in result and result['actions']:
                action = result['actions'][0]
                valid_actions = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
                if action in valid_actions:
                    print(f"✅ Got valid action: {action}")
                    return True
                else:
                    print(f"❌ Invalid action received: {action}")
                    return False
            else:
                print("❌ No actions in response")
                return False
        else:
            print(f"❌ API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def start_api_for_test():
    """Start API server for testing."""
    print("Starting API server for testing...")
    try:
        process = subprocess.Popen([
            sys.executable, "api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give server time to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ API server started")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start API server")
            print("STDOUT:", stdout.decode())
            print("STDERR:", stderr.decode())
            return None
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        return None

def main():
    print("🧪 PPO API Test")
    print("=" * 30)
    
    # Try to test the API (might already be running)
    if test_api():
        print("\n✅ API is already running and working!")
        return 0
    
    # If not running, start it
    print("\n🚀 Starting API server...")
    api_process = start_api_for_test()
    
    if not api_process:
        print("Failed to start API server.")
        return 1
    
    try:
        # Test the API
        if test_api():
            print("\n✅ All tests passed! Your PPO API is ready.")
            return 0
        else:
            print("\n❌ API tests failed.")
            return 1
    finally:
        # Clean up
        if api_process:
            try:
                api_process.terminate()
                api_process.wait(timeout=5)
                print("🧹 API server stopped")
            except:
                api_process.kill()

if __name__ == "__main__":
    sys.exit(main())
