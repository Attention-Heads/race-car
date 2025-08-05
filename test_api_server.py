#!/usr/bin/env python3
"""
Test script to check if the API server is running and responding.
"""

import asyncio
import aiohttp
import json

async def test_api_server():
    """Test if the API server is running and responding."""
    api_url = "http://localhost:9052/predict"
    
    # Sample request data
    test_request = {
        'did_crash': False,
        'elapsed_ticks': 1,
        'distance': 10.0,
        'velocity': {'x': 10.0, 'y': 0.0},
        'sensors': {
            'front': 100.0,
            'right_front': 100.0,
            'right_side': 100.0,
            'right_back': 100.0,
            'back': 100.0,
            'left_back': 100.0,
            'left_side': 100.0,
            'left_front': 100.0,
            'left_side_front': 100.0,
            'front_left_front': 100.0,
            'front_right_front': 100.0,
            'right_side_front': 100.0,
            'right_side_back': 100.0,
            'back_right_back': 100.0,
            'back_left_back': 100.0,
            'left_side_back': 100.0
        }
    }
    
    print(f"🔍 Testing API server at {api_url}")
    print(f"📦 Sending test request: {json.dumps(test_request, indent=2)}")
    
    try:
        timeout = aiohttp.ClientTimeout(total=15)  # 15 second timeout for testing
        async with aiohttp.ClientSession(timeout=timeout) as session:
            print("⏱️  Making API call...")
            async with session.post(api_url, json=test_request) as response:
                print(f"✅ Response status: {response.status}")
                
                if response.status == 200:
                    result = await response.json()
                    print(f"📄 Response data: {json.dumps(result, indent=2)}")
                    
                    if 'actions' in result and result['actions']:
                        print(f"🎮 Predicted action: {result['actions'][0]}")
                        print("🎉 API server is working correctly!")
                        return True
                    elif 'error' in result:
                        print(f"❌ API returned error: {result['error']}")
                        return False
                    else:
                        print("⚠️  API response format unexpected")
                        return False
                else:
                    response_text = await response.text()
                    print(f"❌ API call failed with status {response.status}")
                    print(f"📄 Response body: {response_text}")
                    return False
                    
    except aiohttp.ClientConnectorError as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("💡 Make sure to start the API server first:")
        print("   python api.py")
        return False
    except asyncio.TimeoutError:
        print("⏰ Timeout: API server took longer than 15 seconds to respond")
        print("💡 This usually means:")
        print("   - The server is starting up and loading the model")
        print("   - The model is very large and takes time to load")
        print("   - The server is overloaded")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting API server test...")
    success = asyncio.run(test_api_server())
    
    if success:
        print("\n✅ API server test passed! You can now run the game.")
    else:
        print("\n❌ API server test failed. Please check the issues above.")
        print("\n🔧 To start the API server:")
        print("   1. Open a new terminal/command prompt")
        print("   2. Navigate to the race-car directory")
        print("   3. Run: python api.py")
        print("   4. Wait for the message 'Application startup complete'")
        print("   5. Then run your game")
