#!/usr/bin/env python3
"""
Demo script to run your PPO model (initialized with BC) to play the race car game.

This script will:
1. Start the API server with your trained PPO model
2. Run the game where your model controls the car
3. You can watch your AI play!

Usage:
    python run_ppo_demo.py
"""

import subprocess
import time
import sys
import os
import signal
import pygame

def start_api_server():
    """Start the FastAPI server with the PPO model."""
    print("Starting API server with PPO model...")
    try:
        # Start the API server in the background
        process = subprocess.Popen([
            sys.executable, "api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give the server time to start
        time.sleep(3)
        
        # Check if the process is still running
        if process.poll() is None:
            print("✅ API server started successfully on http://localhost:8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print("❌ Failed to start API server")
            print("STDOUT:", stdout.decode())
            print("STDERR:", stderr.decode())
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def run_game():
    """Run the game with the API-controlled car."""
    print("Starting the race car game...")
    print("Your PPO model will control the car!")
    print("Close the game window or press ESC to exit.")
    
    try:
        # Import here to avoid import issues if pygame isn't ready
        import asyncio
        from src.game.core import initialize_game_state, game_loop, save_game_data
        
        # Initialize pygame
        pygame.init()
        
        # Set up the game with a fixed seed for reproducible results
        seed_value = 565318
        api_url = "http://localhost:8000/predict"
        
        print(f"Initializing game with seed: {seed_value}")
        initialize_game_state(api_url, seed_value)
        
        # Run the game
        print("🚗 Game started! Watch your AI drive...")
        asyncio.run(game_loop(verbose=True))
        
        # Save the game data
        save_game_data(1, seed_value)
        
        print("✅ Game completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Game interrupted by user")
    except Exception as e:
        print(f"❌ Error running game: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()

def main():
    """Main function to orchestrate the demo."""
    print("🏁 PPO Race Car Demo")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        "models/ppo_initialized_with_bc.zip",
        "velocity_scaler.pkl",
        "api.py",
        "src/game/core.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease ensure all required files are present.")
        return 1
    
    # Start the API server
    api_process = start_api_server()
    if not api_process:
        print("Failed to start API server. Exiting.")
        return 1
    
    try:
        # Run the game
        run_game()
        
    finally:
        # Clean up: terminate the API server
        print("\n🧹 Cleaning up...")
        if api_process:
            try:
                api_process.terminate()
                api_process.wait(timeout=5)
                print("✅ API server stopped")
            except subprocess.TimeoutExpired:
                api_process.kill()
                print("⚠️ API server forcefully stopped")
            except Exception as e:
                print(f"⚠️ Error stopping API server: {e}")
    
    print("👋 Demo completed!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        sys.exit(1)
