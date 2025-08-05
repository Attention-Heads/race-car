import pygame
import random
import asyncio
from src.game.core import initialize_game_state, game_loop, continuous_game_loop, save_game_data


'''
Set seed_value to None for random seed.
This example now uses the API to get actions from your trained PPO model.
Make sure to start the API server (api.py) before running this.
'''

def return_action(state):
    # Returns a list of actions - this function is no longer used when using the API
    actions = []
    action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
    for _ in range(10):
        actions.append(random.choice(action_choices))
    return actions


if __name__ == '__main__':
    # For watching your trained model play (single game)
    seed_value = "565318"  # Fixed seed for consistent testing (string required)
    pygame.init()
    
    # Initialize game with API URL pointing to your model server
    api_url = "http://127.0.0.1:8000/predict"  # Changed to http instead of https
    print(f"Initializing game with API URL: {api_url}")
    initialize_game_state(api_url, seed_value)
    
    # Run the game loop - your model will control the car
    asyncio.run(game_loop(verbose=True))  # For pygame window
    # save_game_data(1, seed_value)
    pygame.quit()
    
    # # For continuous data collection (uncomment below and comment above)
    # pygame.init()
    # asyncio.run(continuous_game_loop(verbose=True, max_games=None))  # Set max_games to a number to limit games