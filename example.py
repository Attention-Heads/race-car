import pygame
import random
from src.game.core import initialize_game_state, game_loop, continuous_game_loop, save_game_data


'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''

def return_action(state):
    # Returns a list of actions
    actions = []
    action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
    for _ in range(10):
        actions.append(random.choice(action_choices))
    return actions




if __name__ == '__main__':
    # For continuous data collection (recommended for imitation learning)
    pygame.init()
    continuous_game_loop(verbose=True, max_games=None)  # Set max_games to a number to limit games
    
    # For single game (uncomment below and comment above)
    # seed_value = 565318
    # pygame.init()
    # initialize_game_state("http://example.com/api/predict", seed_value)
    # game_loop(verbose=True) # For pygame window
    # save_game_data(1, seed_value)
    # pygame.quit()