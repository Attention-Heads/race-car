import pygame
import random
from src.game.core import initialize_game_state, game_loop
from src.game.agent import get_action_from_rule_based_agent

'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''


def return_action(sensors):
    # Returns a list of actions
    actions = get_action_from_rule_based_agent(sensors)
    return actions


if __name__ == '__main__':
    seed_value = None
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(verbose=True)  # For pygame window
    pygame.quit()
