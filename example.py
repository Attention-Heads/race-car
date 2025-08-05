import pygame
import random
import numpy as np
from stable_baselines3 import PPO
from src.game.core import initialize_game_state, game_loop
from src.game.racecar_env import RaceCarEnv  # Custom gym-compatible wrapper

# Load the trained PPO model
model = PPO.load("ppo_racecar")  # Adjust path as needed


def get_model_action(state):
    """
    Convert the raw state (e.g., game observation) into model-compatible input,
    then use PPO model to predict the next action.
    """
    if isinstance(state, list):
        state = np.array(state, dtype=np.float32)

    action, _ = model.predict(state, deterministic=True)

    # Wrap the action in a list of actions for 10 cars (only one agent is controlled for now)
    actions = [action] + [random.randint(0, 4) for _ in range(9)]
    return actions


if __name__ == '__main__':
    seed_value = None
    pygame.init()

    # Remove remote API; we use local model
    initialize_game_state(get_model_action, seed_value)

    # Run local testing with rendering
    game_loop(verbose=True)

    pygame.quit()
