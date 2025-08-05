import gym
from gym import spaces
import numpy as np

import pygame
from src.game.core import initialize_game_state, update_game, STATE

# Mapping of action index to string
ACTION_LIST = ["NOTHING", "ACCELERATE",
               "DECELERATE", "STEER_LEFT", "STEER_RIGHT"]


class RaceCarEnv(gym.Env):
    def __init__(self, render_mode=False, seed_value=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.seed_value = seed_value

        # Observation: 16 sensor values normalized (0–1), or 0 for no detection
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(16,), dtype=np.float32)

        # Discrete actions (5 total)
        self.action_space = spaces.Discrete(len(ACTION_LIST))

        # Initialize game state
        pygame.init()
        initialize_game_state(api_url="", seed_value=self.seed_value)
        if self.render_mode:
            self.screen = pygame.display.set_mode((1600, 1200))

    def reset(self):
        initialize_game_state(api_url="", seed_value=self.seed_value)
        obs = self._get_obs()
        return obs

    def step(self, action_idx):
        action_str = ACTION_LIST[action_idx]

        STATE.crashed = False
        update_game(action_str)

        # Reward
        reward = 1.0
        done = False
        if STATE.crashed:
            reward = -100.0
            done = True

        if STATE.ticks >= 3600:  # cap episode at 60 seconds
            done = True

        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self, mode='human'):
        if self.render_mode and self.screen:
            pygame.display.flip()

    def _get_obs(self):
        # Convert sensor values to normalized distances
        sensors = STATE.sensors
        max_dist = 1000  # Adjust based on your sensor range
        readings = [
            min(sensor.reading or 0, max_dist) / max_dist
            for sensor in sensors
        ]
        return np.array(readings, dtype=np.float32)

    def close(self):
        pygame.quit()
