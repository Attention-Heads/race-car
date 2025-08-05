import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Tuple

# Import the new simulation class and preprocessing utilities
from game_simulation import GameSimulation
from preprocessing_utils import StatePreprocessor 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaceCarEnv(gym.Env):
    def __init__(self, config: Dict[str, Any] = None):
        super(RaceCarEnv, self).__init__()
        self.config = config or {}
        
        # [+] Instantiate the game simulation directly. The 'verbose' flag controls the Pygame window.
        self.game = GameSimulation(verbose=self.config.get('render', False))
        
        # Initialize centralized preprocessor
        self.preprocessor = StatePreprocessor(use_velocity_scaler=True)
        
        # Get feature order from preprocessor to ensure consistency
        self.feature_order = self.preprocessor.get_feature_order()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_order),), dtype=np.float32)
        
        self.action_mapping = {0: 'NOTHING', 1: 'ACCELERATE', 2: 'DECELERATE', 3: 'STEER_LEFT', 4: 'STEER_RIGHT'}
        self.action_space = spaces.Discrete(len(self.action_mapping))
        
        # State variables
        self.current_state_dto = None
        self.previous_distance = 0.0
        
        self.reward_weights = self.config.get('reward_config')
        
        # Log the reward configuration being used
        logger.info(f"Environment initialized with reward weights: {self.reward_weights}")

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # [+] Use the provided seed or generate a new one
        if seed is None:
            seed = np.random.randint(0, 1_000_000)
        
        logger.info(f"Resetting environment with seed: {seed}")
        
        # [+] Call the game's reset method directly
        self.current_state_dto = self.game.reset(seed_value=seed)
        self.previous_distance = self.current_state_dto['distance']
        
        observation = self._flatten_and_process_state(self.current_state_dto)
        info = {'distance': self.current_state_dto['distance'], 'crashed': self.current_state_dto['did_crash']}
        
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Convert numpy array action if needed
        if isinstance(action, np.ndarray):
            if action.ndim == 0:  # 0-dimensional array (scalar)
                action = int(action.item())
            else:  # 1 or more dimensional array
                action = int(action[0])

        # [+] Get the action name and pass it to the game simulation
        action_name = self.action_mapping[action]
        next_state_dto, terminated, truncated = self.game.step(action_name)
        
        # Calculate reward based on the outcome
        reward = self._calculate_reward(next_state_dto)
        
        # Update internal state
        self.current_state_dto = next_state_dto
        
        observation = self._flatten_and_process_state(next_state_dto)
        info = {
            'distance': next_state_dto['distance'],
            'crashed': next_state_dto['did_crash'],
            'action_taken': action_name,
            'reward': reward
        }
        
        return observation, reward, terminated, truncated, info

    def _flatten_and_process_state(self, state_dto: Dict) -> np.ndarray:
        """Use centralized preprocessing for consistency."""
        state_array = self.preprocessor.preprocess_state_dict(state_dto)
        
        # Handle any remaining NaN values as backup
        if np.isnan(state_array).any():
            logger.warning("NaN detected in state array, replacing with defaults")
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=0.0)
            
        return state_array

    def _calculate_reward(self, current_dto: Dict) -> float:
        """
        Calculate reward based on the current state.
        Uses reward configuration passed from training script for single source of truth.
        """
        reward = 0.0
        
        # Reward for distance covered
        distance_delta = current_dto['distance'] - self.previous_distance
        if 'distance_progress' in self.reward_weights:
            reward += distance_delta * self.reward_weights['distance_progress']
        self.previous_distance = current_dto['distance']
        
        # Penalty for crashing
        if current_dto['did_crash'] and 'crash_penalty' in self.reward_weights:
            reward += self.reward_weights['crash_penalty']
        
        # Small penalty per tick to encourage finishing faster
        if 'time_penalty' in self.reward_weights:
            reward += self.reward_weights['time_penalty']

        # Optional speed bonus (only if configured)
        if 'speed_bonus' in self.reward_weights and self.reward_weights['speed_bonus'] != 0:
            speed = np.linalg.norm([current_dto['velocity']['x'], current_dto['velocity']['y']])
            reward += speed * self.reward_weights['speed_bonus']
        
        # Optional proximity penalty (only if configured)
        if 'proximity_penalty' in self.reward_weights and self.reward_weights['proximity_penalty'] != 0:
            # Get the minimum distance to walls from available sensors
            sensor_distances = []
            for key, value in current_dto.get('sensors', {}).items():
                if value is not None and value > 0:  # Valid sensor reading
                    sensor_distances.append(value)
            
            if sensor_distances:
                min_distance = min(sensor_distances)
                # Apply penalty inversely proportional to distance (closer = more penalty)
                # Use a threshold to avoid penalties when far from walls
                proximity_threshold = 200.0  # Adjust based on your track scale
                if min_distance < proximity_threshold:
                    # Penalty increases as distance decreases
                    proximity_factor = (proximity_threshold - min_distance) / proximity_threshold
                    reward += proximity_factor * self.reward_weights['proximity_penalty']
        
        return reward
        
    def render(self, mode='human'):
        """
        Render the environment for evaluation or visualization.
        mode: 'human' displays the game window (if supported),
              'rgb_array' returns a frame as a numpy array (if supported).
        """
        if mode == 'human':
            # If the simulation supports explicit rendering, call it
            if hasattr(self.game, 'render'):
                self.game.render()
            # Otherwise, rely on verbose=True to show the window
        elif mode == 'rgb_array':
            # If the simulation can return a frame, do so
            if hasattr(self.game, 'get_frame'):
                return self.game.get_frame()
            else:
                raise NotImplementedError("rgb_array mode not supported by GameSimulation.")
        else:
            raise NotImplementedError(f"Render mode '{mode}' not supported.")

    def close(self):
        self.game.close()


def make_race_car_env(config: Dict[str, Any] = None) -> RaceCarEnv:
    """
    Factory function to create a RaceCarEnv instance.
    
    Args:
        config: Environment configuration dictionary
        
    Returns:
        RaceCarEnv: Configured race car environment
    """
    return RaceCarEnv(config)