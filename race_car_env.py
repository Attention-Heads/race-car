import gymnasium as gym
from gymnasium import spaces
import numpy as np
import joblib
import logging
from typing import Dict, Any, Tuple

# Import the new simulation class
from game_simulation import GameSimulation 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaceCarEnv(gym.Env):
    def __init__(self, config: Dict[str, Any] = None):
        super(RaceCarEnv, self).__init__()
        self.config = config or {}
        
        # [+] Instantiate the game simulation directly. The 'verbose' flag controls the Pygame window.
        self.game = GameSimulation(verbose=self.config.get('render', False))
        
        # --- The rest of your setup is mostly the same and very good ---
        self.feature_order = [
            'velocity_x', 'velocity_y',
            'sensor_back', 'sensor_back_left_back', 'sensor_back_right_back',
            'sensor_front', 'sensor_front_left_front', 'sensor_front_right_front',
            'sensor_left_back', 'sensor_left_front', 'sensor_left_side',
            'sensor_left_side_back', 'sensor_left_side_front',
            'sensor_right_back', 'sensor_right_front', 'sensor_right_side',
            'sensor_right_side_back', 'sensor_right_side_front'
        ]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_order),), dtype=np.float32)
        
        self.action_mapping = {0: 'NOTHING', 1: 'ACCELERATE', 2: 'DECELERATE', 3: 'STEER_LEFT', 4: 'STEER_RIGHT'}
        self.action_space = spaces.Discrete(len(self.action_mapping))
        
        try:
            self.velocity_scaler = joblib.load('velocity_scaler.pkl')
            logger.info("Loaded velocity scaler successfully.")
        except FileNotFoundError:
            self.velocity_scaler = None
            logger.warning("velocity_scaler.pkl not found. Velocity will not be scaled.")
        
        # State variables
        self.current_state_dto = None
        self.previous_distance = 0.0
        self.reward_weights = {
            'distance_progress': 1.5,
            'crash_penalty': -100.0,
            'time_penalty': -0.1,
            'speed_bonus': 0.05
        }

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
        # Your flattening logic is good, just ensure keys match
        flat_state = [state_dto['velocity']['x'], state_dto['velocity']['y']]
        
        # Process sensor readings - ensure we have valid values
        for sensor_name in self.feature_order[2:]:
            sensor_key = sensor_name.replace('sensor_', '')
            sensor_value = state_dto['sensors'].get(sensor_key, 1000.0)
            # Ensure sensor value is not None or NaN
            if sensor_value is None or np.isnan(sensor_value):
                sensor_value = 1000.0  # Max sensor range as default
            flat_state.append(float(sensor_value))
        
        state_array = np.array(flat_state, dtype=np.float32)
        
        # Normalize sensor readings
        state_array[2:] /= 1000.0
        
        # Apply velocity scaling if available
        if self.velocity_scaler:
            try:
                velocity_scaled = self.velocity_scaler.transform([state_array[:2]])[0]
                state_array[:2] = velocity_scaled
            except Exception as e:
                logger.warning(f"Velocity scaling failed: {e}")
                # Keep original velocity values if scaling fails
        
        # Final check for NaN values
        if np.isnan(state_array).any():
            logger.warning("NaN detected in state array, replacing with defaults")
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=0.0)
            
        return state_array

    def _calculate_reward(self, current_dto: Dict) -> float:
        reward = 0.0
        # Reward for distance covered
        distance_delta = current_dto['distance'] - self.previous_distance
        reward += distance_delta * self.reward_weights['distance_progress']
        self.previous_distance = current_dto['distance']
        
        # Penalty for crashing
        if current_dto['did_crash']:
            reward += self.reward_weights['crash_penalty']
        
        # Small penalty per tick to encourage finishing faster
        reward += self.reward_weights['time_penalty']

        # Small bonus for speed
        speed = np.linalg.norm([current_dto['velocity']['x'], current_dto['velocity']['y']])
        reward += speed * self.reward_weights['speed_bonus']
        
        return reward
        
    def render(self, mode='human'):
        # The game simulation handles its own rendering if verbose=True
        pass

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