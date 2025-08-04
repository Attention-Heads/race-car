"""
Race Car Gymnasium Environment for PPO Fine-tuning

This environment uses your trained model for inference and serves as the "world" 
that the PPO agent interacts with during fine-tuning. The observation space matches 
the flattened, processed data your model was trained on.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import joblib
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaceCarEnv(gym.Env):
    """
    Gymnasium environment for race car training using PPO.
    
    The environment bridges between your existing game simulation and the RL agent,
    using your trained model for state transitions and reward calculation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super(RaceCarEnv, self).__init__()
        
        # Initialize configuration
        self.config = config or {}
        self.api_endpoint = self.config.get('api_endpoint', 'http://localhost:8000/predict')
        self.max_steps = self.config.get('max_steps', 1000)
        self.reward_config = self.config.get('reward_config', {})
        
        # Define consistent feature order matching your training data
        self.feature_order = [
            'velocity_x', 'velocity_y',
            'sensor_back', 'sensor_back_left_back', 'sensor_back_right_back',
            'sensor_front', 'sensor_front_left_front', 'sensor_front_right_front',
            'sensor_left_back', 'sensor_left_front', 'sensor_left_side',
            'sensor_left_side_back', 'sensor_left_side_front',
            'sensor_right_back', 'sensor_right_front', 'sensor_right_side',
            'sensor_right_side_back', 'sensor_right_side_front'
        ]
        
        # Observation space: flat vector of processed features
        obs_dim = len(self.feature_order)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Action space: discrete actions from your training data
        self.action_mapping = {
            0: 'NOTHING',
            1: 'ACCELERATE', 
            2: 'DECELERATE',
            3: 'STEER_LEFT',
            4: 'STEER_RIGHT'
        }
        self.action_space = spaces.Discrete(len(self.action_mapping))
        
        # Load the velocity scaler
        try:
            self.velocity_scaler = joblib.load('velocity_scaler.pkl')
            logger.info("Loaded velocity scaler successfully")
        except FileNotFoundError:
            logger.warning("velocity_scaler.pkl not found. Using identity scaling.")
            self.velocity_scaler = None
        
        # Environment state
        self.current_state = None
        self.step_count = 0
        self.total_distance = 0.0
        self.previous_distance = 0.0
        self.game_session_id = None
        
        # Reward configuration with sensible defaults
        self.reward_weights = {
            'distance_progress': self.reward_config.get('distance_progress', 1.0),
            'crash_penalty': self.reward_config.get('crash_penalty', -100.0),
            'time_penalty': self.reward_config.get('time_penalty', -0.1),
            'speed_bonus': self.reward_config.get('speed_bonus', 0.1),
            'collision_proximity_penalty': self.reward_config.get('proximity_penalty', -0.5)
        }
        
    def _flatten_and_process_state(self, request: RaceCarPredictRequestDto) -> np.ndarray:
        """
        Flatten the nested dictionaries into a processed state vector.
        This must match exactly how you processed data during training.
        """
        # Extract features in the defined order
        flat_state = []
        
        # Add velocity features (first 2 features)
        flat_state.extend([
            request.velocity.get('x', 0.0),
            request.velocity.get('y', 0.0)
        ])
        
        # Add sensor features (remaining 16 features)
        sensor_features = []
        for sensor_name in self.feature_order[2:]:  # Skip velocity features
            # Remove 'sensor_' prefix to match the keys in request.sensors
            sensor_key = sensor_name.replace('sensor_', '')
            sensor_value = request.sensors.get(sensor_key, 1000.0)
            sensor_features.append(sensor_value if sensor_value is not None else 1000.0)
        
        flat_state.extend(sensor_features)
        
        # Convert to numpy array
        state_array = np.array(flat_state, dtype=np.float32)
        
        # Apply the same preprocessing as during training
        # 1. Normalize sensors (divide by 1000.0)
        state_array[2:] = state_array[2:] / 1000.0
        
        # 2. Scale velocity using the loaded scaler
        if self.velocity_scaler is not None:
            velocity_scaled = self.velocity_scaler.transform([state_array[:2]])[0]
            state_array[:2] = velocity_scaled
        
        return state_array
    
    def _calculate_reward(self, 
                         current_request: RaceCarPredictRequestDto,
                         action: int) -> float:
        """
        Calculate reward based on the current state and action taken.
        """
        reward = 0.0
        
        # Distance progress reward
        distance_delta = current_request.distance - self.previous_distance
        reward += distance_delta * self.reward_weights['distance_progress']
        
        # Update previous distance for next calculation
        self.previous_distance = current_request.distance
        
        # Crash penalty
        if current_request.did_crash:
            reward += self.reward_weights['crash_penalty']
            logger.info(f"Crash detected! Applied penalty: {self.reward_weights['crash_penalty']}")
        
        # Time penalty (encourages efficiency)
        reward += self.reward_weights['time_penalty']
        
        # Speed bonus (encourages maintaining good velocity)
        velocity_magnitude = np.sqrt(
            current_request.velocity.get('x', 0)**2 + 
            current_request.velocity.get('y', 0)**2
        )
        reward += velocity_magnitude * self.reward_weights['speed_bonus']
        
        # Proximity penalty for getting too close to obstacles
        min_sensor_distance = min([
            v for v in current_request.sensors.values() 
            if v is not None
        ], default=1000.0)
        
        if min_sensor_distance < 100.0:  # Close to obstacles
            proximity_penalty = (100.0 - min_sensor_distance) / 100.0
            reward += proximity_penalty * self.reward_weights['collision_proximity_penalty']
        
        return reward
    
    def _is_terminal(self, request: RaceCarPredictRequestDto) -> Tuple[bool, bool]:
        """
        Determine if the episode should terminate.
        Returns (terminated, truncated)
        """
        # Terminated: episode ends naturally (crash)
        terminated = request.did_crash
        
        # Truncated: episode ends due to time limit or other constraints
        truncated = self.step_count >= self.max_steps
        
        return terminated, truncated
    
    def _make_api_request(self, action: int) -> RaceCarPredictRequestDto:
        """
        Make API request to the game simulation with the given action.
        This is where you would integrate with your actual game/simulation.
        """
        action_name = self.action_mapping[action]
        
        try:
            # Prepare request payload
            payload = {
                'action': action_name,
                'game_session_id': self.game_session_id
            }
            
            # Make API request to your game simulation
            response = requests.post(self.api_endpoint, json=payload, timeout=5.0)
            response.raise_for_status()
            
            # Parse response into DTO
            response_data = response.json()
            return RaceCarPredictRequestDto(**response_data)
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            # Return a default crash state if API fails
            return RaceCarPredictRequestDto(
                did_crash=True,
                elapsed_ticks=self.step_count,
                distance=self.total_distance,
                velocity={'x': 0.0, 'y': 0.0},
                sensors={sensor.replace('sensor_', ''): 0.0 for sensor in self.feature_order[2:]}
            )
    
    def _create_mock_state(self) -> RaceCarPredictRequestDto:
        """
        Create a mock state for testing when no API is available.
        Remove this method when integrating with real game simulation.
        """
        # Generate somewhat realistic mock data
        return RaceCarPredictRequestDto(
            did_crash=False,
            elapsed_ticks=self.step_count,
            distance=self.total_distance + np.random.uniform(0, 5),
            velocity={
                'x': np.random.uniform(-10, 10),
                'y': np.random.uniform(-10, 10)
            },
            sensors={
                sensor.replace('sensor_', ''): np.random.uniform(50, 1000)
                for sensor in self.feature_order[2:]
            }
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment and return the initial observation.
        """
        super().reset(seed=seed)
        
        # Reset environment state
        self.step_count = 0
        self.total_distance = 0.0
        self.previous_distance = 0.0
        self.game_session_id = f"session_{np.random.randint(1000000)}"
        
        logger.info(f"Resetting environment with session ID: {self.game_session_id}")
        
        # Get initial state from game (or create mock state for testing)
        try:
            # Try to initialize a new game session via API
            init_payload = {
                'action': 'RESET',
                'game_session_id': self.game_session_id,
                'seed': seed
            }
            response = requests.post(self.api_endpoint, json=init_payload, timeout=5.0)
            if response.status_code == 200:
                initial_request = RaceCarPredictRequestDto(**response.json())
            else:
                raise requests.RequestException("API not available")
        except:
            logger.warning("API not available, using mock state for testing")
            initial_request = self._create_mock_state()
        
        self.current_state = initial_request
        self.previous_distance = initial_request.distance
        
        # Process state into observation
        observation = self._flatten_and_process_state(initial_request)
        
        info = {
            'distance': initial_request.distance,
            'crashed': initial_request.did_crash,
            'elapsed_ticks': initial_request.elapsed_ticks
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute an action and return the result.
        """
        if self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")
        
        self.step_count += 1
        
        # Execute action in the game simulation
        try:
            next_state = self._make_api_request(action)
        except:
            logger.warning("Using mock state due to API unavailability")
            next_state = self._create_mock_state()
            # Simulate some progression
            if action == 1:  # ACCELERATE
                next_state.distance = self.current_state.distance + np.random.uniform(2, 5)
            elif action in [3, 4]:  # STEER
                next_state.distance = self.current_state.distance + np.random.uniform(0, 2)
            else:
                next_state.distance = self.current_state.distance + np.random.uniform(0, 1)
        
        # Calculate reward
        reward = self._calculate_reward(next_state, action)
        
        # Check termination conditions
        terminated, truncated = self._is_terminal(next_state)
        
        # Update state
        self.current_state = next_state
        self.total_distance = next_state.distance
        
        # Process state into observation
        observation = self._flatten_and_process_state(next_state)
        
        # Create info dictionary
        info = {
            'distance': next_state.distance,
            'crashed': next_state.did_crash,
            'elapsed_ticks': next_state.elapsed_ticks,
            'action_taken': self.action_mapping[action],
            'reward_components': {
                'distance_delta': next_state.distance - self.previous_distance,
                'crash_penalty': self.reward_weights['crash_penalty'] if next_state.did_crash else 0,
                'total_reward': reward
            }
        }
        
        if terminated or truncated:
            logger.info(f"Episode ended. Steps: {self.step_count}, Distance: {self.total_distance:.2f}, "
                       f"Crashed: {next_state.did_crash}")
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment. 
        For now, just print current state info.
        """
        if self.current_state is None:
            return None
            
        if mode == 'human':
            print(f"Step: {self.step_count}, Distance: {self.current_state.distance:.2f}, "
                  f"Crashed: {self.current_state.did_crash}, "
                  f"Velocity: ({self.current_state.velocity.get('x', 0):.1f}, "
                  f"{self.current_state.velocity.get('y', 0):.1f})")
        
        return None
    
    def close(self):
        """Clean up environment resources."""
        pass


# Factory function for creating the environment
def make_race_car_env(config: Optional[Dict[str, Any]] = None) -> RaceCarEnv:
    """
    Factory function to create a RaceCarEnv instance.
    
    Args:
        config: Configuration dictionary with keys:
            - api_endpoint: URL to game simulation API
            - max_steps: Maximum steps per episode
            - reward_config: Dictionary of reward weights
    
    Returns:
        Configured RaceCarEnv instance
    """
    return RaceCarEnv(config)


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'api_endpoint': 'http://localhost:8000/predict',
        'max_steps': 500,
        'reward_config': {
            'distance_progress': 1.0,
            'crash_penalty': -100.0,
            'time_penalty': -0.1,
            'speed_bonus': 0.1,
            'proximity_penalty': -0.5
        }
    }
    
    # Create and test environment
    env = make_race_car_env(test_config)
    
    print("Testing Race Car Environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Action mapping: {env.action_mapping}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test a few steps
    for step in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={env.action_mapping[action]}, "
              f"Reward={reward:.3f}, Done={terminated or truncated}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    env.close()
    print("Environment test completed!")
