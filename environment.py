import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
from utils import print_game_state

@dataclass
class GameState:
    sensors: Dict[str, Optional[int]]
    velocity: Dict[str, int]
    coordinates: Dict[str, int]
    distance: int
    elapsed_time_ms: int
    did_crash: bool

class RaceCarEnvironment:
    def __init__(self, verbose: bool = False):
        # Correctly list all 16 sensors as per README.md
        self.sensor_names = [
            'left_side', 'left_side_front', 'left_front', 'front_left_front',
            'front', 'front_right_front', 'right_front', 'right_side_front',
            'right_side', 'right_side_back', 'right_back', 'back_right_back',
            'back', 'back_left_back', 'left_back', 'left_side_back'
        ]
        self.actions = ['NOTHING', 'ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.idx_to_action = {idx: action for idx, action in enumerate(self.actions)}

        # Define the size of a single timestep's state vector
        self.single_state_size = 16 + 16 + 3 + 4  # dist + ttc + ego + danger
        # Total state size is 8 historical snapshots of the single state vector
        self.state_size = self.single_state_size * 8
        self.action_size = len(self.actions)

        self.verbose = verbose

        # Normalization and feature constants
        self.max_sensor_range = 1000.0
        self.max_velocity = 20.0
        self.max_distance = 50000.0
        self.max_time_ms = 60000.0
        self.max_ttc = 5.0  # seconds, an estimate
        self.danger_threshold = 150.0  # pixels
        self.ttc_epsilon = 1e-6
        
        # Target speed for consistent driving behavior
        self.target_speed = 14.0  # Conservative target speed (can be increased later)

        # --- Buffers for historical data ---
        # We need to store data for up to 64 steps back
        self.history_depth = 65
        self.historical_states = deque(maxlen=self.history_depth)
        self.historical_sensors = deque(maxlen=self.history_depth)
        self.historical_velocities = deque(maxlen=self.history_depth)

        # Indices for the geometric sequence of historical states
        self.historical_indices = [0, 1, 2, 4, 8, 16, 32, 64]

        # Previous state for reward calculation
        self.prev_distance = 0
        self.prev_velocity_x = 0

        # Sensor groupings for danger flags
        self.front_sensors = ['left_front', 'front_left_front', 'front', 'front_right_front', 'right_front']
        self.rear_sensors = ['left_back', 'back_left_back', 'back', 'back_right_back', 'right_back']
        self.left_sensors = ['left_side', 'left_side_front', 'left_side_back']
        self.right_sensors = ['right_side', 'right_side_front', 'right_side_back']
        self.verbose = verbose
        
    def _get_lane(self, y_coord: int) -> int:
        """Determines the current lane based on the car's y-coordinate."""
        # Game dimensions from src/game/core.py and src/elements/road.py
        margin = 40
        lane_count = 5
        screen_height = 1200
        usable_height = screen_height - 2 * margin
        lane_height = usable_height / lane_count

        # Adjust for top margin and calculate lane
        y_relative = y_coord - margin
        lane = (y_relative // lane_height) + 1
        return int(np.clip(lane, 1, lane_count))

    def preprocess_state(self, game_state: GameState) -> np.ndarray:
        """Convert raw game state to the new, comprehensive feature vector."""

        if self.verbose:
            print_game_state(game_state, self.max_sensor_range, self.sensor_names)
        # EXAMPLE OUTPUT: Current GameState:GameState(sensors={'left_side': 1000, 'left_side_front': 1000, 'left_front': 1000, 'front_left_front': 1000, 'front': 1000, 'front_right_front': 1000, 'right_front': 1000, 'right_side_front': 1000, 'right_side': 1000, 'right_side_back': 1000, 'right_back': 1000, 'back_right_back': 1000, 'back': 1000, 'back_left_back': 1000, 'left_back': 1000, 'left_side_back': 1000}, velocity={'x': 10, 'y': 0}, coordinates={'x': 2160, 'y': 619}, distance=1360, elapsed_time_ms=2217, did_crash=False)


        # --- 1. Get current sensor readings ---
        current_sensor_readings = np.array([
            game_state.sensors.get(name) if game_state.sensors.get(name) is not None else self.max_sensor_range
            for name in self.sensor_names
        ])

        # --- 2. Calculate normalized distance (proximity) ---
        # 0 when clear, 1 when touching
        norm_distances = 1.0 - np.clip(current_sensor_readings / self.max_sensor_range, 0, 1)

        # --- 3. Calculate Time-to-Collision (TTC) ---
        if len(self.historical_sensors) > 0:
            prev_sensor_readings = self.historical_sensors[-1]
            # Rate of change: positive value means getting closer
            delta_dist = prev_sensor_readings - current_sensor_readings
            # Avoid division by zero or negative delta (moving away)
            ttc = np.where(delta_dist > self.ttc_epsilon,
                           current_sensor_readings / delta_dist,
                           self.max_ttc) # If moving away or stationary, TTC is max
        else:
            ttc = np.full(16, self.max_ttc)
        norm_ttc = np.clip(ttc / self.max_ttc, 0, 1)

        # --- 4. Calculate Ego Metrics ---
        current_lane = self._get_lane(game_state.coordinates['y'])
        lane_scalar = (current_lane - 3) / 2.0  # -1 (left) to +1 (right)

        v_ego = game_state.velocity['x']
        speed_norm = np.clip(v_ego / self.max_velocity, 0, 1)

        if len(self.historical_velocities) > 0:
            v_prev = self.historical_velocities[-1]
            accel_sign = np.sign(v_ego - v_prev)
        else:
            accel_sign = 0.0
        ego_metrics = np.array([lane_scalar, speed_norm, accel_sign])

        # --- 5. Calculate Danger Flags ---
        sensor_dict = {name: dist for name, dist in zip(self.sensor_names, current_sensor_readings)}
        front_danger = 1.0 if min(sensor_dict[s] for s in self.front_sensors) < self.danger_threshold else 0.0
        rear_danger = 1.0 if min(sensor_dict[s] for s in self.rear_sensors) < self.danger_threshold else 0.0
        left_danger = 1.0 if min(sensor_dict[s] for s in self.left_sensors) < self.danger_threshold else 0.0
        right_danger = 1.0 if min(sensor_dict[s] for s in self.right_sensors) < self.danger_threshold else 0.0
        danger_flags = np.array([front_danger, rear_danger, left_danger, right_danger])

        # --- 6. Assemble single-timestep state and update history ---
        single_timestep_state = np.concatenate([norm_distances, norm_ttc, ego_metrics, danger_flags]).astype(np.float32)
        
        self.historical_states.append(single_timestep_state)
        self.historical_sensors.append(current_sensor_readings)
        self.historical_velocities.append(v_ego)

        # --- 7. Construct the final stacked state vector ---
        # Pad with zero vectors if history is not deep enough
        padded_history = list(self.historical_states)
        while len(padded_history) < self.history_depth:
            padded_history.insert(0, np.zeros(self.single_state_size, dtype=np.float32))

        # Retrieve states from history using geometric indices
        final_state_components = [padded_history[self.history_depth - 1 - i] for i in self.historical_indices]
        
        final_state = np.concatenate(final_state_components)
        return final_state
    
    def calculate_reward(self, prev_state: GameState, curr_state: GameState, action: str) -> float:
        """Calculate reward based on state transition"""
        if curr_state.did_crash:
            return -10.0  # Reduced crash penalty

        # --- Core Rewards, Penalties and Bonuses (as noted by *)---
        # * Progress Reward: Based on distance covered
        distance_reward = (curr_state.distance - prev_state.distance) * 0.005

        # * Speed Reward: Encourage maintaining a target speed
        current_speed = curr_state.velocity['x']
        speed_diff = abs(current_speed - self.target_speed)
        speed_reward = max(0, 1 - speed_diff / self.target_speed) * 0.1  # Max 0.1

        # * Obstacle Proximity Penalty (from sensors): Penalize getting too close to any object.
        valid_sensors = [dist for dist in curr_state.sensors.values() if dist is not None]
        min_sensor_dist = min(valid_sensors) if valid_sensors else self.max_sensor_range
        obstacle_proximity_penalty = 0.0
        if min_sensor_dist < 100:
            obstacle_proximity_penalty = -((100 - min_sensor_dist) / 100.0) * 0.2 # Penalize up to -0.2

        # * Coordinate-Based Wall Penalty: Heavily penalize getting too close to track edges.
        y_coord = curr_state.coordinates['y']
        road_height = 1200  # As defined in src/game/core.py
        wall_distance_threshold = 40 # Pixels from the edge
        
        dist_to_top_wall = y_coord
        dist_to_bottom_wall = road_height - y_coord
        min_dist_to_wall = min(dist_to_top_wall, dist_to_bottom_wall)
        
        wall_penalty = 0.0
        if min_dist_to_wall < wall_distance_threshold:
            # Quadratic penalty for being super-close to the wall
            wall_penalty = -2.0 * ((wall_distance_threshold - min_dist_to_wall) / wall_distance_threshold)**2

        # * Steering Penalty: Penalize sharp turns to encourage smooth driving
        steering_penalty = -0.05 if "STEER" in action else 0.0

        # * Time Penalty: Small penalty per step to encourage efficiency
        time_penalty = -0.001

        total_reward = (distance_reward +
                        speed_reward +
                        obstacle_proximity_penalty +
                        wall_penalty +
                        steering_penalty +
                        time_penalty)

        return total_reward
    
    def is_terminal(self, game_state: GameState) -> bool:
        """Check if episode should terminate"""
        return (game_state.did_crash or 
                game_state.elapsed_time_ms >= self.max_time_ms)
    
    def action_to_index(self, action: str) -> int:
        """Convert action string to index"""
        return self.action_to_idx.get(action, 0)  # Default to NOTHING
    
    def index_to_action(self, index: int) -> str:
        """Convert action index to string"""
        return self.idx_to_action.get(index, 'NOTHING')
    
    def get_action_batch(self, actions: List[str], batch_size: int = 10) -> List[str]:
        """Create action batch for server communication"""
        if len(actions) >= batch_size:
            return actions[:batch_size]
        else:
            # Repeat last action to fill batch
            last_action = actions[-1] if actions else 'NOTHING'
            return actions + [last_action] * (batch_size - len(actions))