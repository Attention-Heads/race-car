import numpy as np
from collections import deque
from typing import Dict, Optional, List
from ..mathematics.vector import Vector


class StateProcessor:
    """
    Processes sensor data with temporal stacking using geometric backoff.
    Creates normalized state vectors for DQN training.
    """
    
    def __init__(self):
        self.sensor_history = deque(maxlen=32)  # Store up to 32 timesteps
        self.sensor_names = [
            'left_side', 'left_side_front', 'left_front', 'front_left_front',
            'front', 'front_right_front', 'right_front', 'right_side_front',
            'right_side', 'right_side_back', 'right_back', 'back_right_back',
            'back', 'back_left_back', 'left_back', 'left_side_back'
        ]
        self.state_size = self._calculate_state_size()
    
    def _calculate_state_size(self) -> int:
        """Calculate total state vector size."""
        # 16 sensors × 7 timesteps + 5 additional features
        return len(self.sensor_names) * 7 + 5
    
    def add_sensor_reading(self, sensors: Dict[str, Optional[float]]):
        """Add new sensor reading to history."""
        # Sanitize sensor data
        sanitized = {}
        for name in self.sensor_names:
            value = sensors.get(name)
            sanitized[name] = value if value is not None else 1000.0
        
        self.sensor_history.append(sanitized)
    
    def get_stacked_state(self, 
                         lane: int, 
                         velocity: Vector, 
                         time_since_lane_change: int, 
                         distance: float) -> np.ndarray:
        """
        Create stacked state with geometric backoff temporal sampling.
        
        Args:
            lane: Current lane (0-4)
            velocity: Current velocity vector
            time_since_lane_change: Ticks since last lane change
            distance: Total distance traveled
            
        Returns:
            Normalized state vector of size state_size
        """
        # Geometric backoff indices: [k, k-1, k-2, k-4, k-8, k-16, k-32]
        indices = [0, 1, 2, 4, 8, 16, 32]
        stacked_features = []
        
        # Stack sensor readings with geometric backoff
        for idx in indices:
            if idx < len(self.sensor_history):
                sensors = self.sensor_history[-(idx+1)]
                stacked_features.extend([
                    sensors.get(name, 1000.0) / 1000.0  # Normalize to [0,1]
                    for name in self.sensor_names
                ])
            else:
                # If we don't have enough history, pad with default values
                stacked_features.extend([1.0] * len(self.sensor_names))
        
        # Add normalized additional state information
        additional_features = [
            (lane - 2) / 2.0,  # Normalize lane to [-1,1] (lane 0-4 -> -2 to +2 -> -1 to +1)
            min(time_since_lane_change / 180.0, 1.0),  # Normalize cooldown to [0,1]
            np.tanh(velocity.x / 50.0),  # Soft normalize vx (tanh keeps outliers bounded)
            np.tanh(velocity.y / 5.0),   # Soft normalize vy 
            min(distance / 36000.0, 1.0)  # Normalize by max possible distance in 60s
        ]
        
        stacked_features.extend(additional_features)
        
        # Ensure we have the expected number of features
        expected_size = self.state_size
        if len(stacked_features) != expected_size:
            raise ValueError(f"State size mismatch: expected {expected_size}, got {len(stacked_features)}")
        
        return np.array(stacked_features, dtype=np.float32)
    
    def reset(self):
        """Reset sensor history for new episode."""
        self.sensor_history.clear()
    
    def get_current_sensors(self) -> Optional[Dict[str, float]]:
        """Get most recent sensor reading."""
        if len(self.sensor_history) > 0:
            return self.sensor_history[-1]
        return None
    
    def is_ready(self) -> bool:
        """Check if we have enough history for meaningful state."""
        return len(self.sensor_history) >= 1  # At least one reading
    
    def get_sensor_at_offset(self, offset: int) -> Optional[Dict[str, float]]:
        """Get sensor reading at specific offset from current time."""
        if offset < len(self.sensor_history):
            return self.sensor_history[-(offset+1)]
        return None


class LaneTracker:
    """
    Tracks current lane position based on sensor readings and lane changes.
    """
    
    def __init__(self, initial_lane: int = 2):
        self.current_lane = initial_lane  # Start in middle lane
        self.lane_change_in_progress = False
        self.lane_change_start_tick = 0
        self.target_lane = initial_lane
    
    def initiate_lane_change(self, direction: str, current_tick: int) -> bool:
        """
        Initiate lane change if valid.
        
        Returns:
            True if lane change initiated, False if invalid
        """
        if self.lane_change_in_progress:
            return False
        
        if direction == "left" and self.current_lane <= 0:
            return False  # Can't go left from leftmost lane
        
        if direction == "right" and self.current_lane >= 4:
            return False  # Can't go right from rightmost lane
        
        self.lane_change_in_progress = True
        self.lane_change_start_tick = current_tick
        
        if direction == "left":
            self.target_lane = self.current_lane - 1
        else:
            self.target_lane = self.current_lane + 1
        
        return True
    
    def update(self, current_tick: int, is_performing_maneuver: bool):
        """Update lane tracking based on maneuver status."""
        if self.lane_change_in_progress and not is_performing_maneuver:
            # Lane change completed
            self.current_lane = self.target_lane
            self.lane_change_in_progress = False
    
    def get_time_since_lane_change(self, current_tick: int) -> int:
        """Get ticks since last lane change completed."""
        if self.lane_change_in_progress:
            return 0  # Currently changing lanes
        return current_tick - self.lane_change_start_tick
    
    def is_valid_action(self, action: int) -> bool:
        """Check if lane change action is valid given current position."""
        if action == 0:  # LANE_CHANGE_LEFT
            return self.current_lane > 0 and not self.lane_change_in_progress
        elif action == 1:  # LANE_CHANGE_RIGHT
            return self.current_lane < 4 and not self.lane_change_in_progress
        else:  # DO_NOTHING
            return True