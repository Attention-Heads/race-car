"""
Centralized preprocessing utilities for the race car environment.

This module provides a single source of truth for all preprocessing logic
used across imitation learning, PPO training, API, and evaluation scripts.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Optional, Union, List

logger = logging.getLogger(__name__)

# Feature order must be consistent across all components
FEATURE_ORDER = [
    'velocity_x', 'velocity_y',
    'sensor_back', 'sensor_back_left_back', 'sensor_back_right_back',
    'sensor_front', 'sensor_front_left_front', 'sensor_front_right_front',
    'sensor_left_back', 'sensor_left_front', 'sensor_left_side',
    'sensor_left_side_back', 'sensor_left_side_front',
    'sensor_right_back', 'sensor_right_front', 'sensor_right_side',
    'sensor_right_side_back', 'sensor_right_side_front'
]

# Constants
SENSOR_MAX_RANGE = 1000.0
VELOCITY_SCALER_PATH = 'velocity_scaler.pkl'


class StatePreprocessor:
    """
    Centralized state preprocessing for the race car environment.
    
    This class ensures consistent preprocessing across all components:
    - Imitation learning training
    - PPO training and inference
    - API endpoints
    - Evaluation scripts
    """
    
    def __init__(self, use_velocity_scaler: bool = True, velocity_scaler_path: str = VELOCITY_SCALER_PATH):
        """
        Initialize the preprocessor.
        
        Args:
            use_velocity_scaler: Whether to use velocity scaling
            velocity_scaler_path: Path to the velocity scaler file
        """
        self.use_velocity_scaler = use_velocity_scaler
        self.velocity_scaler_path = velocity_scaler_path
        self.velocity_scaler = None
        
        # Load velocity scaler if available
        if self.use_velocity_scaler:
            try:
                self.velocity_scaler = joblib.load(velocity_scaler_path)
                logger.info(f"Loaded velocity scaler from {velocity_scaler_path}")
            except FileNotFoundError:
                logger.warning(f"Velocity scaler not found at {velocity_scaler_path}")
                self.velocity_scaler = None
            except Exception as e:
                logger.warning(f"Failed to load velocity scaler: {e}")
                self.velocity_scaler = None
    
    def preprocess_state_dict(self, state_dto: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess a state dictionary (from environment or API).
        
        Args:
            state_dto: State dictionary with 'velocity' and 'sensors' keys
            
        Returns:
            Preprocessed state array
        """
        # Extract velocity
        flat_state = [state_dto['velocity']['x'], state_dto['velocity']['y']]
        
        # Process sensor readings
        for sensor_name in FEATURE_ORDER[2:]:
            sensor_key = sensor_name.replace('sensor_', '')
            sensor_value = state_dto['sensors'].get(sensor_key, SENSOR_MAX_RANGE)
            
            # Ensure sensor value is valid
            if sensor_value is None or np.isnan(sensor_value):
                sensor_value = SENSOR_MAX_RANGE
            
            flat_state.append(float(sensor_value))
        
        return self._apply_preprocessing(np.array(flat_state, dtype=np.float32))
    
    def preprocess_dataframe_row(self, row: Union[pd.Series, Dict]) -> np.ndarray:
        """
        Preprocess a single row from a DataFrame or dictionary.
        
        Args:
            row: DataFrame row or dictionary with feature columns
            
        Returns:
            Preprocessed state array
        """
        # Build state array from feature columns
        flat_state = []
        
        for feature_name in FEATURE_ORDER:
            value = row.get(feature_name, SENSOR_MAX_RANGE if feature_name.startswith('sensor_') else 0.0)
            
            # Handle missing or invalid values
            if pd.isna(value) or value is None:
                if feature_name.startswith('sensor_'):
                    value = SENSOR_MAX_RANGE
                else:  # velocity
                    value = 0.0
            
            flat_state.append(float(value))
        
        return self._apply_preprocessing(np.array(flat_state, dtype=np.float32))
    
    def preprocess_batch(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess a batch of data.
        
        Args:
            data: Batch data as numpy array or DataFrame
            
        Returns:
            Preprocessed batch
        """
        if isinstance(data, pd.DataFrame):
            # Extract features in correct order
            X = data[FEATURE_ORDER].values.astype(np.float32)
        else:
            X = data.astype(np.float32)
        
        # Apply preprocessing to each sample
        return np.array([self._apply_preprocessing(sample) for sample in X])
    
    def _apply_preprocessing(self, state_array: np.ndarray) -> np.ndarray:
        """
        Apply the core preprocessing steps to a state array.
        
        Args:
            state_array: Raw state array [velocity_x, velocity_y, sensor1, ..., sensorN]
            
        Returns:
            Preprocessed state array
        """
        # Normalize sensor readings (divide by max range)
        state_array[2:] /= SENSOR_MAX_RANGE
        
        # Apply velocity scaling if available
        if self.velocity_scaler is not None:
            try:
                # Create DataFrame with proper feature names for velocity
                velocity_df = pd.DataFrame([state_array[:2]], columns=['velocity_x', 'velocity_y'])
                velocity_scaled = self.velocity_scaler.transform(velocity_df)[0]
                state_array[:2] = velocity_scaled
            except Exception as e:
                logger.warning(f"Velocity scaling failed: {e}")
                # Keep original velocity values if scaling fails
        
        # Final check for NaN values
        if np.isnan(state_array).any():
            logger.warning("NaN detected in state array, replacing with defaults")
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        return state_array
    
    def get_feature_order(self) -> List[str]:
        """Get the feature order used by this preprocessor."""
        return FEATURE_ORDER.copy()
    
    def get_input_dim(self) -> int:
        """Get the input dimension after preprocessing."""
        return len(FEATURE_ORDER)


# Convenience functions for backward compatibility
def create_preprocessor(use_velocity_scaler: bool = True) -> StatePreprocessor:
    """Create a state preprocessor instance."""
    return StatePreprocessor(use_velocity_scaler=use_velocity_scaler)


def preprocess_state_dict(state_dto: Dict[str, Any], 
                         velocity_scaler: Optional[Any] = None) -> np.ndarray:
    """
    Legacy function for preprocessing state dictionaries.
    
    Args:
        state_dto: State dictionary
        velocity_scaler: Optional velocity scaler (for backward compatibility)
        
    Returns:
        Preprocessed state array
    """
    preprocessor = StatePreprocessor(use_velocity_scaler=velocity_scaler is not None)
    if velocity_scaler is not None:
        preprocessor.velocity_scaler = velocity_scaler
    return preprocessor.preprocess_state_dict(state_dto)


def preprocess_dataframe_row(row: Union[pd.Series, Dict], 
                           velocity_scaler: Optional[Any] = None) -> np.ndarray:
    """
    Legacy function for preprocessing DataFrame rows.
    
    Args:
        row: DataFrame row or dictionary
        velocity_scaler: Optional velocity scaler (for backward compatibility)
        
    Returns:
        Preprocessed state array
    """
    preprocessor = StatePreprocessor(use_velocity_scaler=velocity_scaler is not None)
    if velocity_scaler is not None:
        preprocessor.velocity_scaler = velocity_scaler
    return preprocessor.preprocess_dataframe_row(row)
