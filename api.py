import uvicorn
from fastapi import Body, FastAPI
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [+] Load your trained PPO agent and the velocity scaler
try:
    AGENT = PPO.load("./models/final_model.zip")
    VELOCITY_SCALER = joblib.load('velocity_scaler.pkl')
    print("Trained agent and scaler loaded successfully.")
except Exception as e:
    print(f"Could not load trained agent or scaler: {e}")
    AGENT = None
    VELOCITY_SCALER = None

# [+] Your DTOs remain the same
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

app = FastAPI()

def process_request_for_model(request: RaceCarPredictRequestDto) -> np.ndarray:
    """This function must perform the EXACT same preprocessing as the environment."""
    # Your feature order must be identical
    feature_order = [
        'velocity_x', 'velocity_y',
        'sensor_back', 'sensor_back_left_back', 'sensor_back_right_back',
        'sensor_front', 'sensor_front_left_front', 'sensor_front_right_front',
        'sensor_left_back', 'sensor_left_front', 'sensor_left_side',
        'sensor_left_side_back', 'sensor_left_side_front',
        'sensor_right_back', 'sensor_right_front', 'sensor_right_side',
        'sensor_right_side_back', 'sensor_right_side_front'
    ]
    
    flat_state = [request.velocity['x'], request.velocity['y']]
    
    # Process sensor readings - ensure we have valid values
    for sensor_name in feature_order[2:]:
        sensor_key = sensor_name.replace('sensor_', '')
        sensor_value = request.sensors.get(sensor_key, 1000.0)
        # Ensure sensor value is not None or NaN
        if sensor_value is None or np.isnan(sensor_value):
            sensor_value = 1000.0  # Max sensor range as default
        flat_state.append(float(sensor_value))
    
    state_array = np.array(flat_state, dtype=np.float32)
    
    # Normalize sensor readings
    state_array[2:] /= 1000.0
    
    # Apply velocity scaling if available (same as environment)
    if VELOCITY_SCALER:
        try:
            # Create DataFrame with proper feature names for velocity
            velocity_df = pd.DataFrame([state_array[:2]], columns=['velocity_x', 'velocity_y'])
            velocity_scaled = VELOCITY_SCALER.transform(velocity_df)[0]
            state_array[:2] = velocity_scaled
        except Exception as e:
            logger.warning(f"Velocity scaling failed: {e}")
            # Keep original velocity values if scaling fails
    
    # Final check for NaN values
    if np.isnan(state_array).any():
        logger.warning("NaN detected in state array, replacing with defaults")
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=0.0)
        
    return state_array

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    try:
        if not AGENT:
            logger.error("Model not loaded")
            return RaceCarPredictResponseDto(error="Model not loaded")
        
        # [+] Process the incoming data into a numpy array
        observation = process_request_for_model(request)
        
        # Additional validation to ensure observation is valid
        if observation is None or len(observation) == 0:
            logger.error("Invalid observation array")
            return RaceCarPredictResponseDto(error="Invalid observation data")
        
        # Check for any remaining NaN or inf values
        if not np.isfinite(observation).all():
            logger.error(f"Invalid values in observation: {observation}")
            return RaceCarPredictResponseDto(error="Invalid observation values")
        
        # [+] Use the agent to predict the action
        action_num, _ = AGENT.predict(observation, deterministic=True)
        
        # [+] Map number to action name
        action_mapping = {0: 'NOTHING', 1: 'ACCELERATE', 2: 'DECELERATE', 3: 'STEER_LEFT', 4: 'STEER_RIGHT'}
        action_name = action_mapping.get(int(action_num), 'NOTHING')
        
        return RaceCarPredictResponseDto(actions=[action_name]*5)
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return RaceCarPredictResponseDto(error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run('api:app', host="0.0.0.0", port=9052, reload=True)