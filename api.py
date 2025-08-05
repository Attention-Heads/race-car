import uvicorn
from fastapi import Body, FastAPI
import numpy as np
import pandas as pd
import joblib
import logging
import os
from preprocessing_utils import StatePreprocessor
from evolutionary_model_wrapper import load_evolutionary_model, get_best_evolutionary_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# [+] Load evolutionary model
try:
    best_model_path = get_best_evolutionary_model("./evolutionary_results")
    AGENT = load_evolutionary_model(best_model_path)
    print(f"Evolutionary model loaded successfully from {best_model_path}")
    
    PREPROCESSOR = StatePreprocessor(use_velocity_scaler=True)
    print("Preprocessor loaded successfully. Using EVOLUTIONARY model.")
except Exception as e:
    print(f"Could not load evolutionary model or preprocessor: {e}")
    AGENT = None
    PREPROCESSOR = None

# [+] Your DTOs remain the same
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

app = FastAPI()

def process_request_for_model(request: RaceCarPredictRequestDto) -> np.ndarray:
    """Use centralized preprocessing for consistency with training environment."""
    if PREPROCESSOR is None:
        raise RuntimeError("Preprocessor not loaded")
    
    # Convert request to state dictionary format (same as training environment)
    state_dto = {
        'velocity': {'x': request.velocity['x'], 'y': request.velocity['y']},
        'sensors': request.sensors
    }
    
    state_array = PREPROCESSOR.preprocess_state_dict(state_dto)
    
    # Apply the same NaN handling as the environment
    if np.isnan(state_array).any():
        logger.warning("NaN detected in state array, replacing with defaults")
        state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=0.0)
    
    return state_array

@app.get('/')
def root():
    return {"message": "Welcome to the Race Car API - Using EVOLUTIONARY model"}

@app.get('/model-info')
def model_info():
    return {"model_type": "EVOLUTIONARY", "model_loaded": AGENT is not None}

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
        
        return RaceCarPredictResponseDto(actions=[action_name])
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return RaceCarPredictResponseDto(error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run('api:app', host="127.0.0.1", port=8000, reload=False)