import uvicorn
from fastapi import Body, FastAPI
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import joblib
import logging
import os
from preprocessing_utils import StatePreprocessor
from bc_model_wrapper import load_bc_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model type configuration - set to 'ppo' or 'bc'
MODEL_TYPE = os.getenv("MODEL_TYPE", "ppo")  # Default to PPO

# [+] Load your trained agent and use centralized preprocessing
try:
    if MODEL_TYPE.lower() == "bc":
        # Load behavioral cloning model
        AGENT = load_bc_model("./models/best_bc_model.pth")
        print("Behavioral cloning model loaded successfully.")
    else:
        # Load PPO model (default)
        AGENT = PPO.load("./models/final_model.zip")
        print("PPO agent loaded successfully.")
    
    PREPROCESSOR = StatePreprocessor(use_velocity_scaler=True)
    print(f"Preprocessor loaded successfully. Using {MODEL_TYPE.upper()} model.")
except Exception as e:
    print(f"Could not load trained agent or preprocessor: {e}")
    AGENT = None
    PREPROCESSOR = None

# [+] Your DTOs remain the same
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto

app = FastAPI()

def process_request_for_model(request: RaceCarPredictRequestDto) -> np.ndarray:
    """Use centralized preprocessing for consistency."""
    if PREPROCESSOR is None:
        raise RuntimeError("Preprocessor not loaded")
    
    # Convert request to state dictionary format
    state_dto = {
        'velocity': {'x': request.velocity['x'], 'y': request.velocity['y']},
        'sensors': request.sensors
    }
    
    return PREPROCESSOR.preprocess_state_dict(state_dto)

@app.get('/')
def root():
    return {"message": f"Welcome to the Race Car API - Using {MODEL_TYPE.upper()} model"}

@app.get('/model-info')
def model_info():
    return {"model_type": MODEL_TYPE.upper(), "model_loaded": AGENT is not None}

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