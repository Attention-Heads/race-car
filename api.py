import uvicorn
from fastapi import Body, FastAPI
from stable_baselines3 import PPO
import numpy as np

# [+] Load your trained PPO agent and the velocity scaler
try:
    AGENT = PPO.load("./models/ppo_racecar_final.zip")
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
    for sensor_name in feature_order[2:]:
        sensor_key = sensor_name.replace('sensor_', '')
        flat_state.append(request.sensors.get(sensor_key, 1000.0))
    
    state_array = np.array(flat_state, dtype=np.float32)
    state_array[2:] /= 1000.0
    if VELOCITY_SCALER:
        state_array[:2] = VELOCITY_SCALER.transform([state_array[:2]])[0]
        
    return state_array

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    if not AGENT:
        return {"error": "Model not loaded"}
    
    # [+] Process the incoming data into a numpy array
    observation = process_request_for_model(request)
    
    # [+] Use the agent to predict the action
    action_num, _ = AGENT.predict(observation, deterministic=True)
    
    # [+] Map number to action name
    action_mapping = {0: 'NOTHING', 1: 'ACCELERATE', 2: 'DECELERATE', 3: 'STEER_LEFT', 4: 'STEER_RIGHT'}
    action_name = action_mapping.get(int(action_num), 'NOTHING')
    
    return RaceCarPredictResponseDto(action=action_name)

if __name__ == '__main__':
    uvicorn.run('api:app', host="0.0.0.0", port=9052)