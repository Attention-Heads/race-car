import time
import uvicorn
import datetime
import os
from fastapi import Body, FastAPI
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from src.game.agent import RuleBasedAgent
from src.agents.dqn_agent import DQNAgent
from src.mathematics.vector import Vector

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI()
start_time = time.time()

# Global agents (initialized on startup)
rule_based_agent = RuleBasedAgent()
dqn_agent = None

# Load DQN model on startup
MODEL_PATH = 'models/checkpoints/dqn_latest.pth'
if os.path.exists(MODEL_PATH):
    try:
        dqn_agent = DQNAgent()
        dqn_agent.load_model(MODEL_PATH)
        dqn_agent.set_eval_mode()
        print(f"✅ Loaded DQN model: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load DQN model: {e}")
        print("   Falling back to rule-based agent")
else:
    print(f"❌ DQN model not found: {MODEL_PATH}")
    print("   Falling back to rule-based agent")


@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    global rule_based_agent, dqn_agent
    
    # Extract sensor data (same as original)
    sensors = request.dict().get('sensors')
    
    if dqn_agent is not None:
        # Use DQN agent
        velocity = Vector(
            request.velocity.get('x', 0), 
            request.velocity.get('y', 0)
        )
        
        # Get DQN lane change decision
        dqn_action = dqn_agent.get_action(
            sensor_data=sensors,
            velocity=velocity,
            distance=request.distance,
            current_tick=request.elapsed_ticks,
            is_performing_maneuver=rule_based_agent.is_performing_maneuver,
            training=False
        )
        
        # Execute lane change if requested
        if dqn_action in [0, 1]:
            success = dqn_agent.execute_lane_change(dqn_action, request.elapsed_ticks)
            if success:
                direction = "left" if dqn_action == 0 else "right"
                rule_based_agent.execute_lane_change(direction)
        
        # Get cruise control actions
        actions = rule_based_agent.get_actions(sensors)
        
        # Update DQN agent
        if not request.did_crash:
            dqn_agent.step(
                sensor_data=sensors,
                velocity=velocity,
                distance=request.distance,
                current_tick=request.elapsed_ticks,
                crashed=request.did_crash,
                is_performing_maneuver=rule_based_agent.is_performing_maneuver
            )
    else:
        # Fallback to original rule-based behavior
        from src.game.agent import get_action_from_rule_based_agent
        actions = get_action_from_rule_based_agent(sensors)
    
    return RaceCarPredictResponseDto(actions=actions)


@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"


if __name__ == '__main__':
    uvicorn.run(
        'api_dqn:app',
        host=HOST,
        port=PORT
    )