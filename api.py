import time
import uvicorn
import datetime
from fastapi import Body, FastAPI
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from example import return_action
from src.game.agent import get_action_from_rule_based_agent

HOST = "0.0.0.0"
PORT = 9052


app = FastAPI()
start_time = time.time()


@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    # action = return_action(request.dict())
    actions = get_action_from_rule_based_agent(
        request.dict().get('sensors'), request.dict().get('elapsed_ticks'))
    return RaceCarPredictResponseDto(
        # action_type=action['action_type'],
        actions=actions
    )


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
        'api:app',
        host=HOST,
        port=PORT
    )
