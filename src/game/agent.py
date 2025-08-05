# src/game/agent.py

import numpy as np
from stable_baselines3 import PPO

# Load PPO model once
model = PPO.load("ppo_racecar")

ACTION_LIST = ["NOTHING", "ACCELERATE",
               "DECELERATE", "STEER_LEFT", "STEER_RIGHT"]


def get_action_from_agent(sensor_data):
    """
    Receives dict of 16 named sensors, returns action list.
    """
    sensor_order = [
        "front", "right_front", "right_side", "right_back",
        "back", "left_back", "left_side", "left_front",
        "left_side_front", "front_left_front", "front_right_front", "right_side_front",
        "right_side_back", "back_right_back", "back_left_back", "left_side_back"
    ]
    obs = np.array([
        min(sensor_data.get(name, 0) or 0, 1000) / 1000.0
        for name in sensor_order
    ], dtype=np.float32)

    action_idx, _ = model.predict(obs, deterministic=True)
    return [ACTION_LIST[action_idx]]
