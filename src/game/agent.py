from typing import List

last_action = ["NOTHING"]  # Can be handled better with stateful agent class


def get_action_from_rule_based_agent(sensor_data: dict) -> List[str]:
    def sanitize(value):
        return value if value is not None else 1000

    # Sensor readings
    front = sanitize(sensor_data.get("front"))
    front_left = sanitize(sensor_data.get("front_left_front"))
    front_right = sanitize(sensor_data.get("front_right_front"))
    left = sanitize(sensor_data.get("left_side"))
    right = sanitize(sensor_data.get("right_side"))
    back = sanitize(sensor_data.get("back"))

    actions = []

    # --- Obstacle Avoidance ---
    if front < 150:
        if front_left > front_right:
            actions.append("STEER_LEFT")
        else:
            actions.append("STEER_RIGHT")
        actions.append("DECELERATE")
    elif front < 300:
        actions.append("DECELERATE")

    # --- Wall Avoidance ---
    if left < 50:
        actions.append("STEER_RIGHT")
    elif right < 50:
        actions.append("STEER_LEFT")

    # --- Acceleration Logic ---
    if front > 400 and front_left > 300 and front_right > 300:
        actions.append("ACCELERATE")

    # --- Maintain Forward Progress ---
    if not actions:
        actions.append("NOTHING")

    global last_action
    last_action = actions
    return actions
