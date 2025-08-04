from typing import List


def get_action_from_rule_based_agent(sensor_data: dict, velocity: dict) -> List[str]:
    def sanitize(value):
        return value if value is not None else 1000

    front = sanitize(sensor_data.get("front"))
    back = sanitize(sensor_data.get("back"))
    left = sanitize(sensor_data.get("left_side"))
    right = sanitize(sensor_data.get("right_side"))
    front_left = sanitize(sensor_data.get("front_left_front"))
    front_right = sanitize(sensor_data.get("front_right_front"))
    back_left = sanitize(sensor_data.get("back_left_back"))
    back_right = sanitize(sensor_data.get("back_right_back"))

    # Crash avoidance logic
    if front < 100:
        return ["STEER_LEFT"] if front_left > front_right else ["STEER_RIGHT"]
    elif back < 100:
        return ["STEER_LEFT"] if back_left > back_right else ["STEER_RIGHT"]

    # Avoid drifting into walls
    if left < 50:
        return ["STEER_RIGHT"]
    elif right < 50:
        return ["STEER_LEFT"]

    return ["NOTHING"]
