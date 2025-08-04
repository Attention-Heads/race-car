from typing import List
import random


def get_action_from_rule_based_agent(sensor_data: dict) -> List[str]:
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

    queue = []

    if len(queue) == 0:
        # Crash avoidance logic
        if front < 1000:
            if front_left > front_right:
                queue.append(["STEER_LEFT"] * 10 + ["STEER_RIGHT"] * 10)
            else:
                queue.append(["STEER_RIGHT"] * 10 + ["STEER_LEFT"] * 10)

        elif back < 300:
            if back_left > back_right:
                queue.append(["STEER_LEFT"] * 10 + ["STEER_RIGHT"] * 10)
            else:
                queue.append(["STEER_RIGHT"] * 10 + ["STEER_LEFT"] * 10)

        # Avoid drifting into walls
        elif left < 100:
            queue.append(["STEER_RIGHT"] * 3 + ["STEER_LEFT"] * 3)
        elif right < 100:
            queue.append(["STEER_LEFT"] * 3 + ["STEER_RIGHT"] * 3)
        else:
            queue.append("NOTHING")
    return queue
