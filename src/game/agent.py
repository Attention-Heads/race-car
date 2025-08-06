from typing import List


def get_action_from_rule_based_agent(sensor_data: dict, ticks: int) -> List[str]:

    def sanitize(value):
        return value if value is not None else 1000

    # Sanitize and extract all 16 sensor readings for comprehensive use
    s = {name: sanitize(sensor_data.get(name)) for name in [
        "left_side", "left_side_front", "left_front", "front_left_front",
        "front", "front_right_front", "right_front", "right_side_front",
        "right_side", "right_side_back", "right_back", "back_right_back",
        "back", "back_left_back", "left_back", "left_side_back"
    ]}

    if ticks > 3580 and s['front'] > 500:
        return ['ACCELERATE']

    # Cruise control model
    if s['front'] < 1000:
        return ['DECELERATE']
    elif s['back'] > 500 and s['back'] < 1000:
        return ['DECELERATE']
    elif s['back'] < 500:
        return ['ACCELERATE']
    elif s['front'] >= 1000:
        return ['ACCELERATE']
    else:
        return ['NOTHING']
