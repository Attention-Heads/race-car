from typing import List

previous_distance = None


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

    if ticks > 3500 and s['front'] > 500:
        return ['ACCELERATE']

    # Cruise control model
    global previous_distance
    current_distance = s['front']

    # If we haven't seen a previous value yet, just store and accelerate
    if previous_distance is None:
        previous_distance = current_distance
        return ['ACCELERATE']

    # Estimate rate of change (assume Δt = 1 for simplicity)
    delta_d = current_distance - previous_distance
    previous_distance = current_distance

    # If distance is increasing, car in front is pulling away → accelerate
    # If distance is decreasing, you're getting closer → decelerate
    if current_distance < 1000:
        if delta_d > 0 and current_distance > 900:
            return ['ACCELERATE']
        elif delta_d < 0:
            return ['DECELERATE']
        else:
            return ['NOTHING']
    else:
        return ['ACCELERATE']
