from typing import List, Dict

N = 48 # 34 for while de/accelerating

class RuleBasedAgent:
    def __init__(self):
        self.maneuver_sequence = []
        self.is_performing_maneuver = False

    def _sanitize(self, value):
        return value if value is not None else 1000

    def _get_cruise_control_actions(self, s: Dict[str, float]) -> List[str]:
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
            return [] # Return empty list for no action

    def initate_change_lane_right(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = ["STEER_RIGHT"] * N + ["STEER_LEFT"] * N

    def initate_change_lane_left(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = ["STEER_LEFT"] * N + ["STEER_RIGHT"] * N
        

    def get_actions(self, sensor_data: dict) -> List[str]:
        # Sanitize sensor data
        s = {name: self._sanitize(sensor_data.get(name)) for name in [
            "left_side", "left_side_front", "left_front", "front_left_front",
            "front", "front_right_front", "right_front", "right_side_front",
            "right_side", "right_side_back", "right_back", "back_right_back",
            "back", "back_left_back", "left_back", "left_side_back"
        ]}

        actions = []
        steering_action = None

        # If a maneuver is in progress, prioritize it over cruise control.
        if self.is_performing_maneuver:
            if self.maneuver_sequence:
                steering_action = self.maneuver_sequence.pop(0)
                actions.append(steering_action)
                return actions # Return only the steering action
            else:
                self.is_performing_maneuver = False
        
        # If no maneuver, perform cruise control
        cruise_actions = self._get_cruise_control_actions(s)
        actions.extend(cruise_actions)

        if not actions:
            return ["NOTHING"]

        return actions

# Keep the old function for now to avoid breaking other parts of the code, we will remove it later.
def get_action_from_rule_based_agent(sensor_data: dict) -> List[str]:

    def sanitize(value):
        return value if value is not None else 1000

    # Sanitize and extract all 16 sensor readings for comprehensive use
    s = {name: sanitize(sensor_data.get(name)) for name in [
        "left_side", "left_side_front", "left_front", "front_left_front",
        "front", "front_right_front", "right_front", "right_side_front",
        "right_side", "right_side_back", "right_back", "back_right_back",
        "back", "back_left_back", "left_back", "left_side_back"
    ]}

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
