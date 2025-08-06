from typing import List, Dict

_N_LANE_CHANGE = 47
_N_LANE_CHANGE_DIAGONAL = 34
_N_LANE_CHANGE_DIAGONAL_BACK = 34


class RuleBasedAgent:
    def __init__(self):
        self.maneuver_sequence = []
        self.is_performing_maneuver = False
        self.promising_car_left = None
        self.promising_car_right = None
        self.has_gone_to_left = False
        self.has_gone_to_right = False

    def _sanitize(self, value):
        return value if value is not None else 1000

    def _get_cruise_control_actions(self, s: Dict[str, float]) -> List[str]:
        # Cruise control model
        if s['front'] < 1000:
            return ['DECELERATE']
        else:
            return ['ACCELERATE']

    def initate_change_lane_right(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = ["STEER_RIGHT"] * \
                _N_LANE_CHANGE + ["STEER_LEFT"] * _N_LANE_CHANGE

    def initate_change_lane_left(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = [
                "STEER_LEFT"] * _N_LANE_CHANGE + ["STEER_RIGHT"] * _N_LANE_CHANGE

    def initate_change_lane_left_front(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = (["STEER_LEFT"] + ['ACCELERATE']) * _N_LANE_CHANGE_DIAGONAL + (
                ["STEER_RIGHT"] + ['DECELERATE']) * _N_LANE_CHANGE_DIAGONAL

    def initate_change_lane_right_front(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = (["STEER_RIGHT"] + ['ACCELERATE']) * _N_LANE_CHANGE_DIAGONAL + (
                ["STEER_LEFT"] + ['DECELERATE']) * _N_LANE_CHANGE_DIAGONAL

    def initate_change_lane_right_back(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = (["STEER_RIGHT"] + ['DECELERATE']) * _N_LANE_CHANGE_DIAGONAL_BACK + (
                ["STEER_LEFT"] + ['ACCELERATE']) * _N_LANE_CHANGE_DIAGONAL_BACK

    def initate_change_lane_left_back(self):
        if not self.is_performing_maneuver:
            self.is_performing_maneuver = True
            self.maneuver_sequence = (["STEER_LEFT"] + ['DECELERATE']) * _N_LANE_CHANGE_DIAGONAL_BACK + (
                ["STEER_RIGHT"] + ['ACCELERATE']) * _N_LANE_CHANGE_DIAGONAL_BACK

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
                return actions  # Return only the steering action
            else:
                self.is_performing_maneuver = False

        # Lane change - left
        if not self.has_gone_to_left:
            if s['back_left_back'] < 600 and self.promising_car_left:
                self.has_gone_to_left = True
                self.has_gone_to_right = False
                self.initate_change_lane_left_front()
            elif s['front_left_front'] < 600:
                self.promising_car_left = 'front_left_front'

            if self.promising_car_left == 'front_left_front' and s['left_side'] < 200:
                self.promising_car_left == True
            else:
                self.promising_car_left = False

        # Lane change - right
        if not self.has_gone_to_right:
            if s['back_right_back'] < 600 and self.promising_car_right:
                self.has_gone_to_right = True
                self.has_gone_to_left = False
                self.initate_change_lane_right_front()
            elif s['front_right_front'] < 600:
                self.promising_car_right = 'front_right_front'

            if self.promising_car_right == 'front_right_front' and s['right_side'] < 200:
                self.promising_car_right == True
            else:
                self.promising_car_right = False

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
