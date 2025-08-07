from typing import List
from typing import List, Dict

_N_LANE_CHANGE = 47
_N_LANE_CHANGE_DIAGONAL = 34
_N_LANE_CHANGE_DIAGONAL_BACK = 34


class RuleBasedAgent:
    def __init__(self):
        self.maneuver_sequence = []
        self.is_performing_maneuver = False
        self.previous_distance = None
        # Lane switching logic removed - will be handled by DQN agent

    def _sanitize(self, value):
        return value if value is not None else 1000

    def _get_cruise_control_actions(self, s: Dict[str, float]) -> List[str]:

        # Cruise control model
        self.previous_distance
        current_distance = s['front']

        # If we haven't seen a previous value yet, just store and accelerate
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return ['ACCELERATE']

        # Estimate rate of change (assume Δt = 1 for simplicity)
        delta_d = current_distance - self.previous_distance
        self.previous_distance = current_distance

        # If distance is increasing, car in front is pulling away → accelerate
        # If distance is decreasing, you're getting closer → decelerate
        if current_distance < 1000:
            if delta_d > 0 and current_distance > 900:
                return ['ACCELERATE']
            else:
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

        # If a lane change maneuver is in progress, prioritize it over cruise control
        if self.is_performing_maneuver:
            if self.maneuver_sequence:
                steering_action = self.maneuver_sequence.pop(0)
                actions.append(steering_action)
                return actions  # Return only the steering action
            else:
                self.is_performing_maneuver = False

        # Lane switching decisions will be handled by DQN agent
        # This agent now only handles cruise control
        cruise_actions = self._get_cruise_control_actions(s)
        actions.extend(cruise_actions)

        if not actions:
            return ["NOTHING"]

        return actions

    def execute_lane_change(self, direction: str):
        """Execute a lane change in the specified direction. Called by DQN agent."""
        if direction == "left":
            self.initate_change_lane_left()
        elif direction == "right":
            self.initate_change_lane_right()

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
