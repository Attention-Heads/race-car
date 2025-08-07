from typing import List
from typing import List, Dict

_N_LANE_CHANGE = 47
_N_LANE_CHANGE_DIAGONAL = 34
_N_LANE_CHANGE_DIAGONAL_BACK = 34


class RuleBasedAgent:
    def __init__(self):
        self.maneuver_sequence = []
        self.is_performing_maneuver = False
        self.promising_car_left_back = False
        self.promising_car_right_back = False
        self.has_gone_to_left = False
        self.has_gone_to_right = False
        self.previous_distance = None
        self.started_maneuver = False

    def _sanitize(self, value):
        return value if value is not None else 1000

    def _get_cruise_control_actions(self, s: Dict[str, float]) -> List[str]:

        # Cruise control model
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

    def car_is_touching_sensors_left(self, sensors) -> bool:
        return sensors['left_side'] < 100 or sensors['left_side_front'] < 200 or sensors['left_front'] < 400 or sensors['left_side_back'] < 200 or sensors['left_back'] < 400 or sensors['back_left_back'] < 600

    def car_is_touching_sensors_right(self, sensors):
        return sensors['right_side'] < 100 or sensors['right_side_front'] < 200 or sensors['right_front'] < 400 or sensors['right_side_back'] < 200 or sensors['right_back'] < 400 or sensors['back_right_back'] < 600

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

        # Lane shift condition - should be updated
        if s['front'] > 850:
            # Lane change - left
            if not self.has_gone_to_left:
                if s['back_left_back'] < 650 and s['back_left_back'] > 400 and self.promising_car_left_back:
                    self.has_gone_to_left = True
                    self.has_gone_to_right = False
                    self.started_maneuver = True
                    self.initate_change_lane_left()
                elif s['front_left_front'] < 650:
                    self.promising_car_left_back = True

                if self.promising_car_left_back and not self.car_is_touching_sensors_left(s):
                    self.promising_car_left_back = False

            # Lane change - right
            if not self.has_gone_to_right and not self.started_maneuver:
                if s['back_right_back'] < 650 and s['back_right_back'] > 400 and self.promising_car_right_back:
                    self.has_gone_to_right = True
                    self.has_gone_to_left = False
                    self.initate_change_lane_right()
                    self.started_maneuver = True
                elif s['front_right_front'] < 650:
                    self.promising_car_right_back = True

                if self.promising_car_right_back and not self.car_is_touching_sensors_right(s):
                    self.promising_car_right_back = False

        self.started_maneuver = False
        # If no maneuver, perform cruise control
        cruise_actions = self._get_cruise_control_actions(s)
        actions.extend(cruise_actions)

        if not actions:
            return ["NOTHING"]
        return actions
