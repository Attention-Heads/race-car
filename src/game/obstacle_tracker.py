import numpy as np

class ObstacleTracker:
    def __init__(self, state):
        self.state = state
        self.lanes_data = {}

    def update(self):
        ego_lane_index = self._get_ego_lane_index()
        if ego_lane_index is None:
            return

        self._update_lanes_data(ego_lane_index)

    def _get_ego_lane_index(self):
        ego_y = self.state.ego.y
        for i, lane in enumerate(self.state.road.lanes):
            if lane.y_start <= ego_y < lane.y_end:
                return i
        return None

    def _update_lanes_data(self, ego_lane_index):
        # Initialize for all relative lanes to ensure all are displayed
        self.lanes_data = {}
        num_lanes = len(self.state.road.lanes)
        for i in range(num_lanes):
            relative_lane = i - ego_lane_index
            self.lanes_data[relative_lane] = []

        for car in self.state.cars:
            if car == self.state.ego:
                continue

            car_lane_index = self._get_car_lane_index(car)
            if car_lane_index is None:
                continue

            relative_lane = car_lane_index - ego_lane_index
            relative_x = car.x - self.state.ego.x

            # The key is guaranteed to exist from the initialization above
            self.lanes_data[relative_lane].append({
                'car': car,
                'relative_x': relative_x
            })

    def _get_car_lane_index(self, car):
        # Assuming car.lane is correctly maintained
        if car.lane and car.lane in self.state.road.lanes:
            try:
                return self.state.road.lanes.index(car.lane)
            except ValueError:
                return self._get_lane_index_by_pos(car)
        return self._get_lane_index_by_pos(car)

    def _get_lane_index_by_pos(self, car):
        car_y = car.y
        for i, lane in enumerate(self.state.road.lanes):
            if lane.y_start <= car_y < lane.y_end:
                return i
        return None

    def get_lanes_data(self):
        return self.lanes_data
