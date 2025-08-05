import pygame
import math
import random as py_random
from typing import Dict, Tuple
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now, we use absolute imports starting from 'src'
# This will work because of the temporary fix above.
from src.elements.car import Car
from src.elements.road import Road
from src.elements.sensor import Sensor
from src.mathematics.vector import Vector
from src.mathematics.randomizer import seed, random_choice, random_number

# --- Constants ---
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
MAX_TICKS_PER_EPISODE = 3600  # 60 seconds at 60 FPS

class StateWrapper:
    """A simple wrapper to provide attribute access to dictionary data for sensors."""
    def __init__(self, state_dict):
        self._state = state_dict
    
    def __getattr__(self, name):
        return self._state.get(name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._state[name] = value

class GameSimulation:
    """
    A class that encapsulates the Pygame simulation logic, making it controllable
    by an external script like an RL environment.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Race Car Simulation")
        
        # State variables will be initialized in reset()
        self.state = None
        self.clock = pygame.time.Clock()

    def reset(self, seed_value: int) -> Dict:
        """
        Resets the game to an initial state and returns the first observation dictionary.
        """
        seed(seed_value)
        
        # This is the logic from your old `initialize_game_state`
        self.state = {} # Use a dictionary for state
        self.state['ticks'] = 0
        self.state['crashed'] = False
        self.state['distance'] = 0.0
        self.state['sensors_enabled'] = True

        road = Road(SCREEN_WIDTH, SCREEN_HEIGHT, LANE_COUNT)
        self.state['road'] = road
        middle_lane = road.middle_lane()
        lane_height = road.get_lane_height()
        
        ego_velocity = Vector(10, 0)
        ego = Car("yellow", ego_velocity, lane=middle_lane, target_height=int(lane_height * 0.8))
        ego_sprite = ego.sprite
        ego.x = (SCREEN_WIDTH // 2) - (ego_sprite.get_width() // 2)  # Match real game starting position
        ego.y = int((middle_lane.y_start + middle_lane.y_end) / 2 - ego_sprite.get_height() / 2)
        self.state['ego'] = ego

        sensor_options = [
            (90, "front"), (135, "right_front"), (180, "right_side"),
            (225, "right_back"), (270, "back"), (315, "left_back"),
            (0, "left_side"), (45, "left_front"), (22.5, "left_side_front"),
            (67.5, "front_left_front"), (112.5, "front_right_front"), (157.5, "right_side_front"),
            (202.5, "right_side_back"), (247.5, "back_right_back"),
            (292.5, "back_left_back"), (337.5, "left_side_back")
        ]
        state_wrapper = StateWrapper(self.state)
        self.state['sensors'] = [Sensor(ego, angle, name, state_wrapper) for angle, name in sensor_options]
        
        self.state['car_bucket'] = []
        for i in range(LANE_COUNT * 2):
            color = random_choice(["blue", "red"])
            car = Car(color, Vector(8, 0), target_height=int(lane_height * 0.8))
            self.state['car_bucket'].append(car)

        self.state['cars'] = [ego]
        
        return self._get_current_state_dto()

    def step(self, action_name: str) -> Tuple[Dict, bool, bool]:
        """
        Executes one time-step of the game.
        Returns: (current_state_dto, terminated, truncated)
        """
        self.state['ticks'] += 1

        # 1. Handle action and update ego car
        self._handle_action(action_name)
        self.state['ego'].update(self.state['ego']) # Ego car updates relative to itself
        self.state['distance'] += self.state['ego'].velocity.x

        # 2. Update environment (other cars, sensors)
        self._update_other_cars()
        self._remove_passed_cars()
        self._place_new_cars()
        for sensor in self.state['sensors']:
            sensor.update()

        # 3. Check for crash conditions
        crashed = self._check_collisions()
        self.state['crashed'] = crashed

        # 4. Determine if the episode is over
        terminated = self.state['crashed']
        truncated = self.state['ticks'] >= MAX_TICKS_PER_EPISODE
        
        # 5. Render if in verbose mode
        if self.verbose:
            self._render()

        return self._get_current_state_dto(), terminated, truncated

    def _get_current_state_dto(self) -> Dict:
        """Captures the current state in a dictionary matching your DTO."""
        sensor_data = {s.name: s.reading for s in self.state['sensors']}
        return {
            "did_crash": self.state['crashed'],
            "elapsed_ticks": self.state['ticks'],
            "distance": self.state['distance'],
            "velocity": {"x": self.state['ego'].velocity.x, "y": self.state['ego'].velocity.y},
            "sensors": sensor_data
        }

    def _handle_action(self, action: str):
        # Your game's action logic
        if action == "ACCELERATE": self.state['ego'].speed_up()
        elif action == "DECELERATE": self.state['ego'].slow_down()
        elif action == "STEER_LEFT": self.state['ego'].turn(-0.1)
        elif action == "STEER_RIGHT": self.state['ego'].turn(0.1)

    def _check_collisions(self) -> bool:
        # Check for car collisions
        for car in self.state['cars']:
            if car != self.state['ego'] and self.state['ego'].rect.colliderect(car.rect):
                return True
        # Check for wall collisions
        for wall in self.state['road'].walls:
            if self.state['ego'].rect.colliderect(wall.rect):
                return True
        return False

    def _render(self):
        """Draws the current game state to the screen."""
        self.clock.tick(60) # Regulate speed for visualization
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.state['road'].surface, (0, 0))
        for wall in self.state['road'].walls: wall.draw(self.screen)
        for car in self.state['cars']: self.screen.blit(car.sprite, (car.x, car.y))
        for sensor in self.state['sensors']: sensor.draw(self.screen)
        # Add velocity display drawing here if desired
        pygame.display.flip()

    def close(self):
        if self.verbose:
            pygame.quit()
    
    # --- Helper methods for car management (adapted from your code) ---
    def _update_other_cars(self):
        for car in self.state['cars']:
            if car != self.state['ego']:
                car.update(self.state['ego'])

    def _remove_passed_cars(self):
        ego_x = self.state['ego'].x
        min_x = ego_x - (SCREEN_WIDTH * 0.75)
        max_x = ego_x + (SCREEN_WIDTH * 1.5)
        
        cars_to_keep = [self.state['ego']]
        for car in self.state['cars']:
            if car != self.state['ego']:
                if min_x < car.x < max_x:
                    cars_to_keep.append(car)
                else:
                    self.state['car_bucket'].append(car)
        self.state['cars'] = cars_to_keep

    def _place_new_cars(self):
        if len(self.state['cars']) > LANE_COUNT + 1: return
        if not self.state['car_bucket']: return
        if py_random.random() > 0.1: return # Only place cars occasionally

        ego_x = self.state['ego'].x
        open_lanes = [lane for lane in self.state['road'].lanes if not any(c.lane == lane for c in self.state['cars'])]
        if not open_lanes: return
        
        lane = random_choice(open_lanes)
        car = self.state['car_bucket'].pop()
        
        x_offset = ego_x + SCREEN_WIDTH * py_random.uniform(0.5, 1.0)
        car.x = x_offset
        car.y = int((lane.y_start + lane.y_end) / 2 - car.sprite.get_height() / 2)
        car.lane = lane
        car.velocity.x = self.state['ego'].velocity.x + py_random.uniform(-2, 5)
        self.state['cars'].append(car)