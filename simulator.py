import os
import sys
import random
from typing import Tuple, Dict, Any
import copy

# Add the project root to the path so we can import game modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment import GameState
from src.game.core import (
    initialize_game_state, handle_action, update_cars, remove_passed_cars, 
    place_car, STATE, intersects
)

class TrainingSimulator:
    """
    Headless version of the race car game for training.
    Simulates the game without pygame visualization.
    """
    
    def __init__(self, seed_value=None):
        self.seed_value = seed_value
        self.episode_count = 0
        self.previous_episodes = {}  # Store episode results for analysis
        
    def reset(self) -> GameState:
        """Reset the game environment for a new episode"""
        # Use different seed for each episode for variety
        episode_seed = self.seed_value if self.seed_value else random.randint(1, 1000000)
        episode_seed += self.episode_count  # Make each episode unique
        
        # Initialize pygame in headless mode
        import pygame
        import os
        os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Use dummy video driver for headless
        if not pygame.get_init():
            pygame.init()
        
        # Initialize game state (this sets up the global STATE)
        initialize_game_state("http://localhost:9052", episode_seed, sensor_removal=0)
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_count += 1
        
        return self._get_current_game_state()
    
    def step(self, action: str) -> Tuple[GameState, float, bool]:
        """
        Take one step in the environment
        Returns: (next_state, reward, done)
        """
        global STATE
        
        # Handle the action
        handle_action(action)
        
        # Update game state
        STATE.distance += STATE.ego.velocity.x
        update_cars()
        remove_passed_cars()
        place_car()
        
        # Update sensors
        for sensor in STATE.sensors:
            sensor.update()
        
        # Check for collisions
        crashed = False
        
        # Check collision with other cars
        for car in STATE.cars:
            if car != STATE.ego and intersects(STATE.ego.rect, car.rect):
                crashed = True
                break
        
        # Check collision with walls
        if not crashed:
            for wall in STATE.road.walls:
                if intersects(STATE.ego.rect, wall.rect):
                    crashed = True
                    break
        
        STATE.crashed = crashed
        STATE.ticks += 1
        STATE.elapsed_game_time += 16.67  # Approximate 60 FPS (1000ms/60fps)
        
        # Check if episode is done
        done = (crashed or 
                STATE.ticks >= 3600 or  # Max ticks (60 seconds)
                STATE.elapsed_game_time >= 60000)  # Max time in ms
        
        # Simple reward (will be overridden by environment's reward function)
        reward = 1.0 if not crashed else -10.0
        
        current_state = self._get_current_game_state()
        
        # Store episode result when done
        if done:
            self.previous_episodes[self.episode_count] = {
                'crashed': crashed,
                'distance': STATE.distance,
                'ticks': STATE.ticks,
                'score': reward
            }
        
        return current_state, reward, done
    
    def _get_current_game_state(self) -> GameState:
        """Convert the internal game state to our GameState format"""
        global STATE
        
        # Extract sensor readings
        sensors = {}
        for sensor in STATE.sensors:
            sensors[sensor.name] = sensor.reading
        
        # Get velocity
        velocity = {
            'x': int(STATE.ego.velocity.x),
            'y': int(STATE.ego.velocity.y)
        }
        
        # Get coordinates  
        coordinates = {
            'x': int(STATE.ego.x),
            'y': int(STATE.ego.y)
        }
        
        return GameState(
            sensors=sensors,
            velocity=velocity,
            coordinates=coordinates,
            distance=int(STATE.distance),
            elapsed_time_ms=int(STATE.elapsed_game_time),
            did_crash=STATE.crashed
        )
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get state in the format expected by the API"""
        game_state = self._get_current_game_state()
        
        return {
            'sensors': game_state.sensors,
            'velocity': game_state.velocity,
            'coordinates': game_state.coordinates,
            'distance': game_state.distance,
            'elapsed_time_ms': game_state.elapsed_time_ms,
            'did_crash': game_state.did_crash
        }

class HeadlessGameSimulator:
    """
    Alternative simulator that directly manipulates game objects
    without relying on the pygame-dependent game loop
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> GameState:
        """Reset for new episode"""
        # Initialize basic state
        self.distance = 0
        self.velocity_x = 10.0
        self.velocity_y = 0.0
        self.x = 800  # Screen center
        self.y = 600  # Screen center
        self.crashed = False
        self.ticks = 0
        self.time_ms = 0
        
        # Initialize sensor readings (no obstacles initially)
        self.sensors = {
            'left_side': 1000, 'left_side_front': 1000, 'left_front': 1000, 'front_left_front': 1000,
            'front': 1000, 'front_right_front': 1000, 'right_front': 1000, 'right_side_front': 1000,
            'right_side': 1000, 'right_side_back': 1000, 'right_back': 1000, 'back_right_back': 1000,
            'back': 1000, 'back_left_back': 1000, 'left_back': 1000, 'left_side_back': 1000
        }
        
        # Simple obstacle simulation
        self.obstacles = []
        self._spawn_obstacles()
        
        # Return initial state
        return GameState(
            sensors=self.sensors.copy(),
            velocity={'x': int(self.velocity_x), 'y': int(self.velocity_y)},
            coordinates={'x': int(self.x), 'y': int(self.y)},
            distance=int(self.distance),
            elapsed_time_ms=int(self.time_ms),
            did_crash=self.crashed
        )
    
    def _spawn_obstacles(self):
        """Spawn some obstacles for the car to avoid"""
        for _ in range(5):
            obstacle_x = random.uniform(1000, 5000)  # Ahead of car
            obstacle_y = random.uniform(200, 1000)   # Various lanes
            self.obstacles.append({'x': obstacle_x, 'y': obstacle_y})
    
    def step(self, action: str) -> Tuple[GameState, float, bool]:
        """Simplified step function"""
        
        # Handle action
        if action == "ACCELERATE":
            self.velocity_x = min(self.velocity_x + 0.1, 15.0)
        elif action == "DECELERATE":
            self.velocity_x = max(self.velocity_x - 0.1, 2.0)
        elif action == "STEER_LEFT":
            self.velocity_y -= 0.1
        elif action == "STEER_RIGHT":
            self.velocity_y += 0.1
        
        # Update position
        self.x += self.velocity_x
        self.y += self.velocity_y
        self.distance += self.velocity_x
        
        # Simple boundary checking (crash into walls)
        if self.y < 100 or self.y > 1100:
            self.crashed = True
        
        # Simple obstacle collision
        for obstacle in self.obstacles:
            if abs(self.x - obstacle['x']) < 50 and abs(self.y - obstacle['y']) < 50:
                self.crashed = True
        
        # Update sensors (simplified - just check nearby obstacles)
        self._update_sensors()
        
        # Update time
        self.ticks += 1
        self.time_ms += 16.67
        
        # Check if done
        done = self.crashed or self.ticks >= 3600 or self.time_ms >= 60000
        
        # Simple reward
        reward = 1.0 if not self.crashed else -10.0
        
        current_state = GameState(
            sensors=self.sensors.copy(),
            velocity={'x': int(self.velocity_x), 'y': int(self.velocity_y)},
            coordinates={'x': int(self.x), 'y': int(self.y)},
            distance=int(self.distance),
            elapsed_time_ms=int(self.time_ms),
            did_crash=self.crashed
        )
        
        return current_state, reward, done
    
    def _update_sensors(self):
        """Update sensor readings based on nearby obstacles"""
        # Reset all sensors to max range
        for key in self.sensors:
            self.sensors[key] = 1000
        
        # Check each obstacle
        for obstacle in self.obstacles:
            dx = obstacle['x'] - self.x
            dy = obstacle['y'] - self.y
            distance = (dx**2 + dy**2)**0.5
            
            if distance < 1000:  # Within sensor range
                # Simplified sensor logic - front sensors detect obstacles ahead
                if dx > 0:  # Obstacle is ahead
                    if abs(dy) < 100:  # Roughly in front
                        self.sensors['front'] = min(self.sensors['front'], distance)
                    elif dy < 0:  # Obstacle to the left
                        self.sensors['left_front'] = min(self.sensors['left_front'], distance)
                    else:  # Obstacle to the right
                        self.sensors['right_front'] = min(self.sensors['right_front'], distance)