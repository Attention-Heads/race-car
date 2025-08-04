import pygame
from time import sleep
import os
import json
import csv
import datetime
import random as py_random
import math
import requests
from typing import List, Optional
from ..mathematics.randomizer import seed, random_choice, random_number
from ..elements.car import Car
from ..elements.road import Road
from ..elements.sensor import Sensor
from ..mathematics.vector import Vector

# Define constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1200
LANE_COUNT = 5
CAR_COLORS = ['yellow', 'blue', 'red']
MAX_TICKS = 60 * 60  # 60 seconds @ 60 fps
MAX_MS = 60 * 1000600   # 60 seconds flat

# Define game state
class GameState:
    def __init__(self, api_url: str):
        self.ego = None
        self.cars = []
        self.car_bucket = []
        self.sensors = []
        self.road = None
        self.statistics = None
        self.sensors_enabled = True
        self.api_url = api_url
        self.crashed = False
        self.elapsed_game_time = 0
        self.distance = 0
        self.latest_action = "NOTHING"
        self.ticks = 0
        self.game_data = []  # Store game data for logging

STATE = None


def intersects(rect1, rect2):
    return rect1.colliderect(rect2)


def draw_velocity_display(screen, ego_car):
    """
    Draw a visual velocity display showing X and Y velocity components.
    """
    # Position for the velocity display (top right corner)
    display_x = SCREEN_WIDTH - 300
    display_y = 100
    
    # Get velocity values
    vel_x = ego_car.velocity.x
    vel_y = ego_car.velocity.y
    speed = math.sqrt(vel_x**2 + vel_y**2)
    
    # Background panel
    panel_width = 280
    panel_height = 120
    panel_rect = pygame.Rect(display_x, display_y, panel_width, panel_height)
    pygame.draw.rect(screen, (0, 0, 0, 180), panel_rect)  # Semi-transparent black
    pygame.draw.rect(screen, (255, 255, 255), panel_rect, 2)  # White border
    
    # Initialize font (you might want to do this once globally)
    try:
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 20)
    except:
        font = pygame.font.SysFont('Arial', 24)
        small_font = pygame.font.SysFont('Arial', 20)
    
    # Text displays
    text_y = display_y + 10
    
    # Speed text
    speed_text = font.render(f"Speed: {speed:.1f}", True, (255, 255, 255))
    screen.blit(speed_text, (display_x + 10, text_y))
    text_y += 25
    
    # X velocity text and bar
    vel_x_text = small_font.render(f"X: {vel_x:.1f}", True, (255, 255, 255))
    screen.blit(vel_x_text, (display_x + 10, text_y))
    
    # X velocity bar
    bar_x = display_x + 80
    bar_y = text_y + 2
    bar_width = 180
    bar_height = 15
    
    # Background bar
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
    
    # Velocity bar (scale velocity to fit bar, max expected velocity ~60)
    max_vel = 60
    vel_x_normalized = max(0, min(1, vel_x / max_vel))  # Clamp to 0-1 range
    bar_fill_width = vel_x_normalized * bar_width
    
    # X velocity bar (0 to 60, green bar from left to right)
    bar_fill_x = bar_x
    color = (0, 255, 0)  # Green
    
    pygame.draw.rect(screen, color, (bar_fill_x, bar_y, bar_fill_width, bar_height))
    
    # No center line needed for 0-60 range
    
    text_y += 25
    
    # Y velocity text and bar
    vel_y_text = small_font.render(f"Y: {vel_y:.1f}", True, (255, 255, 255))
    screen.blit(vel_y_text, (display_x + 10, text_y))
    
    # Y velocity bar
    bar_y = text_y + 2
    
    # Background bar
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
    
    # Velocity bar (keep Y velocity as -20 to +20 range with center line)
    max_vel_y = 20
    vel_y_normalized = max(-1, min(1, vel_y / max_vel_y))
    bar_fill_width = abs(vel_y_normalized) * (bar_width // 2)
    
    if vel_y >= 0:
        # Positive velocity (moving down) - yellow bar from center to right
        bar_fill_x = bar_x + (bar_width // 2)
        color = (255, 255, 0)  # Yellow
    else:
        # Negative velocity (moving up) - blue bar from center to left
        bar_fill_x = bar_x + (bar_width // 2) - bar_fill_width
        color = (0, 150, 255)  # Blue
    
    pygame.draw.rect(screen, color, (bar_fill_x, bar_y, bar_fill_width, bar_height))
    
    # Center line for Y velocity
    center_x = bar_x + (bar_width // 2)
    pygame.draw.line(screen, (255, 255, 255), (center_x, bar_y), (center_x, bar_y + bar_height), 2)
    
    # Velocity vector arrow (mini visualization)
    arrow_center_x = display_x + panel_width - 60
    arrow_center_y = display_y + 60
    arrow_scale = 3  # Scale factor for the arrow
    
    # Draw arrow background circle
    pygame.draw.circle(screen, (30, 30, 30), (arrow_center_x, arrow_center_y), 25)
    pygame.draw.circle(screen, (255, 255, 255), (arrow_center_x, arrow_center_y), 25, 1)
    
    # Draw velocity arrow
    if speed > 0.1:  # Only draw if there's meaningful velocity
        arrow_end_x = arrow_center_x + (vel_x * arrow_scale)
        arrow_end_y = arrow_center_y + (vel_y * arrow_scale)
        
        # Clamp arrow to circle
        arrow_dx = arrow_end_x - arrow_center_x
        arrow_dy = arrow_end_y - arrow_center_y
        arrow_length = math.sqrt(arrow_dx**2 + arrow_dy**2)
        if arrow_length > 20:
            arrow_dx = (arrow_dx / arrow_length) * 20
            arrow_dy = (arrow_dy / arrow_length) * 20
            arrow_end_x = arrow_center_x + arrow_dx
            arrow_end_y = arrow_center_y + arrow_dy
        
        # Draw arrow line
        pygame.draw.line(screen, (255, 255, 255), (arrow_center_x, arrow_center_y), 
                        (arrow_end_x, arrow_end_y), 3)
        
        # Draw arrowhead
        if arrow_length > 5:
            angle = math.atan2(arrow_dy, arrow_dx)
            arrowhead_length = 8
            arrowhead_angle = 0.5
            
            # Calculate arrowhead points
            head1_x = arrow_end_x - arrowhead_length * math.cos(angle - arrowhead_angle)
            head1_y = arrow_end_y - arrowhead_length * math.sin(angle - arrowhead_angle)
            head2_x = arrow_end_x - arrowhead_length * math.cos(angle + arrowhead_angle)
            head2_y = arrow_end_y - arrowhead_length * math.sin(angle + arrowhead_angle)
            
            pygame.draw.line(screen, (255, 255, 255), (arrow_end_x, arrow_end_y), (head1_x, head1_y), 2)
            pygame.draw.line(screen, (255, 255, 255), (arrow_end_x, arrow_end_y), (head2_x, head2_y), 2)

def get_environment_state():
    """
    Capture the current environment state for logging.
    Returns only the essential data that matches RaceCarPredictRequestDto.
    """
    # Get sensor readings
    sensor_data = {}
    for sensor in STATE.sensors:
        sensor_data[sensor.name] = sensor.reading
    
    # Return only the fields needed for model predictions
    return {
        "did_crash": STATE.crashed,
        "elapsed_ticks": STATE.ticks,
        "distance": STATE.distance,
        "velocity": {"x": STATE.ego.velocity.x, "y": STATE.ego.velocity.y},
        "sensors": sensor_data
    }

# Game logic
def handle_action(action: str):
    if action == "ACCELERATE":
        STATE.ego.speed_up()
    elif action == "DECELERATE":
        STATE.ego.slow_down()
    elif action == "STEER_LEFT":
        STATE.ego.turn(-0.1)
    elif action == "STEER_RIGHT":
        STATE.ego.turn(0.1)
    else:
        pass

def update_cars():
    for car in STATE.cars:
        car.update(STATE.ego)


def remove_passed_cars():
    min_distance = -1000
    max_distance = SCREEN_WIDTH + 1000
    cars_to_keep = []
    cars_to_retire = []

    for car in STATE.cars:
        if car.x < min_distance or car.x > max_distance:
            cars_to_retire.append(car)
        else:
            cars_to_keep.append(car)

    for car in cars_to_retire:
        STATE.car_bucket.append(car)
        car.lane = None

    STATE.cars = cars_to_keep

def place_car():
    if len(STATE.cars) > LANE_COUNT:
        return

    speed_coeff_modifier = 5
    x_offset_behind = -0.5
    x_offset_in_front = 1.5

    open_lanes = [lane for lane in STATE.road.lanes if not any(c.lane == lane for c in STATE.cars if c != STATE.ego)]
    lane = random_choice(open_lanes)
    x_offset = random_choice([x_offset_behind, x_offset_in_front])
    horizontal_velocity_coefficient = random_number() * speed_coeff_modifier

    car = STATE.car_bucket.pop() if STATE.car_bucket else None
    if not car:
        return

    velocity_x = STATE.ego.velocity.x + horizontal_velocity_coefficient if x_offset == x_offset_behind else STATE.ego.velocity.x - horizontal_velocity_coefficient
    car.velocity = Vector(velocity_x, 0)
    STATE.cars.append(car)

    car_sprite = car.sprite
    car.x = (SCREEN_WIDTH * x_offset) - (car_sprite.get_width() // 2)
    car.y = int((lane.y_start + lane.y_end) / 2 - car_sprite.get_height() / 2)
    car.lane = lane


def get_action():
    """
    Reads pygame events and returns an action string based on arrow keys or spacebar.
    Up: STEER_LEFT, Down: STEER_RIGHT, Left: DECELERATE, Right: ACCELERATE, Space: NOTHING
    """

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()


    # Holding down keys
    keys = pygame.key.get_pressed()

    # Priority: accelerate, decelerate, steer left, steer right, nothing
    if keys[pygame.K_RIGHT]:
        return "ACCELERATE"
    if keys[pygame.K_LEFT]:
        return "DECELERATE"
    if keys[pygame.K_UP]:
        return "STEER_LEFT"
    if keys[pygame.K_DOWN]:
        return "STEER_RIGHT"
    if keys[pygame.K_SPACE]:
        return "NOTHING"

    # Just clicking once and it keeps doing it until a new press
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                return "ACCELERATE"
            elif event.key == pygame.K_LEFT:
                return "DECELERATE"
            elif event.key == pygame.K_UP:
                return "STEER_LEFT"
            elif event.key == pygame.K_DOWN:
                return "STEER_RIGHT"
            elif event.key == pygame.K_SPACE:
                return "NOTHING"

    
    # If no relevant key is pressed, repeat last action or do nothing
    #return STATE.latest_action if hasattr(STATE, "latest_action") else "NOTHING"
    return "NOTHING"

def get_action_json():
    """
    Get action depending on tick from the actions_log.json.
    Finds the action for the current STATE.ticks.
    """
    try:
        with open("actions_log.json", "r") as f:
            actions = json.load(f)
            for entry in actions:
                if entry.get("tick") == STATE.ticks:
                    return entry.get("action", "NOTHING")
            return "NOTHING"
    except FileNotFoundError:
        return "NOTHING"


def get_action_from_api():
    """
    Call the API to get an action based on current game state.
    Returns a single action string.
    """
    if not STATE.api_url:
        return "NOTHING"
    
    try:
        # Prepare the request data
        velocity_data = {
            'x': STATE.ego.velocity.x,
            'y': STATE.ego.velocity.y
        }
        
        sensors_data = {}
        for sensor in STATE.sensors:
            sensors_data[sensor.name] = sensor.reading
        
        request_data = {
            'did_crash': STATE.crashed,
            'elapsed_ticks': STATE.ticks,
            'distance': STATE.distance,
            'velocity': velocity_data,
            'sensors': sensors_data
        }
        
        # Make the API call
        response = requests.post(
            STATE.api_url,
            json=request_data,
            timeout=5  # 5 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            # Handle both single action and list of actions
            if 'action' in result:
                return result['action']
            elif 'actions' in result and result['actions']:
                return result['actions'][0]  # Take first action from list
            else:
                return "NOTHING"
        else:
            print(f"API call failed with status {response.status_code}")
            return "NOTHING"
            
    except Exception as e:
        print(f"Error calling API: {e}")
        return "NOTHING"


def initialize_game_state( api_url: str, seed_value: str, sensor_removal = 0):
    seed(seed_value)
    global STATE
    STATE = GameState(api_url)

    # Create environment
    STATE.road = Road(SCREEN_WIDTH, SCREEN_HEIGHT, LANE_COUNT)
    middle_lane = STATE.road.middle_lane()
    lane_height = STATE.road.get_lane_height()

    # Create ego car
    ego_velocity = Vector(10, 0)
    STATE.ego = Car("yellow", ego_velocity, lane=middle_lane, target_height=int(lane_height * 0.8))
    ego_sprite = STATE.ego.sprite
    STATE.ego.x = (SCREEN_WIDTH // 2) - (ego_sprite.get_width() // 2)
    STATE.ego.y = int((middle_lane.y_start + middle_lane.y_end) / 2 - ego_sprite.get_height() / 2)
    sensor_options = [
            (90, "front"),
            (135, "right_front"),
            (180, "right_side"),
            (225, "right_back"),
            (270, "back"),
            (315, "left_back"),
            (0, "left_side"),
            (45, "left_front"),
            (22.5, "left_side_front"),
            (67.5, "front_left_front"),
            (112.5, "front_right_front"),
            (157.5, "right_side_front"),
            (202.5, "right_side_back"),
            (247.5, "back_right_back"),
            (292.5, "back_left_back"),
            (337.5, "left_side_back"),
        ]

    for _ in range(sensor_removal): # Removes random sensors
        random_sensor = random_choice(sensor_options)
        sensor_options.remove(random_sensor)
    STATE.sensors = [
        Sensor(STATE.ego, angle, name, STATE)
        for angle, name in sensor_options
    ]

    # Create other cars and add to car bucket
    for i in range(0, LANE_COUNT - 1):
        car_colors = ["blue", "red"]
        color = random_choice(car_colors)
        car = Car(color, Vector(8, 0), target_height=int(lane_height * 0.8))
        STATE.car_bucket.append(car)

    STATE.cars = [STATE.ego]

def update_game(current_action: str):
    # Capture environment state before action
    env_state = get_environment_state()
    
    handle_action(current_action)
    STATE.distance += STATE.ego.velocity.x
    update_cars()
    remove_passed_cars()
    place_car()
    for sensor in STATE.sensors:
        sensor.update()
    
    # Log the state and action as a flat dictionary for CSV
    game_step = {
        "action": current_action,
        "did_crash": env_state["did_crash"],
        "elapsed_ticks": env_state["elapsed_ticks"], 
        "distance": env_state["distance"],
        "velocity_x": env_state["velocity"]["x"],
        "velocity_y": env_state["velocity"]["y"],
        **{f"sensor_{name}": value for name, value in env_state["sensors"].items()}
    }
    STATE.game_data.append(game_step)

    return STATE
    
# Main game loop
ACTION_LOG = []

def save_game_data(game_number: int, seed_value: str):
    """
    Save the game data to a CSV file in the data directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"game_{game_number:04d}_{timestamp}_seed_{seed_value}.csv"
    filepath = os.path.join("data", filename)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    if not STATE.game_data:
        print("No game data to save")
        return
    
    try:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            # Get all possible column names from the data
            fieldnames = set()
            for step in STATE.game_data:
                fieldnames.update(step.keys())
            fieldnames = sorted(list(fieldnames))  # Sort for consistent column order
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(STATE.game_data)
        
        print(f"Game data saved to: {filepath}")
        print(f"Final score: {STATE.distance:.2f}, Ticks: {STATE.ticks}, Crashed: {STATE.crashed}")
        print(f"Total data points: {len(STATE.game_data)}")
        
        # Also save a small metadata JSON file
        metadata_filepath = filepath.replace(".csv", "_metadata.json")
        metadata = {
            "game_number": game_number,
            "seed": seed_value,
            "timestamp": timestamp,
            "final_score": STATE.distance,
            "final_tick": STATE.ticks,
            "crashed": STATE.crashed,
            "elapsed_time": STATE.elapsed_game_time,
            "total_steps": len(STATE.game_data)
        }
        with open(metadata_filepath, "w") as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print(f"Error saving game data: {e}")

def game_loop(verbose: bool = True, log_actions: bool = True, log_path: str = "actions_log.json"):
    global STATE
    clock = pygame.time.Clock()
    screen = None
    actions = []
    if verbose:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Race Car Game")

    while True:
        delta = clock.tick(60)  # Limit to 60 FPS
        STATE.elapsed_game_time += delta
        STATE.ticks += 1

        if STATE.crashed or STATE.ticks > MAX_TICKS or STATE.elapsed_game_time > MAX_MS:
            print(f"Game over: Crashed: {STATE.crashed}, Ticks: {STATE.ticks}, Elapsed time: {STATE.elapsed_game_time} ms, Distance: {STATE.distance}")
            break

        if not actions:
            # Handle action - Call API to get action from your trained model
            action = get_action_from_api()
            actions.append(action)
        
        if actions:
            action = actions.pop(0)
        else:
            action = "NOTHING"

        # Log the action with tick
        if log_actions:
            ACTION_LOG.append({"tick": STATE.ticks, "action": action})

        # Update game with current action and log state
        update_game(action)

        print("Current action:", action)
        print("Current tick:", STATE.ticks)

        # Handle collisions
        for car in STATE.cars:
            if car != STATE.ego and intersects(STATE.ego.rect, car.rect):
                STATE.crashed = True
        
        # Check collision with walls
        for wall in STATE.road.walls:
            if intersects(STATE.ego.rect, wall.rect):
                STATE.crashed = True

        # Render game (only if verbose)
        if verbose:
            screen.fill((0, 0, 0))  # Clear the screen with black

            # Draw the road background
            screen.blit(STATE.road.surface, (0, 0))

            # Draw all walls
            for wall in STATE.road.walls:
                wall.draw(screen)

            # Draw all cars
            for car in STATE.cars:
                if car.sprite:
                    screen.blit(car.sprite, (car.x, car.y))
                    bounds = car.get_bounds()
                    color = (255, 0, 0) if car == STATE.ego else (0, 255, 0)
                    pygame.draw.rect(screen, color, bounds, width=2)
                else:
                    pygame.draw.rect(screen, (255, 255, 0) if car == STATE.ego else (0, 0, 255), car.rect)

            # Draw sensors if enabled
            if STATE.sensors_enabled:
                for sensor in STATE.sensors:
                    sensor.draw(screen)
            
            # Draw velocity display
            draw_velocity_display(screen, STATE.ego)

            pygame.display.flip()

def continuous_game_loop(verbose: bool = True, max_games: int = None):
    """
    Run games continuously, saving data after each game and restarting automatically.
    
    :param verbose: Whether to show the pygame window
    :param max_games: Maximum number of games to play (None for infinite)
    """
    game_number = 1
    
    if verbose:
        pygame.init()
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Race Car Game - Continuous Mode")
    
    print("Starting continuous game loop...")
    print("Press ESC during gameplay to quit, or close the window")
    
    try:
        while max_games is None or game_number <= max_games:
            print(f"\n--- Starting Game {game_number} ---")
            
            # Generate a new seed for each game
            seed_value = py_random.randint(100000, 999999)
            
            # Initialize new game state
            initialize_game_state("http://example.com/api/predict", seed_value)
            
            # Reset action log for this game
            global ACTION_LOG
            ACTION_LOG = []
            
            # Run the game
            game_loop(verbose=verbose)
            
            # Save the game data
            save_game_data(game_number, seed_value)
            
            # Check for quit event
            quit_requested = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_requested = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    quit_requested = True
            
            if quit_requested:
                print("Quit requested, stopping continuous loop...")
                break
                
            game_number += 1
            
            # Small delay before next game
            sleep(1)
            
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, stopping...")
    finally:
        if verbose:
            pygame.quit()
        print(f"Completed {game_number - 1} games total.")

# Initialization - not used
def init(api_url: str):
    global STATE
    STATE = GameState(api_url)
    print(f"Game initialized with API URL: {api_url}")


# Entry point
if __name__ == "__main__":
    # For continuous gameplay with data logging
    continuous_game_loop(verbose=True, max_games=None)  # None for infinite games
    
    # For single game (uncomment below and comment above)
    # seed_value = 565318
    # pygame.init()
    # initialize_game_state("http://example.com/api/predict", seed_value)
    # game_loop(verbose=True)
    # save_game_data(1, seed_value)
    # pygame.quit()