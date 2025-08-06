import pygame
import random
import os
from src.game.core import initialize_game_state, game_loop
from model import PPOAgent
from environment import RaceCarEnvironment, GameState

# Global PPO components for inference
ppo_agent = None
ppo_env = None
action_buffer = []

def initialize_ppo():
    """Initialize PPO agent for inference"""
    global ppo_agent, ppo_env
    
    if ppo_agent is None:
        ppo_env = RaceCarEnvironment()
        ppo_agent = PPOAgent(
            state_size=ppo_env.state_size,
            action_size=ppo_env.action_size
        )
        
        # Load trained model if it exists
        model_path = "models/model.pth"
        if os.path.exists(model_path):
            ppo_agent.load(model_path)
            print("Loaded trained model")
        else:
            print("No trained model found, using random policy")

def return_action(state):
    """
    PPO-based action selection for the race car.
    Returns a batch of actions as required by the API.
    """
    global ppo_agent, ppo_env, action_buffer
    
    # Initialize PPO if not already done
    if ppo_agent is None:
        initialize_ppo()
    
    # Convert state dict to GameState object
    game_state = GameState(
        sensors=state.get('sensors', {}),
        velocity=state.get('velocity', {'x': 10, 'y': 0}),
        coordinates=state.get('coordinates', {'x': 800, 'y': 600}),
        distance=state.get('distance', 0),
        elapsed_time_ms=state.get('elapsed_time_ms', 0),
        did_crash=state.get('did_crash', False)
    )
    
    # Preprocess state for PPO
    processed_state = ppo_env.preprocess_state(game_state)
    
    # If we have a trained model, use it
    if os.path.exists("models/racecar.pth"):
        # Predict next batch of actions
        batch_size = 10
        action_indices = ppo_agent.predict_action_sequence(processed_state, batch_size)
        actions = [ppo_env.index_to_action(idx) for idx in action_indices]
    else:
        # Fallback to random actions if no trained model
        actions = []
        action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
        for _ in range(10):
            actions.append(random.choice(action_choices))
    
    return actions

def return_action_single(state):
    """
    Single action version for manual testing/debugging
    """
    global ppo_agent, ppo_env
    
    if ppo_agent is None:
        initialize_ppo()
    
    game_state = GameState(
        sensors=state.get('sensors', {}),
        velocity=state.get('velocity', {'x': 10, 'y': 0}),
        coordinates=state.get('coordinates', {'x': 800, 'y': 600}),
        distance=state.get('distance', 0),
        elapsed_time_ms=state.get('elapsed_time_ms', 0),
        did_crash=state.get('did_crash', False)
    )
    
    processed_state = ppo_env.preprocess_state(game_state)
    
    if os.path.exists("models/racecar.pth"):
        action_idx = ppo_agent.act(processed_state, training=False)
        action = ppo_env.index_to_action(action_idx)
    else:
        action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
        action = random.choice(action_choices)
    
    return action


if __name__ == '__main__':
    seed_value = 565318
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(verbose=True) # For pygame window
    pygame.quit()