import pygame
import random
import os
from src.game.core import initialize_game_state, game_loop
from dummy_model import DummyAgent
from environment import RaceCarEnvironment, GameState

# Global PPO components for inference
MODEL_PATH = "models/model.pth"
agent = None
ppo_env = None
action_buffer = []

def initialize_agent():
    """Initialize agent for inference"""
    global agent
    
    if agent is None:
        agent = DummyAgent()
        
        # Load trained model if it exists (dummy agent doesn't load)
        if os.path.exists(MODEL_PATH):
            agent.load(MODEL_PATH)
            print("Loaded trained model")
        else:
            print("No trained model found, using dummy agent")

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
    if os.path.exists(MODEL_PATH):
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
    global agent
    
    if agent is None:
        initialize_agent()

    # The game state is passed directly to the agent
    action = agent.act(state)
    
    return [action] # Return as a list of actions


if __name__ == '__main__':
    #seed_value = 565318
    seed_value = random.randint(0, 1000000)
    pygame.init()
    # Pass the action handler to the game state
    initialize_game_state(None, seed_value, action_handler=return_action_single)
    game_loop(verbose=True) # For pygame window
    pygame.quit()