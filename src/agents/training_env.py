"""
Training environment wrapper for DQN agent integration with the race car game.
"""

import pygame
import numpy as np
from typing import Dict, Tuple, Optional, Any
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.game.core import GameState, initialize_game_state, update_game, handle_action
from src.game.agent import RuleBasedAgent
from src.agents.dqn_agent import DQNAgent
from src.mathematics.vector import Vector
from src.mathematics.randomizer import seed


class TrainingEnvironment:
    """
    Environment wrapper for training DQN agents in the race car simulation.
    Manages episode lifecycle, state collection, and reward computation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training environment.
        
        Args:
            config: Configuration dictionary containing environment and training settings
        """
        self.config = config
        self.env_config = config.get('environment', {})
        self.training_config = config.get('training', {})
        
        # Environment parameters
        self.max_episode_steps = self.training_config.get('max_episode_steps', 3600)
        self.verbose = self.env_config.get('verbose', False)
        self.fps = self.env_config.get('fps', 60)
        
        # Initialize pygame if running in verbose mode
        if self.verbose:
            pygame.init()
            self.screen = pygame.display.set_mode((
                self.env_config.get('screen_width', 1600),
                self.env_config.get('screen_height', 1200)
            ))
            pygame.display.set_caption("DQN Training - Race Car")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
            self.clock = None
        
        # Game state
        self.game_state = None
        self.rule_based_agent = None
        self.dqn_agent = None
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        self.total_steps = 0
        
        # Statistics
        self.episode_stats = []
        self.training_stats = {
            'total_episodes': 0,
            'total_crashes': 0,
            'best_distance': 0.0,
            'average_distance': 0.0,
            'success_rate': 0.0
        }
    
    def reset(self, seed_value: Optional[str] = None) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Args:
            seed_value: Random seed for reproducibility
            
        Returns:
            Initial state observation
        """
        # Seed randomizer
        if seed_value is not None:
            seed(seed_value)
        else:
            seed(None)  # Random seed
        
        # Initialize game state
        initialize_game_state("http://localhost:9052", seed_value)
        
        # Import STATE after initialization to avoid circular imports
        from src.game.core import STATE
        self.game_state = STATE
        
        # Create fresh rule-based agent for cruise control
        self.rule_based_agent = RuleBasedAgent()
        
        # Reset episode tracking
        self.current_step = 0
        
        # Get initial state
        if self.dqn_agent:
            # Let DQN agent process initial state
            sensor_data = self._get_sensor_data()
            initial_state = self.dqn_agent.get_action(
                sensor_data=sensor_data,
                velocity=self.game_state.ego.velocity,
                distance=self.game_state.distance,
                current_tick=self.current_step,
                is_performing_maneuver=self.rule_based_agent.is_performing_maneuver,
                training=True
            )
            
            return self.dqn_agent.current_state
        
        return np.zeros(117)  # Default state if no DQN agent
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action from DQN agent (0=left, 1=right, 2=nothing)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Handle DQN lane change decisions
        if action in [0, 1] and self.dqn_agent:
            success = self.dqn_agent.execute_lane_change(action, self.current_step)
            if success:
                # Execute lane change via rule-based agent
                direction = "left" if action == 0 else "right"
                self.rule_based_agent.execute_lane_change(direction)
        
        # Get cruise control actions from rule-based agent
        sensor_data = self._get_sensor_data()
        cruise_actions = self.rule_based_agent.get_actions(sensor_data)
        
        # Execute cruise control action
        if cruise_actions:
            cruise_action = cruise_actions[0]
            handle_action(cruise_action)
        
        # Update game state
        self._update_game_state()
        
        # Check for crashes and termination
        done = self._check_termination()
        
        # Calculate reward using DQN agent
        reward = 0.0
        if self.dqn_agent:
            reward = self.dqn_agent.step(
                sensor_data=sensor_data,
                velocity=self.game_state.ego.velocity,
                distance=self.game_state.distance,
                current_tick=self.current_step,
                crashed=self.game_state.crashed,
                is_performing_maneuver=self.rule_based_agent.is_performing_maneuver
            )
        
        # Get next state
        next_state = self.dqn_agent.current_state if self.dqn_agent else np.zeros(117)
        
        # Prepare info dictionary
        info = {
            'distance': self.game_state.distance,
            'velocity': self.game_state.ego.velocity.x,
            'crashed': self.game_state.crashed,
            'step': self.current_step,
            'lane': self.dqn_agent.lane_tracker.current_lane if self.dqn_agent else 2
        }
        
        # Render if verbose
        if self.verbose:
            self._render()
        
        self.current_step += 1
        self.total_steps += 1
        
        return next_state, reward, done, info
    
    def set_dqn_agent(self, agent: DQNAgent):
        """Set the DQN agent for training."""
        self.dqn_agent = agent
    
    def _get_sensor_data(self) -> Dict[str, Optional[float]]:
        """Get current sensor readings."""
        return {
            sensor.name: sensor.reading 
            for sensor in self.game_state.sensors
        }
    
    def _update_game_state(self):
        """Update the game simulation state."""
        from src.game.core import update_cars, remove_passed_cars, place_car
        
        # Update distance
        self.game_state.distance += self.game_state.ego.velocity.x
        
        # Update cars
        update_cars()
        remove_passed_cars()
        place_car()
        
        # Update sensors
        for sensor in self.game_state.sensors:
            sensor.update()
        
        # Check collisions
        self._check_collisions()
    
    def _check_collisions(self):
        """Check for collisions with cars and walls."""
        # Check car collisions
        for car in self.game_state.cars:
            if car != self.game_state.ego:
                if self._intersects(self.game_state.ego.rect, car.rect):
                    self.game_state.crashed = True
                    return
        
        # Check wall collisions
        for wall in self.game_state.road.walls:
            if self._intersects(self.game_state.ego.rect, wall.rect):
                self.game_state.crashed = True
                return
    
    def _intersects(self, rect1: pygame.Rect, rect2: pygame.Rect) -> bool:
        """Check if two rectangles intersect."""
        return rect1.colliderect(rect2)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        return (
            self.game_state.crashed or 
            self.current_step >= self.max_episode_steps
        )
    
    def _render(self):
        """Render the game state (for debugging)."""
        if not self.screen:
            return
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw road
        self.screen.blit(self.game_state.road.surface, (0, 0))
        
        # Draw walls
        for wall in self.game_state.road.walls:
            wall.draw(self.screen)
        
        # Draw cars
        for car in self.game_state.cars:
            if car.sprite:
                self.screen.blit(car.sprite, (car.x, car.y))
        
        # Draw sensors
        if self.game_state.sensors_enabled:
            for sensor in self.game_state.sensors:
                sensor.draw(self.screen)
        
        # Draw statistics
        self._draw_stats()
        
        pygame.display.flip()
        
        if self.clock:
            self.clock.tick(self.fps)
    
    def _draw_stats(self):
        """Draw training statistics on screen."""
        if not self.screen:
            return
        
        font = pygame.font.SysFont("monospace", 16)
        
        stats_text = [
            f"Episode: {self.episode_count}",
            f"Step: {self.current_step}",
            f"Distance: {self.game_state.distance:.1f}",
            f"Velocity: {self.game_state.ego.velocity.x:.1f}",
            f"Lane: {self.dqn_agent.lane_tracker.current_lane if self.dqn_agent else 'N/A'}",
            f"Epsilon: {self.dqn_agent.epsilon:.3f}" if self.dqn_agent else "",
            f"Reward: {self.dqn_agent.episode_reward:.1f}" if self.dqn_agent else ""
        ]
        
        y_offset = 10
        for text in stats_text:
            if text:  # Skip empty strings
                text_surface = font.render(text, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, y_offset))
                y_offset += 20
    
    def close(self):
        """Clean up environment resources."""
        if self.verbose and pygame.get_init():
            pygame.quit()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        if len(self.episode_stats) > 0:
            distances = [stats['distance'] for stats in self.episode_stats]
            crashes = [stats['crashed'] for stats in self.episode_stats]
            
            self.training_stats.update({
                'total_episodes': len(self.episode_stats),
                'total_crashes': sum(crashes),
                'best_distance': max(distances),
                'average_distance': np.mean(distances),
                'success_rate': 1.0 - (sum(crashes) / len(crashes)) if crashes else 1.0
            })
        
        return self.training_stats.copy()
    
    def end_episode(self) -> Dict[str, Any]:
        """End current episode and collect statistics."""
        episode_info = {
            'episode': self.episode_count,
            'steps': self.current_step,
            'distance': self.game_state.distance,
            'crashed': self.game_state.crashed,
            'final_velocity': self.game_state.ego.velocity.x,
            'reward': self.dqn_agent.episode_reward if self.dqn_agent else 0.0
        }
        
        self.episode_stats.append(episode_info)
        self.episode_count += 1
        
        # End episode for DQN agent
        if self.dqn_agent:
            return self.dqn_agent.end_episode(
                final_distance=self.game_state.distance,
                crashed=self.game_state.crashed
            )
        
        return episode_info


def create_training_environment(config: Dict[str, Any]) -> TrainingEnvironment:
    """
    Factory function to create training environment.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured TrainingEnvironment instance
    """
    return TrainingEnvironment(config)