import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import torch

from .neural_network import DQNTrainer, DoubleDQNTrainer, DuelingDQNNetwork
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from ..utils.state_processor import StateProcessor, LaneTracker
from ..utils.reward_calculator import RewardCalculator, AdaptiveRewardCalculator
from ..mathematics.vector import Vector


class DQNAgent:
    """
    Main DQN agent for lane switching decisions.
    Integrates state processing, neural network, experience replay, and reward calculation.
    """
    
    def __init__(self, 
                 state_size: int = 117,
                 action_size: int = 3,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 32,
                 target_update_frequency: int = 1000,
                 cooldown_ticks: int = 180,
                 use_double_dqn: bool = True,
                 use_prioritized_replay: bool = False,
                 device: str = None):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            action_size: Number of actions (3: left, right, nothing)
            learning_rate: Learning rate for neural network
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_size: Experience replay buffer size
            batch_size: Training batch size
            target_update_frequency: How often to update target network
            cooldown_ticks: Minimum ticks between lane changes (3 seconds = 180 ticks)
            use_double_dqn: Whether to use Double DQN
            use_prioritized_replay: Whether to use prioritized experience replay
            device: Device to run on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.cooldown_ticks = cooldown_ticks
        self.batch_size = batch_size
        
        # Initialize components
        self.state_processor = StateProcessor()
        self.lane_tracker = LaneTracker(initial_lane=2)  # Start in middle lane
        self.reward_calculator = AdaptiveRewardCalculator()
        
        # Initialize neural network trainer
        if use_double_dqn:
            self.trainer = DoubleDQNTrainer(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                gamma=gamma,
                target_update_frequency=target_update_frequency,
                device=device
            )
        else:
            self.trainer = DQNTrainer(
                state_size=state_size,
                action_size=action_size,
                learning_rate=learning_rate,
                gamma=gamma,
                target_update_frequency=target_update_frequency,
                device=device
            )
        
        # Initialize experience replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size, batch_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.training_losses = []
        
        # Current episode state
        self.current_state = None
        self.last_action = None
        self.episode_reward = 0.0
        self.episode_distance = 0.0
        
    def get_action(self, 
                   sensor_data: Dict[str, Optional[float]], 
                   velocity: Vector, 
                   distance: float, 
                   current_tick: int,
                   is_performing_maneuver: bool = False,
                   training: bool = True) -> int:
        """
        Get action from DQN agent.
        
        Args:
            sensor_data: Current sensor readings
            velocity: Current velocity vector
            distance: Total distance traveled
            current_tick: Current game tick
            is_performing_maneuver: Whether currently executing lane change
            training: Whether in training mode (affects exploration)
            
        Returns:
            Action index (0=left, 1=right, 2=nothing)
        """
        # Update lane tracker
        self.lane_tracker.update(current_tick, is_performing_maneuver)
        
        # Add sensor data to history
        self.state_processor.add_sensor_reading(sensor_data)
        
        # Check if we have enough sensor history
        if not self.state_processor.is_ready():
            return 2  # Do nothing until we have sensor history
        
        # Get time since last lane change for cooldown enforcement
        time_since_lane_change = self.lane_tracker.get_time_since_lane_change(current_tick)
        
        # Create state vector
        current_state = self.state_processor.get_stacked_state(
            lane=self.lane_tracker.current_lane,
            velocity=velocity,
            time_since_lane_change=time_since_lane_change,
            distance=distance
        )
        
        # Store current state for experience replay
        self.current_state = current_state
        
        # Enforce cooldown period
        if time_since_lane_change < self.cooldown_ticks:
            return 2  # Do nothing during cooldown
        
        # Get action from neural network
        epsilon = self.epsilon if training else 0.0  # No exploration during evaluation
        action = self.trainer.get_action(current_state, epsilon)
        
        # Validate action (check lane boundaries)
        if not self.lane_tracker.is_valid_action(action):
            return 2  # Fall back to do nothing for invalid actions
        
        self.last_action = action
        return action
    
    def step(self, 
             sensor_data: Dict[str, Optional[float]], 
             velocity: Vector, 
             distance: float, 
             current_tick: int,
             crashed: bool,
             is_performing_maneuver: bool = False) -> float:
        """
        Process one step of interaction with environment.
        
        Args:
            sensor_data: Current sensor readings
            velocity: Current velocity vector
            distance: Total distance traveled
            current_tick: Current game tick
            crashed: Whether the car crashed
            is_performing_maneuver: Whether currently executing lane change
            
        Returns:
            Reward for this step
        """
        # Calculate reward for previous action
        if self.current_state is not None and self.last_action is not None:
            time_since_lane_change = self.lane_tracker.get_time_since_lane_change(current_tick)
            
            reward = self.reward_calculator.calculate_reward(
                current_distance=distance,
                velocity=velocity,
                action=self.last_action,
                crashed=crashed,
                sensors=sensor_data,
                lane=self.lane_tracker.current_lane,
                time_since_lane_change=time_since_lane_change,
                is_performing_maneuver=is_performing_maneuver,
                attempted_invalid_action=False
            )
            
            # Get next state
            self.state_processor.add_sensor_reading(sensor_data)
            next_state = self.state_processor.get_stacked_state(
                lane=self.lane_tracker.current_lane,
                velocity=velocity,
                time_since_lane_change=time_since_lane_change,
                distance=distance
            )
            
            # Store experience in replay buffer
            self.replay_buffer.add(
                state=self.current_state,
                action=self.last_action,
                reward=reward,
                next_state=next_state,
                done=crashed
            )
            
            # Train if we have enough experiences
            if self.replay_buffer.is_ready() and len(self.replay_buffer.buffer) % 4 == 0:
                self._train_step()
            
            self.episode_reward += reward
            self.episode_distance = distance
            self.total_steps += 1
            
            return reward
        
        return 0.0
    
    def execute_lane_change(self, action: int, current_tick: int) -> bool:
        """
        Execute lane change action.
        
        Args:
            action: Lane change action (0=left, 1=right)
            current_tick: Current game tick
            
        Returns:
            True if lane change initiated, False otherwise
        """
        if action == 0:
            return self.lane_tracker.initiate_lane_change("left", current_tick)
        elif action == 1:
            return self.lane_tracker.initiate_lane_change("right", current_tick)
        return False
    
    def _train_step(self):
        """Perform one training step."""
        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            # Prioritized experience replay
            states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size)
            loss, td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, weights)
            
            # Update priorities
            priorities = td_errors + 1e-6  # Small epsilon for numerical stability
            self.replay_buffer.update_priorities(indices, priorities)
        else:
            # Standard experience replay
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            loss, _ = self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        self.training_losses.append(loss)
    
    def end_episode(self, final_distance: float, crashed: bool):
        """
        End current episode and update training parameters.
        
        Args:
            final_distance: Final distance achieved
            crashed: Whether episode ended in crash
        """
        self.episode_count += 1
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update reward calculator with training progress
        self.reward_calculator.set_training_progress(self.episode_count, 10000)
        
        # Reset episode state
        self._reset_episode()
        
        # Log episode statistics
        return {
            'episode': self.episode_count,
            'reward': self.episode_reward,
            'distance': final_distance,
            'crashed': crashed,
            'epsilon': self.epsilon,
            'buffer_size': self.replay_buffer.size()
        }
    
    def _reset_episode(self):
        """Reset agent state for new episode."""
        self.current_state = None
        self.last_action = None
        self.episode_reward = 0.0
        self.episode_distance = 0.0
        
        # Reset components
        self.state_processor.reset()
        self.lane_tracker = LaneTracker(initial_lane=2)
        self.reward_calculator.reset_episode()
    
    def save_model(self, filepath: str):
        """Save agent model."""
        self.trainer.save_model(filepath)
        
        # Save additional agent state
        agent_state = {
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'training_losses': self.training_losses
        }
        torch.save(agent_state, filepath.replace('.pth', '_agent.pth'))
    
    def load_model(self, filepath: str):
        """Load agent model."""
        self.trainer.load_model(filepath)
        
        # Load additional agent state
        try:
            agent_state = torch.load(filepath.replace('.pth', '_agent.pth'))
            self.episode_count = agent_state['episode_count']
            self.epsilon = agent_state['epsilon']
            self.total_steps = agent_state['total_steps']
            self.training_losses = agent_state['training_losses']
        except FileNotFoundError:
            print("Agent state file not found, using defaults")
    
    def set_eval_mode(self):
        """Set agent to evaluation mode (no exploration, no training)."""
        self.trainer.set_eval_mode()
    
    def set_train_mode(self):
        """Set agent to training mode."""
        self.trainer.set_train_mode()
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'buffer_size': self.replay_buffer.size(),
            'average_loss': np.mean(self.training_losses[-100:]) if self.training_losses else 0.0,
            'current_lane': self.lane_tracker.current_lane
        }


class EnsembleDQNAgent:
    """
    Ensemble of multiple DQN agents for improved robustness.
    """
    
    def __init__(self, num_agents: int = 3, **kwargs):
        """
        Initialize ensemble of DQN agents.
        
        Args:
            num_agents: Number of agents in ensemble
            **kwargs: Arguments passed to individual DQN agents
        """
        self.agents = [DQNAgent(**kwargs) for _ in range(num_agents)]
        self.num_agents = num_agents
    
    def get_action(self, *args, **kwargs) -> int:
        """Get action using majority voting."""
        actions = [agent.get_action(*args, **kwargs) for agent in self.agents]
        
        # Majority voting
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in actions:
            action_counts[action] += 1
        
        return max(action_counts, key=action_counts.get)
    
    def step(self, *args, **kwargs) -> float:
        """Step all agents."""
        rewards = [agent.step(*args, **kwargs) for agent in self.agents]
        return np.mean(rewards)
    
    def end_episode(self, *args, **kwargs):
        """End episode for all agents."""
        return [agent.end_episode(*args, **kwargs) for agent in self.agents]
    
    def save_model(self, filepath_prefix: str):
        """Save all agent models."""
        for i, agent in enumerate(self.agents):
            agent.save_model(f"{filepath_prefix}_agent_{i}.pth")
    
    def load_model(self, filepath_prefix: str):
        """Load all agent models."""
        for i, agent in enumerate(self.agents):
            agent.load_model(f"{filepath_prefix}_agent_{i}.pth")