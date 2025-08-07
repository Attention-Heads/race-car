import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple


# Experience tuple for storing transitions
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    Stores experiences and provides random sampling for training.
    """
    
    def __init__(self, buffer_size: int = 100000, batch_size: int = 32):
        """
        Initialize replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Size of batches to sample for training
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken (0=left, 1=right, 2=nothing)
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample random batch of experiences.
        
        Args:
            batch_size: Size of batch to sample (uses default if None)
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp.state for exp in batch], dtype=np.float32)
        actions = np.array([exp.action for exp in batch], dtype=np.int32)
        rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
        next_states = np.array([exp.next_state for exp in batch], dtype=np.float32)
        dones = np.array([exp.done for exp in batch], dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones
    
    def size(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
    
    def is_ready(self, min_size: int = None) -> bool:
        """
        Check if buffer has enough experiences for training.
        
        Args:
            min_size: Minimum required size (uses batch_size if None)
        """
        if min_size is None:
            min_size = self.batch_size
        return len(self.buffer) >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.
    Samples experiences based on their temporal difference error.
    """
    
    def __init__(self, buffer_size: int = 100000, batch_size: int = 32, 
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Size of batches to sample for training
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Amount to increment beta per sample
        """
        super().__init__(buffer_size, batch_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities = deque(maxlen=buffer_size)
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, priority: float = None):
        """
        Add experience to buffer with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
            priority: TD error priority (uses max if None)
        """
        super().add(state, action, reward, next_state, done)
        
        if priority is None:
            priority = self.max_priority
        
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Sample batch based on priorities.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        priorities = priorities ** self.alpha
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), size=batch_size, 
                                 replace=False, p=probabilities)
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize weights
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states = np.array([exp.state for exp in batch], dtype=np.float32)
        actions = np.array([exp.action for exp in batch], dtype=np.int32)
        rewards = np.array([exp.reward for exp in batch], dtype=np.float32)
        next_states = np.array([exp.next_state for exp in batch], dtype=np.float32)
        dones = np.array([exp.done for exp in batch], dtype=np.bool_)
        
        return states, actions, rewards, next_states, dones, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities for given indices.
        
        Args:
            indices: Indices of experiences to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)


class CircularBuffer:
    """
    Circular buffer for efficient memory usage during training.
    """
    
    def __init__(self, capacity: int, state_shape: Tuple[int, ...]):
        """
        Initialize circular buffer.
        
        Args:
            capacity: Maximum number of experiences
            state_shape: Shape of state vectors
        """
        self.capacity = capacity
        self.size = 0
        self.index = 0
        
        # Pre-allocate arrays for efficiency
        self.states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity,) + state_shape, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to circular buffer."""
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample random batch from buffer."""
        indices = np.random.choice(self.size, size=min(batch_size, self.size), replace=False)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences."""
        return self.size >= min_size