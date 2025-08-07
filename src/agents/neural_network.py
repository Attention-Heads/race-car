import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import copy


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for lane switching decisions.
    
    Architecture:
    - Input: 117 features (16 sensors × 7 timesteps + 5 additional features)
    - Hidden layers: 256 -> 128 -> 64 neurons (ReLU activation)
    - Output: 3 Q-values (left, right, do_nothing)
    """
    
    def __init__(self, state_size: int = 117, action_size: int = 3, 
                 hidden_layers: Tuple[int, ...] = (256, 128, 64),
                 dropout_rate: float = 0.1):
        """
        Initialize DQN network.
        
        Args:
            state_size: Size of input state vector
            action_size: Number of possible actions (3 for lane switching)
            hidden_layers: Sizes of hidden layers
            dropout_rate: Dropout rate for regularization
        """
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_size = hidden_size
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor [batch_size, state_size]
            
        Returns:
            Q-values tensor [batch_size, action_size]
        """
        return self.network(state)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Input state as numpy array
            epsilon: Exploration probability
            
        Returns:
            Action index (0=left, 1=right, 2=nothing)
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        # Convert to tensor and get Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()


class DQNTrainer:
    """
    Trainer class for DQN with target network and experience replay.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.0001, gamma: float = 0.99,
                 target_update_frequency: int = 1000,
                 device: str = None):
        """
        Initialize DQN trainer.
        
        Args:
            state_size: Size of input state vector
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            target_update_frequency: How often to update target network
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.update_count = 0
        
        # Create main and target networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability
    
    def train_step(self, states: np.ndarray, actions: np.ndarray, 
                   rewards: np.ndarray, next_states: np.ndarray, 
                   dones: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Perform one training step.
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of next states
            dones: Batch of episode termination flags
            weights: Importance sampling weights (for prioritized replay)
            
        Returns:
            Tuple of (loss_value, td_errors)
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        
        if weights is not None:
            # Weighted loss for prioritized replay
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        else:
            loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Return TD errors for prioritized replay
        td_errors_np = td_errors.abs().detach().cpu().numpy().flatten()
        
        return loss.item(), td_errors_np
    
    def update_target_network(self):
        """Update target network with current network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Get action from main network."""
        return self.q_network.get_action(state, epsilon)
    
    def save_model(self, filepath: str):
        """Save model state dict."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state dict."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
    
    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        self.q_network.eval()
        self.target_network.eval()
    
    def set_train_mode(self):
        """Set main network to training mode."""
        self.q_network.train()
        self.target_network.eval()  # Target network always in eval mode


class DoubleDQNTrainer(DQNTrainer):
    """
    Double DQN trainer to reduce overestimation bias.
    Uses main network to select actions and target network to evaluate them.
    """
    
    def train_step(self, states: np.ndarray, actions: np.ndarray, 
                   rewards: np.ndarray, next_states: np.ndarray, 
                   dones: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """
        Training step with Double DQN update rule.
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        
        if weights is not None:
            loss = (weights.unsqueeze(1) * td_errors.pow(2)).mean()
        else:
            loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        td_errors_np = td_errors.abs().detach().cpu().numpy().flatten()
        return loss.item(), td_errors_np


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture with separate value and advantage streams.
    """
    
    def __init__(self, state_size: int = 117, action_size: int = 3,
                 hidden_layers: Tuple[int, ...] = (256, 128),
                 dropout_rate: float = 0.1):
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()