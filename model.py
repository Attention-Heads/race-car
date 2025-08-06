import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import List, Tuple, Dict, Any
import os

class ActorCritic(nn.Module):
    def __init__(self, state_size: int = 12, action_size: int = 5, hidden_size: int = 128):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy network)
        self.actor_fc = nn.Linear(hidden_size, 64)
        self.actor_out = nn.Linear(64, action_size)
        
        # Critic head (value network) 
        self.critic_fc = nn.Linear(hidden_size, 64)
        self.critic_out = nn.Linear(64, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        # Shared layers
        x = F.relu(self.shared_fc1(state))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        x = self.dropout(x)
        
        # Actor (policy) output - action probabilities
        actor_x = F.relu(self.actor_fc(x))
        action_logits = self.actor_out(actor_x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic (value) output
        critic_x = F.relu(self.critic_fc(x))
        state_value = self.critic_out(critic_x)
        
        return action_probs, state_value
    
    def act(self, state):
        """Select action using current policy"""
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        return action.item(), action_log_prob, state_value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear() 
        self.values.clear()
        self.dones.clear()
    
    def get_batches(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.values),
            torch.BoolTensor(self.dones)
        )

class PPOAgent:
    def __init__(
        self,
        state_size: int = 12,
        action_size: int = 5,
        lr: float = 1e-4,           # Reduced from 3e-4 for stability
        gamma: float = 0.99,
        epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.03,    # Increased from 0.01 for exploration
        value_loss_coef: float = 1.0,  # Increased from 0.5 for value stability
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        rollout_length: int = 2048
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Memory
        self.memory = PPOMemory()
        
        # Training metrics
        self.training_step = 0
    
    def act(self, state, training=True):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor)
            
        if training:
            return action, log_prob.item(), value.item()
        else:
            return action
    
    def store_transition(self, state, action, log_prob, reward, value, done):
        """Store experience in memory"""
        self.memory.store(state, action, log_prob, reward, value, done)
    
    def compute_gae(self, rewards, values, dones, next_value=0):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[i]
                next_val = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_val * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def update(self):
        """Update policy using PPO algorithm"""
        if len(self.memory.states) < self.batch_size:
            return {}
        
        # Get batch data
        states, actions, old_log_probs, rewards, values, dones = self.memory.get_batches()
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            advantages, returns = self.compute_gae(
                rewards.cpu().numpy(),
                values.cpu().numpy(), 
                dones.cpu().numpy()
            )
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0  
        total_entropy_loss = 0
        
        for _ in range(self.ppo_epochs):
            # Get current policy outputs
            action_probs, current_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Policy loss with clipping
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss - 
                         self.entropy_coef * entropy)
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy.item()
        
        # Clear memory
        self.memory.clear()
        self.training_step += 1
        
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy': total_entropy_loss / self.ppo_epochs,
            'training_step': self.training_step
        }
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_step = checkpoint['training_step']
            print(f"Model loaded from {filepath}")
        else:
            print(f"No model found at {filepath}")
    
    def predict_action_sequence(self, state, sequence_length: int = 10) -> List[int]:
        """Predict sequence of actions for batch processing"""
        actions = []
        current_state = state.copy()
        
        with torch.no_grad():
            for _ in range(sequence_length):
                action = self.act(current_state, training=False)
                actions.append(action)
        
        return actions