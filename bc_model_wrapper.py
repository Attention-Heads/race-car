"""
Wrapper for the behavioral cloning model to provide a consistent interface
with the PPO agent for prediction.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Tuple
from preprocessing_utils import StatePreprocessor

logger = logging.getLogger(__name__)


class BehavioralCloningModel(nn.Module):
    """
    Behavioral cloning model that mimics the expert policy.
    This is a copy of the model from imitation_learning.py for standalone use.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout_rate: float = 0.2):
        super().__init__()
        
        # Use the same feature extraction architecture with increased dropout
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),  # Reduced dropout in later layers
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        
        # Action prediction head with final dropout
        self.action_head = nn.Sequential(
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.action_head(features)


class BCModelWrapper:
    """
    Wrapper class for the behavioral cloning model to provide a PPO-like interface.
    """
    
    def __init__(self, model_path: str, input_dim: int = 18, output_dim: int = 5):
        """
        Initialize the BC model wrapper.
        
        Args:
            model_path: Path to the saved model state dict
            input_dim: Input dimension (should match preprocessing output)
            output_dim: Output dimension (number of actions)
        """
        self.model_path = model_path
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load the behavioral cloning model from the saved state dict."""
        try:
            self.model = BehavioralCloningModel(self.input_dim, self.output_dim)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Behavioral cloning model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load behavioral cloning model: {e}")
            raise e
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Optional[dict]]:
        """
        Predict an action given an observation.
        
        This method provides the same interface as PPO.predict() for compatibility.
        
        Args:
            observation: Input observation array
            deterministic: Whether to use deterministic prediction (ignored for BC)
            
        Returns:
            Tuple of (action_index, info_dict)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            obs_tensor = torch.FloatTensor(observation).to(self.device)
        else:
            obs_tensor = observation
        
        # Get prediction
        with torch.no_grad():
            logits = self.model(obs_tensor)
            
            if deterministic:
                # Take the action with highest probability
                action = torch.argmax(logits, dim=-1)
            else:
                # Sample from the probability distribution
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze()
        
        # Return action as int (same as PPO interface)
        return int(action.cpu().numpy()), None
    
    def predict_proba(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for the given observation.
        
        Args:
            observation: Input observation array
            
        Returns:
            Array of action probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert observation to tensor
        if isinstance(observation, np.ndarray):
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            obs_tensor = torch.FloatTensor(observation).to(self.device)
        else:
            obs_tensor = observation
        
        # Get probabilities
        with torch.no_grad():
            logits = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
        
        return probs.cpu().numpy().squeeze()


def load_bc_model(model_path: str = "./models/best_bc_model.pth") -> BCModelWrapper:
    """
    Convenience function to load a behavioral cloning model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        BCModelWrapper instance
    """
    return BCModelWrapper(model_path)
