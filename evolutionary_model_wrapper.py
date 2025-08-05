"""
Wrapper for evolutionary neural network models to provide a unified interface
compatible with the existing API structure.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class SimpleNeuralNet(nn.Module):
    """Simple feedforward neural network for the race car agent"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], output_size: int = 5):
        super(SimpleNeuralNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=-1))  # For action probabilities
        
        self.network = nn.Sequential(*layers)
        
        # Store network structure for genetic operations
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        
    def forward(self, x):
        return self.network(x)
    
    def get_weights(self) -> np.ndarray:
        """Extract all network weights as a flat numpy array"""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights(self, weights: np.ndarray):
        """Set network weights from a flat numpy array"""
        start_idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_shape = param.shape
            
            param_weights = weights[start_idx:start_idx + param_size].reshape(param_shape)
            param.data = torch.tensor(param_weights, dtype=param.dtype)
            start_idx += param_size

class EvolutionaryModelWrapper:
    """Wrapper class to make evolutionary models compatible with PPO interface"""
    
    def __init__(self, model_path: str = None, network: SimpleNeuralNet = None):
        """
        Initialize the wrapper with either a model path or a network directly
        
        Args:
            model_path: Path to the saved evolutionary model (.pkl or .pt file)
            network: Pre-loaded SimpleNeuralNet instance
        """
        self.network = network
        self.model_path = model_path
        
        if model_path and network is None:
            self.load_model(model_path)
        elif network is None:
            raise ValueError("Either model_path or network must be provided")
    
    def load_model(self, model_path: str):
        """Load an evolutionary model from file"""
        try:
            if model_path.endswith('.pkl'):
                # Load from pickle file (contains weights and metadata)
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract network architecture if available
                if 'network_architecture' in data:
                    layer_sizes = data['network_architecture']
                    input_size = layer_sizes[0]
                    hidden_sizes = layer_sizes[1:-1]
                    output_size = layer_sizes[-1]
                else:
                    # Default architecture (assuming standard setup)
                    input_size = 10  # 2 velocity + 8 sensors
                    hidden_sizes = [64, 32]
                    output_size = 5
                
                # Create network and load weights
                self.network = SimpleNeuralNet(input_size, hidden_sizes, output_size)
                self.network.set_weights(data['weights'])
                
                logger.info(f"Loaded evolutionary model from {model_path}")
                logger.info(f"Model fitness: {data.get('fitness', 'unknown')}")
                logger.info(f"Model generation: {data.get('generation', 'unknown')}")
                
            elif model_path.endswith('.pt'):
                # Load from PyTorch state dict
                # We need to know the architecture to create the network first
                # This is a limitation - we'll need the architecture info
                raise NotImplementedError("Loading from .pt files requires knowing the network architecture. "
                                        "Please use .pkl files which contain architecture information.")
            else:
                raise ValueError(f"Unsupported file format: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load evolutionary model from {model_path}: {e}")
            raise
    
    def predict(self, observation: Union[np.ndarray, torch.Tensor], deterministic: bool = True) -> Tuple[int, None]:
        """
        Predict action given observation
        
        Args:
            observation: Input observation (numpy array or torch tensor)
            deterministic: If True, returns the most likely action. If False, samples from distribution.
        
        Returns:
            Tuple of (action_index, None) to match PPO interface
        """
        try:
            # Ensure observation is a torch tensor
            if isinstance(observation, np.ndarray):
                obs_tensor = torch.tensor(observation, dtype=torch.float32)
            else:
                obs_tensor = observation.float()
            
            # Add batch dimension if needed
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Get action probabilities from network
            with torch.no_grad():
                action_probs = self.network(obs_tensor)
            
            if deterministic:
                # Return the action with highest probability
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from the probability distribution
                action = torch.multinomial(action_probs, 1).item()
            
            return action, None
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Return a safe default action (do nothing)
            return 0, None

def load_evolutionary_model(model_path: str) -> EvolutionaryModelWrapper:
    """
    Convenience function to load an evolutionary model
    
    Args:
        model_path: Path to the saved evolutionary model
    
    Returns:
        EvolutionaryModelWrapper instance
    """
    return EvolutionaryModelWrapper(model_path=model_path)

def get_best_evolutionary_model(results_dir: str = "./evolutionary_results") -> str:
    """
    Find the best evolutionary model in the results directory
    
    Args:
        results_dir: Directory containing evolutionary results
    
    Returns:
        Path to the best model file
    """
    import os
    from pathlib import Path
    
    results_path = Path(results_dir)
    
    # Look for final model first
    final_model = results_path / "best_individual_final.pkl"
    if final_model.exists():
        return str(final_model)
    
    # If no final model, find the highest generation model
    pkl_files = list(results_path.glob("best_individual_gen_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No evolutionary models found in {results_dir}")
    
    # Extract generation numbers and find the highest
    generations = []
    for file in pkl_files:
        try:
            gen_str = file.stem.split("_gen_")[1]
            generations.append((int(gen_str), str(file)))
        except (IndexError, ValueError):
            continue
    
    if not generations:
        raise FileNotFoundError(f"No valid evolutionary models found in {results_dir}")
    
    # Return the model from the highest generation
    generations.sort(key=lambda x: x[0], reverse=True)
    best_model_path = generations[0][1]
    
    logger.info(f"Found best evolutionary model: {best_model_path} (generation {generations[0][0]})")
    return best_model_path
