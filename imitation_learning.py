"""
Imitation Learning Implementation for Race Car Environment

This module implements behavioral cloning and other imitation learning techniques
to create an initial policy from expert demonstrations that can be fine-tuned with PPO.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from race_car_env import make_race_car_env
import joblib
import logging
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
from preprocessing_utils import StatePreprocessor, FEATURE_ORDER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceCarFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the race car environment.
    This will be used by both the imitation learning model and PPO.
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Input size is the observation space dimension (velocity + sensors)
        input_dim = observation_space.shape[0]
        
        # Neural network architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ExpertDataset(Dataset):
    """Dataset class for expert demonstrations."""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class BehavioralCloningModel(nn.Module):
    """
    Behavioral cloning model that mimics the expert policy.
    Uses the same architecture as the PPO feature extractor for consistency.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Use the same feature extraction architecture
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        
        # Action prediction head
        self.action_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.action_head(features)


class ImitationLearner:
    """
    Main class for imitation learning from expert demonstrations.
    """
    
    def __init__(self, 
                 expert_data_path: str = "processed_balanced_training_data.csv",
                 model_save_dir: str = "models",
                 use_velocity_scaler: bool = True):
        """
        Initialize the imitation learner.
        
        Args:
            expert_data_path: Path to expert training data CSV
            model_save_dir: Directory to save trained models
            use_velocity_scaler: Whether to use velocity scaling
        """
        self.expert_data_path = expert_data_path
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(exist_ok=True)
        self.use_velocity_scaler = use_velocity_scaler
        
        # Action mapping (must match environment)
        self.action_mapping = {
            'NOTHING': 0, 
            'ACCELERATE': 1, 
            'DECELERATE': 2, 
            'STEER_LEFT': 3, 
            'STEER_RIGHT': 4
        }
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.preprocessor = StatePreprocessor(use_velocity_scaler=use_velocity_scaler)
        self.model = None
        self.env = None
        
    def load_and_preprocess_data(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load expert data and preprocess it for training.
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading expert data from {self.expert_data_path}")
        
        # Load data
        df = pd.read_csv(self.expert_data_path)
        logger.info(f"Loaded {len(df)} expert demonstrations")
        
        # Remove crashed episodes (optional - you might want to keep some)
        # df_clean = df[df['did_crash'] == False].copy()
        # logger.info(f"After removing crashes: {len(df_clean)} demonstrations")
        
        # For now, keep all data including crashes for more diverse training
        df_clean = df.copy()
        
        # Use centralized preprocessing
        X = self.preprocessor.preprocess_batch(df_clean)
        
        # Encode actions
        actions = df_clean['action'].values
        y = np.array([self.action_mapping[action] for action in actions])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Action distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for action_idx, count in zip(unique, counts):
            action_name = list(self.action_mapping.keys())[action_idx]
            logger.info(f"  {action_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_behavioral_cloning(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                epochs: int = 1000,
                                batch_size: int = 512,
                                learning_rate: float = 1e-3,
                                patience: int = 10) -> BehavioralCloningModel:
        """
        Train a behavioral cloning model on expert data.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            patience: Early stopping patience
            
        Returns:
            Trained model
        """
        logger.info("Training behavioral cloning model...")
        
        # Create datasets and dataloaders
        train_dataset = ExpertDataset(X_train, y_train)
        test_dataset = ExpertDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_dim = self.preprocessor.get_input_dim()
        output_dim = len(self.action_mapping)
        self.model = BehavioralCloningModel(input_dim, output_dim)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_test_acc = 0.0
        patience_counter = 0
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for states, actions in train_loader:
                optimizer.zero_grad()
                outputs = self.model(states)
                loss = criterion(outputs, actions)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Evaluation phase
            self.model.eval()
            correct = 0
            total = 0
            test_loss = 0.0
            
            with torch.no_grad():
                for states, actions in test_loader:
                    outputs = self.model(states)
                    loss = criterion(outputs, actions)
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += actions.size(0)
                    correct += (predicted == actions).sum().item()
            
            test_acc = correct / total
            avg_test_loss = test_loss / len(test_loader)
            test_accuracies.append(test_acc)
            
            # Learning rate scheduling
            scheduler.step(avg_test_loss)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                           f"Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.model_save_dir / "best_bc_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        logger.info(f"Training completed. Best test accuracy: {best_test_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.model_save_dir / "best_bc_model.pth"))
        
        # Plot training curves
        self._plot_training_curves(train_losses, test_accuracies)
        
        return self.model
    
    def _plot_training_curves(self, train_losses: list, test_accuracies: list):
        """Plot training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        ax2.plot(test_accuracies)
        ax2.set_title('Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / "training_curves.png")
        plt.close()
        logger.info(f"Training curves saved to {self.model_save_dir / 'training_curves.png'}")
    
    def create_ppo_policy(self, env_config: Dict[str, Any] = None) -> PPO:
        """
        Create a PPO agent initialized with the behavioral cloning weights.
        
        Args:
            env_config: Environment configuration
            
        Returns:
            PPO agent with pretrained weights
        """
        logger.info("Creating PPO policy with behavioral cloning initialization...")
        
        # Create environment
        env_config = env_config or {}
        env = make_race_car_env(env_config)
        
        # Custom policy class that uses our feature extractor with matching network sizes
        class CustomActorCriticPolicy(ActorCriticPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs,
                               features_extractor_class=RaceCarFeaturesExtractor,
                               features_extractor_kwargs=dict(features_dim=256),
                               net_arch=[256])  # Match the feature extractor output size
        
        # Create PPO agent
        ppo_agent = PPO(
            CustomActorCriticPolicy,
            env,
            verbose=1,
            device='cpu',
            tensorboard_log="./tensorboard_logs/"
        )
        
        # Initialize policy network with behavioral cloning weights
        if self.model is not None:
            self._transfer_weights_to_ppo(ppo_agent)
        
        return ppo_agent
    
    def _transfer_weights_to_ppo(self, ppo_agent: PPO):
        """
        Transfer behavioral cloning weights to PPO policy network.
        
        Args:
            ppo_agent: PPO agent to initialize
        """
        logger.info("Transferring behavioral cloning weights to PPO policy...")
        
        try:
            # Get the feature extractor from PPO policy
            ppo_features_extractor = ppo_agent.policy.features_extractor.net
            bc_features_extractor = self.model.feature_extractor
            
            # Transfer feature extractor weights
            with torch.no_grad():
                for ppo_layer, bc_layer in zip(ppo_features_extractor, bc_features_extractor):
                    if isinstance(ppo_layer, nn.Linear) and isinstance(bc_layer, nn.Linear):
                        ppo_layer.weight.copy_(bc_layer.weight)
                        ppo_layer.bias.copy_(bc_layer.bias)
            
            # Initialize action head with behavioral cloning weights
            # Note: PPO has both action and value heads, we only initialize action head
            action_net = ppo_agent.policy.action_net
            bc_action_head = self.model.action_head
            
            with torch.no_grad():
                action_net.weight.copy_(bc_action_head.weight)
                action_net.bias.copy_(bc_action_head.bias)
            
            logger.info("Successfully transferred weights from behavioral cloning model to PPO")
            
        except Exception as e:
            logger.warning(f"Failed to transfer weights: {e}")
            logger.warning("PPO will start with random initialization")
    
    def evaluate_model(self, model, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate a model in the environment.
        
        Args:
            model: Model to evaluate (can be BC model or PPO agent)
            env: Environment to evaluate in
            num_episodes: Number of episodes to run
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        crash_count = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                if hasattr(model, 'predict'):  # PPO agent
                    action, _ = model.predict(obs, deterministic=True)
                else:  # BC model
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action_probs = model(obs_tensor)
                        action = torch.argmax(action_probs, dim=1).item()
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if info.get('crashed', False):
                    crash_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'crash_rate': crash_count / num_episodes,
            'max_reward': np.max(episode_rewards)
        }
        
        logger.info(f"Evaluation results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.3f}")
        
        return metrics
    
    def save_model(self, model_name: str = "imitation_model"):
        """Save the trained behavioral cloning model."""
        if self.model is not None:
            model_path = self.model_save_dir / f"{model_name}.pth"
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str, input_dim: int, output_dim: int):
        """Load a saved behavioral cloning model."""
        self.model = BehavioralCloningModel(input_dim, output_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")


def main():
    """
    Main function to demonstrate imitation learning workflow.
    """
    # Initialize imitation learner
    learner = ImitationLearner()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = learner.load_and_preprocess_data()
    
    # Train behavioral cloning model
    bc_model = learner.train_behavioral_cloning(X_train, y_train, X_test, y_test)
    
    # Save the model
    learner.save_model("behavioral_cloning_model")
    
    # Create environment for evaluation
    env = make_race_car_env({'render': False})
    
    # Evaluate behavioral cloning model
    bc_metrics = learner.evaluate_model(bc_model, env)
    
    # Create PPO agent with behavioral cloning initialization
    ppo_agent = learner.create_ppo_policy()
    
    # Save the initialized PPO model
    ppo_agent.save(learner.model_save_dir / "ppo_initialized_with_bc")
    
    logger.info("Imitation learning pipeline completed successfully!")
    logger.info("You can now fine-tune the PPO agent using train_ppo.py with the saved model.")


if __name__ == "__main__":
    main()
