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
    Includes regularization techniques to prevent overfitting.
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
        
    def load_and_preprocess_data(self, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load expert data and preprocess it for training.
        
        Args:
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
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
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        # Adjust validation size relative to remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        logger.info(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        logger.info(f"Action distribution in training set:")
        unique, counts = np.unique(y_train, return_counts=True)
        for action_idx, count in zip(unique, counts):
            action_name = list(self.action_mapping.keys())[action_idx]
            logger.info(f"  {action_name}: {count} ({count/len(y_train)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_behavioral_cloning(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                epochs: int = 200,
                                batch_size: int = 256,
                                learning_rate: float = 1e-3,
                                patience: int = 20,
                                weight_decay: float = 1e-4,
                                dropout_rate: float = 0.3) -> BehavioralCloningModel:
        """
        Train a behavioral cloning model on expert data.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            patience: Early stopping patience
            weight_decay: L2 regularization strength
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Trained model
        """
        logger.info("Training behavioral cloning model...")
        
        # Create datasets and dataloaders
        train_dataset = ExpertDataset(X_train, y_train)
        val_dataset = ExpertDataset(X_val, y_val)
        test_dataset = ExpertDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model with configurable dropout
        input_dim = self.preprocessor.get_input_dim()
        output_dim = len(self.action_mapping)
        self.model = BehavioralCloningModel(input_dim, output_dim, dropout_rate=dropout_rate)
        
        # Loss function and optimizer with weight decay for regularization
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6)
        
        # Training loop with validation-based early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
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
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for states, actions in val_loader:
                    outputs = self.model(states)
                    loss = criterion(outputs, actions)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += actions.size(0)
                    val_correct += (predicted == actions).sum().item()
            
            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            
            # Test evaluation (only for monitoring, not for early stopping)
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for states, actions in test_loader:
                    outputs = self.model(states)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += actions.size(0)
                    test_correct += (predicted == actions).sum().item()
            
            test_acc = test_correct / test_total
            test_accuracies.append(test_acc)
            
            # Learning rate scheduling based on validation loss
            scheduler.step(avg_val_loss)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                           f"Test Acc: {test_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping based on validation loss to prevent overfitting
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.model_save_dir / "best_bc_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch} - no improvement in validation loss")
                    break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Final test accuracy: {test_accuracies[-1]:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.model_save_dir / "best_bc_model.pth"))
        
        # Final evaluation on test set with best model
        final_test_metrics = self._evaluate_on_dataset(test_loader, criterion)
        logger.info(f"Final test evaluation - Loss: {final_test_metrics['loss']:.4f}, "
                   f"Accuracy: {final_test_metrics['accuracy']:.4f}")
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses, val_accuracies, test_accuracies)
        
        # Save training history
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'test_accuracies': test_accuracies,
            'final_test_metrics': final_test_metrics
        }
        torch.save(training_history, self.model_save_dir / "training_history.pth")
        
        return self.model
    
    def _evaluate_on_dataset(self, data_loader, criterion):
        """Evaluate model on a dataset and return metrics."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for states, actions in data_loader:
                outputs = self.model(states)
                loss = criterion(outputs, actions)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += actions.size(0)
                correct += (predicted == actions).sum().item()
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': correct / total
        }
    
    def _plot_training_curves(self, train_losses: list, val_losses: list, val_accuracies: list, test_accuracies: list):
        """Plot training curves including validation metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Validation accuracy
        ax2.plot(val_accuracies, label='Validation Accuracy', color='orange')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Test accuracy
        ax3.plot(test_accuracies, label='Test Accuracy', color='green')
        ax3.set_title('Test Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True)
        
        # Combined accuracy comparison
        ax4.plot(val_accuracies, label='Validation Accuracy', color='orange')
        ax4.plot(test_accuracies, label='Test Accuracy', color='green')
        ax4.set_title('Validation vs Test Accuracy')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
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
    X_train, X_val, X_test, y_train, y_val, y_test = learner.load_and_preprocess_data()
    
    # Train behavioral cloning model
    bc_model = learner.train_behavioral_cloning(X_train, y_train, X_val, y_val, X_test, y_test)
    
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
