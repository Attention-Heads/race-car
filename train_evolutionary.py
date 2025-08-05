"""
Evolutionary Algorithm Training Script for Race Car Environment

This script implements a genetic algorithm to evolve neural network agents
that maximize distance traveled in the race car environment.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from race_car_env import make_race_car_env
import argparse
from typing import Dict, Any, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pandas as pd
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set matplotlib style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('default')
    logger.warning("Seaborn style not available, using default matplotlib style")

@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm"""
    population_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_ratio: float = 0.1  # Top % of population to keep unchanged
    tournament_size: int = 5
    evaluations_per_individual: int = 3  # Number of episodes to average performance
    max_workers: int = None  # For parallel evaluation
    
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
    
    def get_num_weights(self) -> int:
        """Get total number of weights in the network"""
        return sum(p.numel() for p in self.parameters())
    
    def clone(self) -> 'SimpleNeuralNet':
        """Create a copy of this network"""
        new_net = SimpleNeuralNet(self.input_size, 
                                  self.layer_sizes[1:-1], 
                                  self.output_size)
        new_net.set_weights(self.get_weights())
        return new_net

class Individual:
    """Represents an individual in the evolutionary population"""
    
    def __init__(self, network: SimpleNeuralNet, fitness: float = 0.0):
        self.network = network
        self.fitness = fitness
        self.raw_scores = []  # Store individual evaluation scores
        
    def evaluate(self, env_config: Dict, num_evaluations: int = 3, seed_offset: int = 0) -> float:
        """Evaluate individual's fitness over multiple episodes"""
        total_distance = 0.0
        self.raw_scores = []
        
        for i in range(num_evaluations):
            # Create environment for this evaluation
            env = make_race_car_env(env_config)
            
            # Reset with different seed each time
            obs, info = env.reset(seed=seed_offset + i)
            done = False
            episode_distance = 0.0
            
            while not done:
                # Convert observation to tensor
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                
                # Get action probabilities from network
                with torch.no_grad():
                    action_probs = self.network(obs_tensor)
                
                # Sample action from probabilities
                action = torch.multinomial(action_probs, 1).item()
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Extract distance from info if available
                if 'distance' in info:
                    episode_distance = info['distance']
            
            self.raw_scores.append(episode_distance)
            total_distance += episode_distance
            env.close()
        
        # Average distance across evaluations
        self.fitness = total_distance / num_evaluations
        return self.fitness
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """Apply random mutations to the network weights"""
        weights = self.network.get_weights()
        
        # Create mutation mask
        mutation_mask = np.random.random(weights.shape) < mutation_rate
        
        # Apply mutations
        mutations = np.random.normal(0, mutation_strength, weights.shape)
        weights[mutation_mask] += mutations[mutation_mask]
        
        self.network.set_weights(weights)
    
    def crossover(self, other: 'Individual', crossover_rate: float = 0.7) -> Tuple['Individual', 'Individual']:
        """Create two offspring through crossover with another individual"""
        if np.random.random() > crossover_rate:
            # No crossover, return copies of parents
            return Individual(self.network.clone()), Individual(other.network.clone())
        
        # Get parent weights
        weights1 = self.network.get_weights()
        weights2 = other.network.get_weights()
        
        # Single-point crossover
        crossover_point = np.random.randint(1, len(weights1))
        
        # Create offspring weights
        child1_weights = np.concatenate([weights1[:crossover_point], weights2[crossover_point:]])
        child2_weights = np.concatenate([weights2[:crossover_point], weights1[crossover_point:]])
        
        # Create offspring networks
        child1_net = self.network.clone()
        child2_net = other.network.clone()
        
        child1_net.set_weights(child1_weights)
        child2_net.set_weights(child2_weights)
        
        return Individual(child1_net), Individual(child2_net)

def evaluate_individual_parallel(args: Tuple[Individual, Dict, int, int, int]) -> Tuple[int, float, List[float]]:
    """Wrapper function for parallel evaluation of individuals"""
    individual, env_config, num_evaluations, seed_offset, individual_id = args
    
    try:
        fitness = individual.evaluate(env_config, num_evaluations, seed_offset)
        return individual_id, fitness, individual.raw_scores
    except Exception as e:
        logger.error(f"Error evaluating individual {individual_id}: {e}")
        return individual_id, 0.0, [0.0] * num_evaluations

class EvolutionaryTrainer:
    """Main evolutionary training class"""
    
    def __init__(self, config: EvolutionConfig, env_config: Dict, save_dir: str = "evolutionary_results"):
        self.config = config
        self.env_config = env_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize environment to get observation space
        temp_env = make_race_car_env(env_config)
        temp_obs, _ = temp_env.reset()
        self.obs_size = len(temp_obs)
        temp_env.close()
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Training history
        self.history = {
            'generation': [],
            'best_fitness': [],
            'average_fitness': [],
            'worst_fitness': [],
            'fitness_std': []
        }
        
        # Set number of workers for parallel evaluation
        if self.config.max_workers is None:
            self.config.max_workers = min(mp.cpu_count(), self.config.population_size)
        
        logger.info(f"Initialized evolutionary trainer with {self.config.population_size} individuals")
        logger.info(f"Network architecture: {self.obs_size} -> [64, 32] -> 5")
        logger.info(f"Using {self.config.max_workers} workers for parallel evaluation")
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.config.population_size):
            # Create random network
            network = SimpleNeuralNet(self.obs_size)
            
            # Initialize with random weights (Xavier/Glorot initialization is already done by PyTorch)
            individual = Individual(network)
            population.append(individual)
        
        return population
    
    def _tournament_selection(self, population: List[Individual], tournament_size: int) -> Individual:
        """Select individual using tournament selection"""
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def _evaluate_population(self, generation: int):
        """Evaluate fitness of entire population in parallel"""
        logger.info(f"Evaluating population for generation {generation}...")
        
        # Prepare arguments for parallel evaluation
        eval_args = []
        for i, individual in enumerate(self.population):
            seed_offset = generation * 1000 + i * 100  # Ensure different seeds
            eval_args.append((individual, self.env_config, self.config.evaluations_per_individual, seed_offset, i))
        
        # Evaluate in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_id = {executor.submit(evaluate_individual_parallel, args): args[4] for args in eval_args}
            
            for future in as_completed(future_to_id):
                individual_id, fitness, raw_scores = future.result()
                self.population[individual_id].fitness = fitness
                self.population[individual_id].raw_scores = raw_scores
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update history
        fitnesses = [ind.fitness for ind in self.population]
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(max(fitnesses))
        self.history['average_fitness'].append(np.mean(fitnesses))
        self.history['worst_fitness'].append(min(fitnesses))
        self.history['fitness_std'].append(np.std(fitnesses))
        
        logger.info(f"Generation {generation}: Best={max(fitnesses):.2f}, "
                   f"Avg={np.mean(fitnesses):.2f}, Worst={min(fitnesses):.2f}")
    
    def _create_next_generation(self) -> List[Individual]:
        """Create next generation using selection, crossover, and mutation"""
        new_population = []
        
        # Keep elite individuals
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        elite_individuals = [Individual(ind.network.clone(), ind.fitness) for ind in self.population[:elite_count]]
        new_population.extend(elite_individuals)
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Select parents
            parent1 = self._tournament_selection(self.population, self.config.tournament_size)
            parent2 = self._tournament_selection(self.population, self.config.tournament_size)
            
            # Create offspring
            child1, child2 = parent1.crossover(parent2, self.config.crossover_rate)
            
            # Apply mutations
            child1.mutate(self.config.mutation_rate)
            child2.mutate(self.config.mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.config.population_size]
    
    def train(self):
        """Main training loop"""
        logger.info("Starting evolutionary training...")
        
        try:
            for generation in range(self.config.generations):
                # Evaluate current population
                self._evaluate_population(generation)
                
                # Save best individual and plot progress every 10 generations
                if generation % 10 == 0:
                    self._save_best_individual(generation)
                    self._plot_progress()
                
                # Check for early stopping (optional)
                if generation > 50 and self.history['best_fitness'][-1] > 1000:  # Adjust threshold as needed
                    logger.info(f"Early stopping at generation {generation} due to high fitness")
                    break
                
                # Create next generation (skip for last generation)
                if generation < self.config.generations - 1:
                    self.population = self._create_next_generation()
            
            # Final evaluation and save
            logger.info("Training completed!")
            self._save_best_individual(self.config.generations - 1, final=True)
            self._plot_progress(final=True)
            self._save_training_history()
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self._save_best_individual(len(self.history['generation']) - 1, final=True)
            self._plot_progress(final=True)
    
    def _save_best_individual(self, generation: int, final: bool = False):
        """Save the best individual from current generation"""
        best_individual = self.population[0]  # Population is sorted by fitness
        
        suffix = "_final" if final else f"_gen_{generation}"
        
        # Save network weights
        weights_path = self.save_dir / f"best_individual{suffix}.pkl"
        with open(weights_path, 'wb') as f:
            pickle.dump({
                'weights': best_individual.network.get_weights(),
                'fitness': best_individual.fitness,
                'raw_scores': best_individual.raw_scores,
                'generation': generation,
                'network_architecture': best_individual.network.layer_sizes
            }, f)
        
        # Save network model
        model_path = self.save_dir / f"best_model{suffix}.pt"
        torch.save(best_individual.network.state_dict(), model_path)
        
        logger.info(f"Saved best individual (fitness: {best_individual.fitness:.2f}) to {weights_path}")
    
    def _plot_progress(self, final: bool = False):
        """Plot training progress"""
        if len(self.history['generation']) < 2:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot fitness evolution
        plt.subplot(2, 2, 1)
        plt.plot(self.history['generation'], self.history['best_fitness'], 'b-', label='Best', linewidth=2)
        plt.plot(self.history['generation'], self.history['average_fitness'], 'g-', label='Average', linewidth=2)
        plt.plot(self.history['generation'], self.history['worst_fitness'], 'r-', label='Worst', linewidth=2)
        plt.fill_between(self.history['generation'], 
                        np.array(self.history['average_fitness']) - np.array(self.history['fitness_std']),
                        np.array(self.history['average_fitness']) + np.array(self.history['fitness_std']),
                        alpha=0.3, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Distance (Fitness)')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot fitness distribution for latest generation
        plt.subplot(2, 2, 2)
        recent_fitnesses = [ind.fitness for ind in self.population]
        plt.hist(recent_fitnesses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Distance (Fitness)')
        plt.ylabel('Count')
        plt.title(f'Fitness Distribution (Gen {self.history["generation"][-1]})')
        plt.grid(True, alpha=0.3)
        
        # Plot fitness improvement rate
        plt.subplot(2, 2, 3)
        if len(self.history['best_fitness']) > 1:
            improvements = np.diff(self.history['best_fitness'])
            plt.plot(self.history['generation'][1:], improvements, 'purple', linewidth=2)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.xlabel('Generation')
            plt.ylabel('Fitness Improvement')
            plt.title('Generation-to-Generation Improvement')
            plt.grid(True, alpha=0.3)
        
        # Plot diversity (fitness standard deviation)
        plt.subplot(2, 2, 4)
        plt.plot(self.history['generation'], self.history['fitness_std'], 'orange', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Standard Deviation')
        plt.title('Population Diversity')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        suffix = "_final" if final else f"_gen_{self.history['generation'][-1]}"
        plot_path = self.save_dir / f"training_progress{suffix}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training progress plot to {plot_path}")
    
    def _save_training_history(self):
        """Save complete training history"""
        history_path = self.save_dir / "training_history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in self.history.items():
            if isinstance(value, np.ndarray):
                history_json[key] = value.tolist()
            else:
                history_json[key] = value
        
        with open(history_path, 'w') as f:
            json.dump({
                'config': {
                    'population_size': self.config.population_size,
                    'generations': self.config.generations,
                    'mutation_rate': self.config.mutation_rate,
                    'crossover_rate': self.config.crossover_rate,
                    'elite_ratio': self.config.elite_ratio,
                    'tournament_size': self.config.tournament_size,
                    'evaluations_per_individual': self.config.evaluations_per_individual
                },
                'history': history_json,
                'env_config': self.env_config
            }, f, indent=2)
        
        logger.info(f"Saved training history to {history_path}")

def load_evolved_agent(weights_path: str, obs_size: int) -> SimpleNeuralNet:
    """Load a trained evolutionary agent"""
    with open(weights_path, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct network
    if 'network_architecture' in data:
        layer_sizes = data['network_architecture']
        network = SimpleNeuralNet(layer_sizes[0], layer_sizes[1:-1], layer_sizes[-1])
    else:
        # Fallback to default architecture
        network = SimpleNeuralNet(obs_size)
    
    network.set_weights(data['weights'])
    return network

def test_evolved_agent(weights_path: str, env_config: Dict, num_episodes: int = 10):
    """Test a saved evolutionary agent"""
    logger.info(f"Testing evolved agent from {weights_path}")
    
    # Create environment
    env = make_race_car_env(env_config)
    obs, _ = env.reset()
    
    # Load agent
    network = load_evolved_agent(weights_path, len(obs))
    network.eval()
    
    episode_distances = []
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        done = False
        episode_distance = 0.0
        
        while not done:
            # Get action from network
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = network(obs_tensor)
                action = torch.multinomial(action_probs, 1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if 'distance' in info:
                episode_distance = info['distance']
        
        episode_distances.append(episode_distance)
        logger.info(f"Episode {episode + 1}: Distance = {episode_distance:.2f}")
    
    env.close()
    
    avg_distance = np.mean(episode_distances)
    std_distance = np.std(episode_distances)
    
    logger.info(f"Test Results: Average Distance = {avg_distance:.2f} ± {std_distance:.2f}")
    logger.info(f"Best Distance = {max(episode_distances):.2f}")
    logger.info(f"Worst Distance = {min(episode_distances):.2f}")
    
    return episode_distances

def main():
    parser = argparse.ArgumentParser(description="Train race car agent using evolutionary algorithm")
    parser.add_argument("--population-size", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.7, help="Crossover rate")
    parser.add_argument("--elite-ratio", type=float, default=0.1, help="Elite ratio to keep unchanged")
    parser.add_argument("--tournament-size", type=int, default=5, help="Tournament selection size")
    parser.add_argument("--evaluations", type=int, default=3, help="Episodes per individual evaluation")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--save-dir", type=str, default="evolutionary_results", help="Directory to save results")
    parser.add_argument("--test", type=str, default=None, help="Path to weights file for testing")
    parser.add_argument("--test-episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    
    args = parser.parse_args()
    
    # Environment configuration
    env_config = {
        'render': args.render,
        'reward_config': {
            'distance_weight': 1.0,
            'survival_weight': 0.1,
            'crash_penalty': -10.0,
            'action_penalty': -0.01
        }
    }
    
    if args.test:
        # Test mode
        test_evolved_agent(args.test, env_config, args.test_episodes)
    else:
        # Training mode
        config = EvolutionConfig(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            elite_ratio=args.elite_ratio,
            tournament_size=args.tournament_size,
            evaluations_per_individual=args.evaluations,
            max_workers=args.workers
        )
        
        trainer = EvolutionaryTrainer(config, env_config, args.save_dir)
        trainer.train()

if __name__ == "__main__":
    main()
