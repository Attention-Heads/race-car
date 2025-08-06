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
    population_size: int = 128
    generations: int = 200
    mutation_rate: float = 0.05  # Reduced to preserve good traits
    crossover_rate: float = 0.8  # Increased for more mixing of good genes
    elite_ratio: float = 0.2  # Increased to keep more top performers
    tournament_size: int = 15  # Increased for stronger selection pressure
    evaluations_per_individual: int = 5  # Number of episodes to average performance
    max_workers: int = None  # For parallel evaluation
    fitness_sharing: bool = True  # Enable fitness sharing to maintain diversity
    selection_pressure: float = 2.0  # For rank-based selection
    min_population_diversity: float = 0.1  # Minimum fitness std to maintain
    
class SimpleNeuralNet(nn.Module):
    """Simple feedforward neural network for the race car agent"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64], output_size: int = 5):
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
        self.scaled_fitness = fitness  # For fitness scaling
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
        logger.info(f"Network architecture: {self.obs_size} -> [128, 64] -> 5")
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
    
    def _rank_based_selection(self, population: List[Individual]) -> Individual:
        """Select individual using rank-based selection with exponential bias toward top performers"""
        # Population should already be sorted by fitness (descending)
        n = len(population)
        
        # Create exponential probabilities favoring top individuals
        ranks = np.arange(n, 0, -1)  # Higher rank for better fitness
        probabilities = np.power(ranks, self.config.selection_pressure)
        probabilities = probabilities / np.sum(probabilities)
        
        # Select based on probabilities
        selected_idx = np.random.choice(n, p=probabilities)
        return population[selected_idx]
    
    def _fitness_proportionate_selection(self, population: List[Individual]) -> Individual:
        """Select individual using fitness proportionate selection (roulette wheel)"""
        fitnesses = np.array([ind.fitness for ind in population])
        
        # Handle negative fitnesses by shifting
        min_fitness = np.min(fitnesses)
        if min_fitness < 0:
            fitnesses = fitnesses - min_fitness + 1
        
        # Avoid division by zero
        if np.sum(fitnesses) == 0:
            return np.random.choice(population)
        
        probabilities = fitnesses / np.sum(fitnesses)
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx]
    
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
        
        # Sort population by raw fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Get RAW fitnesses for history and logging
        raw_fitnesses = [ind.fitness for ind in self.population]
        
        # Apply fitness scaling for selection purposes (this modifies ind.scaled_fitness)
        if hasattr(self.config, 'fitness_sharing') and self.config.fitness_sharing:
            self._apply_fitness_scaling()
        
        # Update history using the RAW fitnesses
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(np.max(raw_fitnesses))
        self.history['average_fitness'].append(np.mean(raw_fitnesses))
        self.history['worst_fitness'].append(np.min(raw_fitnesses))
        self.history['fitness_std'].append(np.std(raw_fitnesses))
        
        # Log using the RAW fitnesses
        logger.info(f"Generation {generation}: Best={np.max(raw_fitnesses):.2f}, "
                f"Avg={np.mean(raw_fitnesses):.2f}, Worst={np.min(raw_fitnesses):.2f}, "
                f"Std={np.std(raw_fitnesses):.2f}")
    
    def _apply_fitness_scaling(self):
        """Apply fitness scaling to reduce the impact of outliers"""
        fitnesses = np.array([ind.fitness for ind in self.population])
        
        # Sigma scaling - reduces the impact of fitness outliers
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        if std_fitness > 0:
            # Sigma scaling: f'(i) = max(f(i) - (mean - c*std), 0)
            c = 2.0  # Scaling factor
            scaled_fitnesses = np.maximum(fitnesses - (mean_fitness - c * std_fitness), 0.1)
        else:
            scaled_fitnesses = fitnesses
        
        # Linear ranking: assign fitness based on rank
        sorted_indices = np.argsort(fitnesses)[::-1]  # Descending order
        n = len(self.population)
        
        for i, idx in enumerate(sorted_indices):
            rank = i + 1
            # Linear ranking: best gets 2.0, worst gets 0.0
            rank_fitness = 2.0 - 2.0 * (rank - 1) / (n - 1) if n > 1 else 1.0
            self.population[idx].scaled_fitness = rank_fitness
    
    def _create_next_generation(self) -> List[Individual]:
        """Create next generation using selection, crossover, and mutation"""
        new_population = []
        
        # Keep elite individuals (top performers)
        elite_count = int(self.config.population_size * self.config.elite_ratio)
        elite_individuals = [Individual(ind.network.clone(), ind.fitness) for ind in self.population[:elite_count]]
        new_population.extend(elite_individuals)
        
        # Check population diversity and adjust selection strategy
        fitnesses = [ind.fitness for ind in self.population]
        fitness_std = np.std(fitnesses)
        fitness_mean = np.mean(fitnesses)
        diversity_ratio = fitness_std / (abs(fitness_mean) + 1e-8)
        
        logger.info(f"Population diversity ratio: {diversity_ratio:.4f}")
        
        # Generate rest of population through crossover and mutation
        while len(new_population) < self.config.population_size:
            # Use different selection strategies based on diversity
            if diversity_ratio < self.config.min_population_diversity:
                # Low diversity - use tournament selection to maintain some variety
                parent1 = self._tournament_selection(self.population, max(3, self.config.tournament_size // 3))
                parent2 = self._tournament_selection(self.population, max(3, self.config.tournament_size // 3))
            elif diversity_ratio > 0.3:
                # High diversity - use strong selection pressure
                parent1 = self._rank_based_selection(self.population)
                parent2 = self._rank_based_selection(self.population)
            else:
                # Medium diversity - mix of selection strategies
                if np.random.random() < 0.7:
                    parent1 = self._tournament_selection(self.population, self.config.tournament_size)
                    parent2 = self._tournament_selection(self.population, self.config.tournament_size)
                else:
                    parent1 = self._rank_based_selection(self.population)
                    parent2 = self._rank_based_selection(self.population)
            
            # Ensure parents are different
            if parent1 == parent2 and len(self.population) > 1:
                parent2 = self._tournament_selection(self.population, self.config.tournament_size)
            
            # Create offspring
            child1, child2 = parent1.crossover(parent2, self.config.crossover_rate)
            
            # Apply adaptive mutation based on population diversity
            mutation_rate = self.config.mutation_rate
            if diversity_ratio < self.config.min_population_diversity:
                mutation_rate *= 2.0  # Increase mutation for low diversity
            elif diversity_ratio > 0.4:
                mutation_rate *= 0.5  # Decrease mutation for high diversity
            
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        trimmed_population = new_population[:self.config.population_size]
        
        # Optional: Apply fitness-based culling if we have too many poor performers
        if len(trimmed_population) == self.config.population_size:
            # Calculate fitness threshold (remove bottom 10% if fitness variance is high)
            sorted_by_fitness = sorted(trimmed_population, key=lambda x: x.fitness, reverse=True)
            if diversity_ratio > 0.5:  # High variance indicates many poor performers
                keep_count = int(0.9 * len(sorted_by_fitness))
                trimmed_population = sorted_by_fitness[:keep_count]
                
                # Fill remaining spots with mutated copies of top performers
                while len(trimmed_population) < self.config.population_size:
                    # Pick from top 20%
                    top_20_percent = int(0.2 * len(sorted_by_fitness))
                    parent = sorted_by_fitness[np.random.randint(0, max(1, top_20_percent))]
                    child = Individual(parent.network.clone())
                    child.mutate(self.config.mutation_rate * 1.5)  # Higher mutation for diversity
                    trimmed_population.append(child)
        
        return trimmed_population
    
    def train(self, resume_from: Optional[str] = None, start_generation: int = 0):
        """Main training loop"""
        logger.info("Starting evolutionary training...")
        
        # Resume from checkpoint if specified
        if resume_from:
            start_generation = self.load_population_checkpoint(resume_from) + 1
            logger.info(f"Resuming training from generation {start_generation}")
        
        try:
            for generation in range(start_generation, self.config.generations):
                # Evaluate current population
                self._evaluate_population(generation)
                
                # Save best individual and plot progress every 10 generations
                if generation % 10 == 0:
                    self._save_best_individual(generation)
                    self._save_population_checkpoint(generation)  # Save population checkpoint
                    self._plot_progress()
                
                # # Check for early stopping (optional)
                # if generation > 50 and self.history['best_fitness'][-1] > 1000:  # Adjust threshold as needed
                #     logger.info(f"Early stopping at generation {generation} due to high fitness")
                #     break
                
                # Create next generation (skip for last generation)
                if generation < self.config.generations - 1:
                    self.population = self._create_next_generation()
            
            # Final evaluation and save
            logger.info("Training completed!")
            final_generation = len(self.history['generation']) - 1
            self._save_best_individual(final_generation, final=True)
            self._save_population_checkpoint(final_generation)  # Save final checkpoint
            self._plot_progress(final=True)
            self._save_training_history()
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            final_generation = len(self.history['generation']) - 1
            self._save_best_individual(final_generation, final=True)
            self._save_population_checkpoint(final_generation)  # Save interrupted checkpoint
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
    
    def _save_population_checkpoint(self, generation: int):
        """Save entire population for resuming training"""
        checkpoint_path = self.save_dir / f"population_checkpoint_gen_{generation}.pkl"
        
        population_data = []
        for individual in self.population:
            individual_data = {
                'weights': individual.network.get_weights(),
                'fitness': individual.fitness,
                'raw_scores': individual.raw_scores,
                'network_architecture': individual.network.layer_sizes
            }
            population_data.append(individual_data)
        
        checkpoint = {
            'generation': generation,
            'population': population_data,
            'config': {
                'population_size': self.config.population_size,
                'mutation_rate': self.config.mutation_rate,
                'crossover_rate': self.config.crossover_rate,
                'elite_ratio': self.config.elite_ratio,
                'tournament_size': self.config.tournament_size,
                'evaluations_per_individual': self.config.evaluations_per_individual
            },
            'env_config': self.env_config,
            'history': self.history
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved population checkpoint for generation {generation} to {checkpoint_path}")
    
    def load_population_checkpoint(self, checkpoint_path: str) -> int:
        """Load population from checkpoint and return the generation number"""
        logger.info(f"Loading population checkpoint from {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        generation = checkpoint['generation']
        population_data = checkpoint['population']
        self.history = checkpoint.get('history', {
            'generation': [],
            'best_fitness': [],
            'average_fitness': [],
            'worst_fitness': [],
            'fitness_std': []
        })
        
        # Reconstruct population
        self.population = []
        for individual_data in population_data:
            # Create network with saved architecture
            arch = individual_data['network_architecture']
            network = SimpleNeuralNet(arch[0], arch[1:-1], arch[-1])
            network.set_weights(individual_data['weights'])
            
            # Create individual
            individual = Individual(network, individual_data['fitness'])
            individual.raw_scores = individual_data.get('raw_scores', [])
            self.population.append(individual)
        
        logger.info(f"Loaded population of {len(self.population)} individuals from generation {generation}")
        return generation
    
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
    """Load a trained evolutionary agent (pickle or torch state_dict)"""
    try:
        with open(weights_path, 'rb') as f:
            data = pickle.load(f)
        # Reconstruct network from pickled data
        if 'network_architecture' in data:
            layer_sizes = data['network_architecture']
            network = SimpleNeuralNet(layer_sizes[0], layer_sizes[1:-1], layer_sizes[-1])
        else:
            network = SimpleNeuralNet(obs_size)
        network.set_weights(data['weights'])
    except pickle.UnpicklingError:
        # Fallback to loading PyTorch state_dict (.pt files)
        state_dict = torch.load(weights_path, map_location='cpu')
        network = SimpleNeuralNet(obs_size)
        network.load_state_dict(state_dict)
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

def create_checkpoint_from_best_individual(best_individual_path: str, checkpoint_path: str, 
                                         population_size: int = 100, generation: int = 0):
    """Create a population checkpoint by replicating the best individual with mutations"""
    logger.info(f"Creating checkpoint from best individual: {best_individual_path}")
    
    # Load the best individual
    with open(best_individual_path, 'rb') as f:
        best_data = pickle.load(f)
    
    # Create population by mutating the best individual
    population_data = []
    
    # First individual is the exact copy of the best
    population_data.append({
        'weights': best_data['weights'].copy(),
        'fitness': best_data.get('fitness', 0.0),
        'raw_scores': best_data.get('raw_scores', []),
        'network_architecture': best_data['network_architecture']
    })
    
    # Create the rest with mutations
    for i in range(1, population_size):
        weights = best_data['weights'].copy()
        
        # Apply random mutations (stronger for diversity)
        mutation_rate = 0.2 + (i / population_size) * 0.3  # Varying mutation rates
        mutation_strength = 0.1 + (i / population_size) * 0.2
        
        mutation_mask = np.random.random(weights.shape) < mutation_rate
        mutations = np.random.normal(0, mutation_strength, weights.shape)
        weights[mutation_mask] += mutations[mutation_mask]
        
        population_data.append({
            'weights': weights,
            'fitness': 0.0,  # Will be evaluated
            'raw_scores': [],
            'network_architecture': best_data['network_architecture']
        })
    
    # Create checkpoint structure
    checkpoint = {
        'generation': generation,
        'population': population_data,
        'config': {
            'population_size': population_size,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7,
            'elite_ratio': 0.1,
            'tournament_size': 5,
            'evaluations_per_individual': 3
        },
        'env_config': {
            'render': False,
            'reward_config': {
                'distance_weight': 1.0,
                'survival_weight': 0.1,
                'crash_penalty': -10.0,
                'action_penalty': -0.01,
                'speed_bonus': 0.05  # Reward for driving fast
            }
        },
        'history': {
            'generation': [],
            'best_fitness': [],
            'average_fitness': [],
            'worst_fitness': [],
            'fitness_std': []
        }
    }
    
    # Save checkpoint
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    logger.info(f"Created population checkpoint with {population_size} individuals at {checkpoint_path}")
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser(description="Train race car agent using evolutionary algorithm")
    parser.add_argument("--population-size", type=int, default=128, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate")
    parser.add_argument("--elite-ratio", type=float, default=0.2, help="Elite ratio to keep unchanged")
    parser.add_argument("--tournament-size", type=int, default=16, help="Tournament selection size")
    parser.add_argument("--evaluations", type=int, default=8, help="Episodes per individual evaluation")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--save-dir", type=str, default="evolutionary_results", help="Directory to save results")
    parser.add_argument("--test", type=str, default=None, help="Path to weights file for testing")
    parser.add_argument("--test-episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--render", action="store_true", help="Enable rendering during training")
    parser.add_argument("--resume", type=str, default=None, help="Path to population checkpoint file to resume training from")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from the latest checkpoint in save-dir")
    parser.add_argument("--create-checkpoint", type=str, default=None, help="Create population checkpoint from best individual file")
    parser.add_argument("--checkpoint-output", type=str, default=None, help="Output path for created checkpoint (used with --create-checkpoint)")
    parser.add_argument("--selection-pressure", type=float, default=2.0, help="Selection pressure for rank-based selection")
    parser.add_argument("--min-diversity", type=float, default=0.1, help="Minimum population diversity to maintain")
    
    args = parser.parse_args()
    
    # Environment configuration
    env_config = {
        'render': args.render,
        'reward_config': {
            'distance_weight': 1.0,
            'survival_weight': 0.1,
            'crash_penalty': -100.0,
            'action_penalty': -0.01,
            'speed_bonus': 0.5 
        }
    }
    
    if args.test:
        # Test mode
        test_evolved_agent(args.test, env_config, args.test_episodes)
    elif args.create_checkpoint:
        # Create checkpoint mode
        output_path = args.checkpoint_output or f"{args.save_dir}/population_checkpoint_from_best.pkl"
        create_checkpoint_from_best_individual(
            args.create_checkpoint, 
            output_path, 
            args.population_size
        )
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
            max_workers=args.workers,
            fitness_sharing=True,
            selection_pressure=args.selection_pressure,
            min_population_diversity=args.min_diversity
        )
        
        trainer = EvolutionaryTrainer(config, env_config, args.save_dir)
        
        # Handle resume options
        resume_from = None
        if args.resume:
            resume_from = args.resume
        elif args.resume_latest:
            # Find the latest checkpoint in save_dir
            save_path = Path(args.save_dir)
            if save_path.exists():
                checkpoint_files = list(save_path.glob("population_checkpoint_gen_*.pkl"))
                if checkpoint_files:
                    # Sort by generation number
                    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
                    resume_from = str(checkpoint_files[-1])
                    logger.info(f"Found latest checkpoint: {resume_from}")
                else:
                    logger.warning("No checkpoint files found for --resume-latest")
            else:
                logger.warning(f"Save directory {args.save_dir} does not exist")
        
        trainer.train(resume_from=resume_from)

if __name__ == "__main__":
    main()
