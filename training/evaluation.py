"""
Evaluation script for DQN lane switching agent.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from typing import Dict, List, Any
import pygame

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agents.dqn_agent import DQNAgent
from src.agents.training_env import TrainingEnvironment
from src.game.core import initialize_game_state, game_loop_with_dqn
from src.game.agent import RuleBasedAgent
from training.hyperparameters import get_default_config


class AgentEvaluator:
    """
    Evaluator for comparing DQN agent performance against baselines.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config['evaluation']
        
    def evaluate_dqn_agent(self, model_path: str, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate trained DQN agent.
        
        Args:
            model_path: Path to trained model
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results dictionary
        """
        print(f"🤖 Evaluating DQN agent: {model_path}")
        
        # Load trained agent
        agent = DQNAgent(**self.config['dqn'])
        agent.load_model(model_path)
        agent.set_eval_mode()
        
        # Run evaluation episodes
        results = []
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}", end=" ")
            
            # Initialize game
            pygame.init()
            initialize_game_state("http://localhost:9052", f"eval_{episode}")
            
            # Run episode with DQN agent
            distance, crashed, ticks = game_loop_with_dqn(
                dqn_agent=agent,
                verbose=False,
                log_actions=False
            )
            
            episode_result = {
                'episode': episode,
                'distance': distance,
                'crashed': crashed,
                'survival_time': ticks / 60.0,  # Convert ticks to seconds
                'success': not crashed
            }
            results.append(episode_result)
            
            status = "💥 CRASHED" if crashed else "✅ SUCCESS"
            print(f"- Distance: {distance:.1f}, {status}")
            
            pygame.quit()
        
        # Calculate statistics
        distances = [r['distance'] for r in results]
        crashes = [r['crashed'] for r in results]
        survival_times = [r['survival_time'] for r in results]
        
        stats = {
            'agent_type': 'DQN',
            'model_path': model_path,
            'num_episodes': num_episodes,
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances),
            'crash_rate': np.mean(crashes),
            'success_rate': 1.0 - np.mean(crashes),
            'avg_survival_time': np.mean(survival_times),
            'results': results
        }
        
        return stats
    
    def evaluate_rule_based_agent(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate rule-based agent baseline.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation results dictionary
        """
        print("🔧 Evaluating rule-based agent baseline")
        
        results = []
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}", end=" ")
            
            # Initialize game
            pygame.init()
            initialize_game_state("http://localhost:9052", f"baseline_{episode}")
            
            # Run episode with rule-based agent only
            distance, crashed, ticks = game_loop_with_dqn(
                dqn_agent=None,  # No DQN agent
                verbose=False,
                log_actions=False
            )
            
            episode_result = {
                'episode': episode,
                'distance': distance,
                'crashed': crashed,
                'survival_time': ticks / 60.0,
                'success': not crashed
            }
            results.append(episode_result)
            
            status = "💥 CRASHED" if crashed else "✅ SUCCESS"
            print(f"- Distance: {distance:.1f}, {status}")
            
            pygame.quit()
        
        # Calculate statistics
        distances = [r['distance'] for r in results]
        crashes = [r['crashed'] for r in results]
        survival_times = [r['survival_time'] for r in results]
        
        stats = {
            'agent_type': 'Rule-Based',
            'model_path': None,
            'num_episodes': num_episodes,
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances),
            'min_distance': np.min(distances),
            'crash_rate': np.mean(crashes),
            'success_rate': 1.0 - np.mean(crashes),
            'avg_survival_time': np.mean(survival_times),
            'results': results
        }
        
        return stats
    
    def compare_agents(self, dqn_stats: Dict[str, Any], baseline_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare DQN agent performance against baseline.
        
        Args:
            dqn_stats: DQN agent evaluation statistics
            baseline_stats: Baseline agent evaluation statistics
            
        Returns:
            Comparison results
        """
        comparison = {
            'distance_improvement': (dqn_stats['avg_distance'] - baseline_stats['avg_distance']) / baseline_stats['avg_distance'] * 100,
            'success_rate_improvement': (dqn_stats['success_rate'] - baseline_stats['success_rate']) * 100,
            'survival_time_improvement': (dqn_stats['avg_survival_time'] - baseline_stats['avg_survival_time']) / baseline_stats['avg_survival_time'] * 100,
            'dqn_stats': dqn_stats,
            'baseline_stats': baseline_stats
        }
        
        return comparison
    
    def create_comparison_plots(self, comparison: Dict[str, Any], save_path: str = None):
        """Create comparison plots."""
        dqn_stats = comparison['dqn_stats']
        baseline_stats = comparison['baseline_stats']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distance comparison
        distances_dqn = [r['distance'] for r in dqn_stats['results']]
        distances_baseline = [r['distance'] for r in baseline_stats['results']]
        
        axes[0, 0].hist(distances_dqn, alpha=0.7, label='DQN', bins=10)
        axes[0, 0].hist(distances_baseline, alpha=0.7, label='Rule-Based', bins=10)
        axes[0, 0].set_title('Distance Distribution')
        axes[0, 0].set_xlabel('Distance')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Success rate comparison
        success_rates = [dqn_stats['success_rate'], baseline_stats['success_rate']]
        agent_names = ['DQN', 'Rule-Based']
        axes[0, 1].bar(agent_names, success_rates)
        axes[0, 1].set_title('Success Rate Comparison')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        
        # Average distance comparison
        avg_distances = [dqn_stats['avg_distance'], baseline_stats['avg_distance']]
        axes[1, 0].bar(agent_names, avg_distances)
        axes[1, 0].set_title('Average Distance Comparison')
        axes[1, 0].set_ylabel('Average Distance')
        
        # Survival time comparison
        survival_times = [dqn_stats['avg_survival_time'], baseline_stats['avg_survival_time']]
        axes[1, 1].bar(agent_names, survival_times)
        axes[1, 1].set_title('Average Survival Time Comparison')
        axes[1, 1].set_ylabel('Survival Time (seconds)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"📊 Comparison plots saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_comparison_report(self, comparison: Dict[str, Any]):
        """Print detailed comparison report."""
        dqn_stats = comparison['dqn_stats']
        baseline_stats = comparison['baseline_stats']
        
        print("\n" + "="*60)
        print("📊 AGENT PERFORMANCE COMPARISON REPORT")
        print("="*60)
        
        print(f"\n🤖 DQN Agent Results:")
        print(f"   • Average Distance: {dqn_stats['avg_distance']:.1f} ± {dqn_stats['std_distance']:.1f}")
        print(f"   • Success Rate: {dqn_stats['success_rate']:.1%}")
        print(f"   • Crash Rate: {dqn_stats['crash_rate']:.1%}")
        print(f"   • Avg Survival Time: {dqn_stats['avg_survival_time']:.1f}s")
        print(f"   • Best Distance: {dqn_stats['max_distance']:.1f}")
        
        print(f"\n🔧 Rule-Based Agent Results:")
        print(f"   • Average Distance: {baseline_stats['avg_distance']:.1f} ± {baseline_stats['std_distance']:.1f}")
        print(f"   • Success Rate: {baseline_stats['success_rate']:.1%}")
        print(f"   • Crash Rate: {baseline_stats['crash_rate']:.1%}")
        print(f"   • Avg Survival Time: {baseline_stats['avg_survival_time']:.1f}s")
        print(f"   • Best Distance: {baseline_stats['max_distance']:.1f}")
        
        print(f"\n📈 Performance Improvements:")
        print(f"   • Distance: {comparison['distance_improvement']:+.1f}%")
        print(f"   • Success Rate: {comparison['success_rate_improvement']:+.1f}%")
        print(f"   • Survival Time: {comparison['survival_time_improvement']:+.1f}%")
        
        # Performance verdict
        if comparison['distance_improvement'] > 10:
            verdict = "🎉 Excellent improvement!"
        elif comparison['distance_improvement'] > 0:
            verdict = "✅ Good improvement!"
        elif comparison['distance_improvement'] > -10:
            verdict = "⚠️  Similar performance"
        else:
            verdict = "❌ Needs improvement"
        
        print(f"\n🏆 Overall Verdict: {verdict}")
        print("="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate DQN Lane Switching Agent')
    parser.add_argument('--model', type=str, help='Path to trained DQN model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--baseline-only', action='store_true', help='Evaluate only baseline agent')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = get_default_config()
    evaluator = AgentEvaluator(config)
    
    if args.baseline_only:
        # Evaluate only baseline
        baseline_stats = evaluator.evaluate_rule_based_agent(args.episodes)
        
        # Save results
        results_path = os.path.join(args.output_dir, 'baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(baseline_stats, f, indent=2)
        
        print(f"\n💾 Results saved: {results_path}")
        
    else:
        # Evaluate both agents
        if not args.model:
            print("❌ Please specify --model path for DQN evaluation")
            return
        
        if not os.path.exists(args.model):
            print(f"❌ Model file not found: {args.model}")
            return
        
        # Evaluate DQN agent
        dqn_stats = evaluator.evaluate_dqn_agent(args.model, args.episodes)
        
        # Evaluate baseline
        baseline_stats = evaluator.evaluate_rule_based_agent(args.episodes)
        
        # Compare results
        comparison = evaluator.compare_agents(dqn_stats, baseline_stats)
        
        # Print report
        evaluator.print_comparison_report(comparison)
        
        # Create plots
        plots_path = os.path.join(args.output_dir, 'comparison_plots.png')
        evaluator.create_comparison_plots(comparison, plots_path)
        
        # Save results
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n💾 Results saved: {results_path}")


if __name__ == "__main__":
    main()