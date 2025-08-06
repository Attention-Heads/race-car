import os
import time

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch

from model import PPOAgent
from environment import RaceCarEnvironment, GameState
from simulator import HeadlessGameSimulator

class PPOTrainer:
    def __init__(self, 
                 episodes: int = 1000000,
                 max_steps: int = 3600,
                 rollout_length: int = 2048,
                 save_freq: int = 1000,
                 model_path: str = "models/model.pth",
                 verbose: bool = False):
        
        self.episodes = episodes
        self.max_steps = max_steps
        self.rollout_length = rollout_length
        self.save_freq = save_freq
        self.model_path = model_path
        self.verbose = verbose
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.env = RaceCarEnvironment(verbose=self.verbose)
        self.agent = PPOAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            lr=1e-4,
            gamma=0.99,
            epsilon=0.2,
            gae_lambda=0.95,
            entropy_coef=0.001, 
            value_loss_coef=1.0,
            ppo_epochs=10, 
            batch_size=64,
            rollout_length=rollout_length
        )
        self.simulator = HeadlessGameSimulator()
        
        self.episode_rewards = []
        self.episode_distances = []
        self.episode_lengths = []
        self.crash_rates = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self._debug_print_done = False
    
    def collect_rollout(self) -> Tuple[List[float], List[int], List[bool]]:
        rewards = []
        distances = []
        crashes = []
        
        state_raw = self.simulator.reset()
        state = self.env.preprocess_state(state_raw)

        if not self._debug_print_done:
            print("--- First State Vector Verification ---")
            print(f"State vector shape: {state.shape}")
            print(f"Expected shape: ({self.env.state_size},)")
            print(f"Sample of state vector (first 40 elements):\n{state[:40]}")
            print("-----------------------------------------")
            self._debug_print_done = True
        
        episode_reward = 0
        steps = 0
        done = False
        prev_state_raw = state_raw
        
        while not done and steps < self.max_steps and len(self.agent.memory.states) < self.rollout_length:
            action, log_prob, value = self.agent.act(state, training=True)
            action_str = self.env.index_to_action(action)
            
            next_state_raw, reward, done = self.simulator.step(action_str)
            next_state = self.env.preprocess_state(next_state_raw)
            
            if steps > 0:
                reward = self.env.calculate_reward(prev_state_raw, next_state_raw, action_str)
            
            self.agent.store_transition(state, action, log_prob, reward, value, done)
            
            state = next_state
            prev_state_raw = next_state_raw
            episode_reward += reward
            steps += 1
            
            if done:
                rewards.append(episode_reward)
                distances.append(next_state_raw.distance)
                crashes.append(next_state_raw.did_crash)
                
                # Reset for next episode in rollout
                if len(self.agent.memory.states) < self.rollout_length:
                    state_raw = self.simulator.reset()
                    state = self.env.preprocess_state(state_raw)
                    episode_reward = 0
                    steps = 0
                    done = False
                    prev_state_raw = state_raw
        
        # If we exit the loop without completing an episode, add partial results
        if not done and episode_reward > 0:
            rewards.append(episode_reward)
            distances.append(next_state_raw.distance)
            crashes.append(False)  # Didn't crash, just ran out of steps
        
        return rewards, distances, crashes
    
    def train(self, resume: bool = False):
        """Main PPO training loop"""
        if resume:
            self.agent.load(self.model_path)
            print("Resumed training from saved model")
        
        print(f"Starting training for {self.episodes} episodes")
        print(f"Device: {self.agent.device}")
        print(f"Rollout length: {self.rollout_length}")
        
        start_time = time.time()
        training_start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_crashes = 0
        total_episodes = 0
        
        episode = 0
        while episode < self.episodes:
            rollout_start = time.time()
            
            rollout_rewards, rollout_distances, rollout_crashes = self.collect_rollout()
            
            self.episode_rewards.extend(rollout_rewards)
            self.episode_distances.extend(rollout_distances)
            self.episode_lengths.extend([len(rollout_rewards)])
            
            total_crashes += sum(rollout_crashes)
            total_episodes += len(rollout_rewards)
            episode += len(rollout_rewards)
            
            crash_rate = total_crashes / total_episodes if total_episodes > 0 else 0.0
            self.crash_rates.append(crash_rate)
            
            update_info = self.agent.update()
            
            if update_info:
                self.policy_losses.append(update_info['policy_loss'])
                self.value_losses.append(update_info['value_loss'])
                self.entropies.append(update_info['entropy'])
            
            rollout_time = time.time() - rollout_start
            
            # Periodic logging
            if episode % 10 == 0 or len(rollout_rewards) > 0:
                avg_reward = np.mean(rollout_rewards) if rollout_rewards else 0
                avg_distance = np.mean(rollout_distances) if rollout_distances else 0
                
                GREEN = '\033[92m'
                RESET = '\033[0m'
                log_str = (f"Ep {episode:4d} | Ep's in rollout: {len(rollout_rewards):2d} | "
                           f"Avg Rwrd: {avg_reward: 7.2f} | {GREEN}Avg Dist: {avg_distance: 6.0f}{RESET} | "
                           f"Crash Rt: {crash_rate: .3f} | Rollout Time: {rollout_time:.1f}s")
                if update_info:
                    log_str += (f" | Policy Loss: {update_info['policy_loss']: .8f} | "
                                f"Value Loss: {update_info['value_loss']: .4f} | "
                                f"Entropy: {update_info['entropy']: .6f}")
                print(log_str)
            
            # Periodic save
            if episode > 0 and episode % self.save_freq == 0:
                periodic_save_path = os.path.join(
                    os.path.dirname(self.model_path),
                    f"model_{training_start_timestamp}_{episode}.pth"
                )
                self.agent.save(periodic_save_path)
                self.save_training_plots()
                print(f"Saved periodic model to {periodic_save_path}")
        
        # Final save
        self.agent.save(self.model_path)
        self.save_training_plots()
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Total episodes: {total_episodes}")
        print(f"Average final reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"Average final distance: {np.mean(self.episode_distances[-100:]):.0f}")
        print(f"Final crash rate: {crash_rate:.3f}")
    
    def save_training_plots(self):
        """Save training progress plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        if self.episode_distances:
            axes[0, 1].plot(self.episode_distances)
            axes[0, 1].set_title('Episode Distances')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Distance')
            axes[0, 1].grid(True)
        
        if self.crash_rates:
            axes[0, 2].plot(self.crash_rates)
            axes[0, 2].set_title('Crash Rate')
            axes[0, 2].set_xlabel('Rollout')
            axes[0, 2].set_ylabel('Crash Rate')
            axes[0, 2].grid(True)
        
        if self.policy_losses:
            axes[1, 0].plot(self.policy_losses)
            axes[1, 0].set_title('Policy Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        if self.value_losses:
            axes[1, 1].plot(self.value_losses)
            axes[1, 1].set_title('Value Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        if self.entropies:
            axes[1, 2].plot(self.entropies)
            axes[1, 2].set_title('Policy Entropy')
            axes[1, 2].set_xlabel('Update')
            axes[1, 2].set_ylabel('Entropy')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('progress_training.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate trained model"""
        print(f"Evaluating model for {num_episodes} episodes...")
        
        self.agent.load(self.model_path)
        
        eval_rewards = []
        eval_distances = []
        eval_crashes = 0
        
        for episode in range(num_episodes):
            state_raw = self.simulator.reset()
            state = self.env.preprocess_state(state_raw)
            
            total_reward = 0
            steps = 0
            done = False
            prev_state_raw = state_raw
            
            while not done and steps < self.max_steps:
                action = self.agent.act(state, training=False)
                action_str = self.env.index_to_action(action)
                
                next_state_raw, reward, done = self.simulator.step(action_str)
                next_state = self.env.preprocess_state(next_state_raw)
                
                if steps > 0:
                    reward = self.env.calculate_reward(prev_state_raw, next_state_raw, action_str)
                
                state = next_state
                prev_state_raw = next_state_raw
                total_reward += reward
                steps += 1
            
            eval_rewards.append(total_reward)
            eval_distances.append(next_state_raw.distance)
            if next_state_raw.did_crash:
                eval_crashes += 1
            
            print(f"Eval Episode {episode}: Reward {total_reward:.2f}, Distance {next_state_raw.distance}, "
                  f"Crashed: {next_state_raw.did_crash}")
        
        print(f"\nEvaluation Results:")
        print(f"Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"Average Distance: {np.mean(eval_distances):.0f} ± {np.std(eval_distances):.0f}")
        print(f"Crash Rate: {eval_crashes/num_episodes:.3f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Race Car Agent')
    parser.add_argument('--episodes', type=int, default=1000000, help='Number of training episodes')
    parser.add_argument('--rollout', type=int, default=2048, help='Rollout length')
    parser.add_argument('--resume', action='store_true', help='Resume training from saved model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate trained model')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    
    trainer = PPOTrainer(
        episodes=args.episodes,
        rollout_length=args.rollout,
        verbose=args.verbose
    )
    
    if args.evaluate:
        trainer.evaluate(args.eval_episodes)
    else:
        trainer.train(resume=args.resume)