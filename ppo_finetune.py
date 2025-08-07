"""
Fine-tune an evolved evolutionary model using PPO.
"""

import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
import torch.nn as nn
from evolutionary_model_wrapper import load_evolutionary_model
from race_car_env import make_race_car_env
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path


class BestModelCallback:
    """
    Callback to track and save the best model based on distance traveled.
    """
    def __init__(self, best_model_path="best_model.zip", metadata_path="best_model_metadata.json"):
        self.best_model_path = best_model_path
        self.metadata_path = metadata_path
        self.best_distance = self._load_best_distance()
        
    def _load_best_distance(self):
        """Load the best distance from metadata file if it exists."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get('best_distance', 0.0)
            except:
                return 0.0
        return 0.0
    
    def _save_metadata(self, distance, timestep, episode):
        """Save metadata about the best model."""
        metadata = {
            'best_distance': float(distance),  # ensure JSON serializable
            'timestep': int(timestep),         # ensure JSON serializable
            'episode': int(episode),          # ensure JSON serializable
            'timestamp': datetime.now().isoformat(),
            'model_path': self.best_model_path
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def update_best(self, model, distance, timestep, episode):
        """Update the best model if the current distance is better."""
        if distance > self.best_distance:
            print(f"New best model! Distance: {distance:.2f} (previous: {self.best_distance:.2f})")
            self.best_distance = distance
            model.save(self.best_model_path)
            self._save_metadata(distance, timestep, episode)
            return True
        return False


def evaluate_model(model, env, n_episodes=10, render=False, per_env=False):
    """
    Evaluate a model and return average distance traveled.
    """
    # Parallel evaluation for vectorized envs (including single-env VecEnv)
    if hasattr(env, 'num_envs'):
        n_envs = env.num_envs
        # Determine total episodes to run across all envs when per_env is True
        target_episodes = n_episodes * n_envs if per_env else n_episodes
        episodes_done = 0
        total_distances = []
        current_distances = [0.0] * n_envs
        dones = [False] * n_envs
        # Reset all envs at once
        # Reset vector env; handle different reset return signatures
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, infos = reset_result[0], reset_result[-1]
        else:
            obs, infos = reset_result, {}
        while episodes_done < target_episodes:
            actions, _ = model.predict(obs, deterministic=True)
            # Handle different step return signatures
            step_result = env.step(actions)
            if len(step_result) == 5:
                obs, rewards, terminateds, truncateds, infos = step_result
            elif len(step_result) == 4:
                obs, rewards, dones_vec, infos = step_result
                terminateds = dones_vec
                truncateds = dones_vec
            else:
                raise ValueError(f"Unexpected number of elements from env.step: {len(step_result)}")
            for i in range(n_envs):
                if not dones[i]:
                    current_distances[i] = infos[i].get('distance', current_distances[i])
                    if terminateds[i] or truncateds[i]:
                        total_distances.append(current_distances[i])
                        episodes_done += 1
                        dones[i] = True
                        current_distances[i] = 0.0
            if render:
                env.render()
            # If all envs finished but more episodes needed, reset
            if episodes_done < target_episodes and all(dones):
                # Reset and unpack
                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    obs, infos = reset_result[0], reset_result[-1]
                else:
                    obs, infos = reset_result, {}
                dones = [False] * n_envs
        avg_distance = np.mean(total_distances)
        std_distance = np.std(total_distances)
        print(f"Average distance: {avg_distance:.2f} ± {std_distance:.2f}")
        return avg_distance, total_distances
    # Fallback sequential evaluation
    total_distances = []
    for episode in range(n_episodes):
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        done = False
        episode_distance = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_vec, reward_vec, done_vec, info_vec = step_result
                obs = obs_vec[0] if hasattr(obs_vec, '__len__') else obs_vec
                done = done_vec[0] if hasattr(done_vec, '__len__') else done_vec
                info = info_vec[0] if isinstance(info_vec, (list, tuple)) else info_vec
            episode_distance = info.get('distance', 0.0)
            if render:
                env.render()
        total_distances.append(episode_distance)
        print(f"Episode {episode + 1}: Distance = {episode_distance:.2f}")
    avg_distance = np.mean(total_distances)
    std_distance = np.std(total_distances)
    print(f"Average distance: {avg_distance:.2f} ± {std_distance:.2f}")
    return avg_distance, total_distances


def copy_evolutionary_weights_to_policy(evo_wrapper: object, policy_model: PPO):
    """
    Copy the shared hidden layer weights from the evolved network into the PPO policy and value networks.
    """
    # Get evolved network sequential layers
    evo_seq = evo_wrapper.network.network
    # Extract linear layers
    linear_layers = [l for l in evo_seq if isinstance(l, nn.Linear)]
    # Number of hidden layers is total linear layers minus output layer
    num_hidden = len(linear_layers) - 1

    # Extract policy and value MLPs
    extractor = policy_model.policy.mlp_extractor
    policy_linears = [l for l in extractor.policy_net if isinstance(l, nn.Linear)]
    value_linears = [l for l in extractor.value_net if isinstance(l, nn.Linear)]

    # Copy each hidden layer weights and biases
    for idx in range(num_hidden):
        with torch.no_grad():
            policy_linears[idx].weight.copy_(linear_layers[idx].weight)
            policy_linears[idx].bias.copy_(linear_layers[idx].bias)
            value_linears[idx].weight.copy_(linear_layers[idx].weight)
            value_linears[idx].bias.copy_(linear_layers[idx].bias)
    # Copy output layer weights to PPO action head so policy matches evolutionary model
    with torch.no_grad():
        evo_output = linear_layers[-1]
        # action_net: linear mapping from hidden to action space
        policy_model.policy.action_net.weight.copy_(evo_output.weight)
        policy_model.policy.action_net.bias.copy_(evo_output.bias)


def make_vec_env(render: bool, n_envs: int = 16, use_subproc: bool = True, reward_config: dict = None, eval_mode: bool = False):
    """
    Create a vectorized RaceCarEnv for PPO with multiple environments.
    Defaults to training reward_config, but sets evaluation config when eval_mode is True.
    Can still be overridden with reward_config.
    """
    def make_env_fn():
        def _init():
            # Determine reward config: explicit override, eval defaults, or training defaults
            if reward_config is not None:
                rc = reward_config
            elif eval_mode:
                rc = {'distance_progress': 1.0}
            else:
                rc = {'distance_progress': 1.0, 'crash_penalty': -100.0, 'speed_bonus': 0.5}
            return make_race_car_env({'render': render, 'reward_config': rc})
        return _init

    env_fns = [make_env_fn() for _ in range(n_envs)]
    if use_subproc:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune evolutionary model using PPO")
    parser.add_argument('--model-path', type=str, required=True, help="Path to evolved model (.pkl) or PPO model (.zip)")
    parser.add_argument('--timesteps', type=int, default=5_000_000, help="Number of PPO timesteps")
    parser.add_argument('--learning-rate', type=float, default=5e-5, help="PPO learning rate")
    parser.add_argument('--batch-size', type=int, default=256, help="PPO batch size")
    parser.add_argument('--n-epochs', type=int, default=10, help="Number of epochs per update")
    parser.add_argument('--clip-range', type=float, default=0.2, help="PPO clip range")
    parser.add_argument('--ent-coef', type=float, default=0.05, help="Entropy coefficient")
    parser.add_argument('--vf-coef', type=float, default=0.5, help="Value function coefficient")
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help="Max gradient norm")
    parser.add_argument('--n-envs', type=int, default=16, help="Number of parallel environments")
    parser.add_argument('--render', action='store_true', help="Enable environment rendering")
    parser.add_argument('--output-path', type=str, default='ppo_finetuned_model.zip', help="Path to save fine-tuned model")
    parser.add_argument('--best-model-path', type=str, default='best_model.zip', help="Path to save best model")
    parser.add_argument('--checkpoint-freq', type=int, default=500_000, help="Checkpoint frequency (timesteps)")
    parser.add_argument('--eval-freq', type=int, default=250_000, help="Evaluation frequency (timesteps)")
    parser.add_argument('--mode', choices=['train', 'eval', 'resume'], default='train', help="Mode: train, eval, or resume")
    parser.add_argument('--eval-episodes', type=int, default=16, help="Number of episodes during evaluation")
    parser.add_argument('--resume-timesteps', type=int, default=None, help="Additional timesteps when resuming (if None, uses --timesteps)")
    args = parser.parse_args()
    
    # Create directories for checkpoints and logs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Determine mode
    if args.mode == 'eval':
        # Evaluate a saved PPO model
        print(f"Evaluating model: {args.model_path}")
        # Use parallel vec env when not rendering, else fall back to single env
        if not args.render:
            env_eval = make_vec_env(
                render=False,
                n_envs=args.eval_episodes,
                use_subproc=True,
                eval_mode=True
            )
            env_eval = VecMonitor(env_eval)
        else:
            raw_env = make_race_car_env({
                'render': True,
                'reward_config': {'distance_progress': 1.0}
            })
            env_eval = Monitor(raw_env)
        # Load trained PPO model
        model_eval = PPO.load(args.model_path)
        # Evaluate model
        avg_distance, distances = evaluate_model(
            model_eval,
            env_eval,
            args.eval_episodes,
            args.render
        )
        env_eval.close()
        return

    # Training or Resume mode
    # Prepare vectorized environment
    env = make_vec_env(args.render, n_envs=args.n_envs, use_subproc=not args.render)
    # Wrap training env with VecMonitor to match eval env type and suppress warning
    env = VecMonitor(env)
    
    # Create evaluation environment with subprocess vectorization and wrap in VecMonitor using eval defaults
    eval_env = make_vec_env(
        render=False,
        n_envs=args.eval_episodes,
        use_subproc=True,
        eval_mode=True
    )
    eval_env = VecMonitor(eval_env)
    
    # Initialize best model callback
    best_model_callback = BestModelCallback(args.best_model_path, args.best_model_path.replace('.zip', '_metadata.json'))
    
    if args.mode == 'resume':
        # Resume training from existing PPO model
        print(f"Resuming training from: {args.model_path}")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        model = PPO.load(args.model_path, env=env)
        total_timesteps = args.resume_timesteps or args.timesteps
    else:
        # Training mode - determine model type by extension
        ext = os.path.splitext(args.model_path)[1].lower()
        if ext == '.zip':
            # Load existing PPO model for fine-tuning
            print(f"Fine-tuning existing PPO model: {args.model_path}")
            model = PPO.load(args.model_path, env=env)
        elif ext in ['.pkl', '.pt']:
            # Load evolved evolutionary model and initialize PPO
            print(f"Initializing PPO from evolutionary model: {args.model_path}")
            evo_wrapper = load_evolutionary_model(args.model_path)
            # Derive hidden sizes from evolved network
            evo_seq = evo_wrapper.network.network
            linear_layers = [l for l in evo_seq if isinstance(l, nn.Linear)]
            hidden_sizes = [layer.out_features for layer in linear_layers[:-1]]  # exclude output layer
            
            # Configure PPO with optimized hyperparameters for fine-tuning
            policy_kwargs = dict(
                net_arch=dict(pi=hidden_sizes, vf=hidden_sizes),
                activation_fn=nn.ReLU,
                ortho_init=False  # Preserve evolved weights
            )
            
            model = PPO(
                'MlpPolicy', 
                env, 
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                policy_kwargs=policy_kwargs, 
                verbose=1,
                tensorboard_log="./logs/"
            )
            # Initialize PPO policy weights from evolved model
            copy_evolutionary_weights_to_policy(evo_wrapper, model)
        else:
            raise ValueError(f"Unsupported model file format: {ext}")
        
        total_timesteps = args.timesteps

    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,  # Adjusted for vectorized env
        save_path="./checkpoints/",
        name_prefix="ppo_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Custom evaluation callback that updates best model
    class CustomEvalCallback(EvalCallback):
        def __init__(self, *args, best_callback=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_callback = best_callback
            self.episode_count = 0
        
        def _on_step(self) -> bool:
            result = super()._on_step()
            
            # Check if evaluation was performed
            if len(self.evaluations_results) > 0:
                # Get average distance from last evaluation (rewards already total per episode)
                last_eval_distances = list(self.evaluations_results[-1])
                avg_distance = np.mean(last_eval_distances)
                self.episode_count += len(last_eval_distances)
                
                # Update best model if needed
                if self.best_callback:
                    self.best_callback.update_best(
                        self.model, 
                        avg_distance, 
                        self.num_timesteps, 
                        self.episode_count
                    )
            
            return result
    
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=16,
        eval_freq=args.eval_freq // args.n_envs,  # Adjusted for vectorized env
        best_model_save_path=None,  # We handle this with our custom callback
        log_path="./logs/",
        deterministic=True,
        best_callback=best_model_callback
    )
    callbacks.append(eval_callback)
    
    # Create callback list
    callback_list = CallbackList(callbacks)

    print(f"Starting training for {total_timesteps:,} timesteps...")
    print(f"Hyperparameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  N epochs: {args.n_epochs}")
    print(f"  Clip range: {args.clip_range}")
    print(f"  Entropy coef: {args.ent_coef}")
    print(f"  Value coef: {args.vf_coef}")
    print(f"  Max grad norm: {args.max_grad_norm}")
    print(f"  N environments: {args.n_envs}")
    
    # Fine-tune with callbacks and progress bar
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback_list,
        progress_bar=True,
        reset_num_timesteps=(args.mode != 'resume')
    )

    # Save final model
    model.save(args.output_path)
    print(f"Final model saved to {args.output_path}")
    
    # Final evaluation
    print("\nPerforming final evaluation...")
    final_avg_distance, final_distances = evaluate_model(model, eval_env, args.eval_episodes)
    
    # Update best model one last time
    best_model_callback.update_best(model, final_avg_distance, total_timesteps, len(final_distances))
    
    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
