"""
Hyperparameter configuration for DQN lane switching agent.
"""

# DQN Agent Configuration
DQN_CONFIG = {
    'state_size': 117,  # 16 sensors × 7 timesteps + 5 additional features
    'action_size': 3,   # left, right, do_nothing
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 100000,
    'batch_size': 32,
    'target_update_frequency': 1000,
    'cooldown_ticks': 180,  # 3 seconds at 60 FPS
    'use_double_dqn': True,
    'use_prioritized_replay': False,
    'device': None  # Auto-detect CUDA/CPU
}

# Neural Network Architecture
NETWORK_CONFIG = {
    'hidden_layers': (256, 128, 64),
    'dropout_rate': 0.1,
    'activation': 'relu',
    'weight_init': 'xavier'
}

# Training Configuration
TRAINING_CONFIG = {
    'max_episodes': 10000,
    'max_episode_steps': 3600,  # 60 seconds at 60 FPS (matches game MAX_TICKS)
    'training_start_step': 1000,  # Start training after collecting experiences
    'train_frequency': 4,  # Train every N steps
    'eval_frequency': 100,  # Evaluate every N episodes
    'save_frequency': 500,  # Save model every N episodes
    'log_frequency': 10,  # Log statistics every N episodes
}

# Environment Configuration (matches game specifications exactly)
ENV_CONFIG = {
    'screen_width': 1600,  # SCREEN_WIDTH from core.py
    'screen_height': 1200,  # SCREEN_HEIGHT from core.py
    'lane_count': 5,  # LANE_COUNT from core.py
    'fps': 60,  # Game runs at 60 FPS
    'max_ticks': 3600,  # 60 seconds * 60 FPS (MAX_TICKS from core.py)
    'spawn_cooldown': 60,  # Minimum ticks between car spawns
    'max_cars': 5,
    'verbose': False  # Set to True for visual debugging
}

# Reward Configuration
REWARD_CONFIG = {
    'distance_weight': 1.0,
    'speed_weight': 0.1,
    'survival_weight': 0.5,
    'crash_penalty': -1000.0,
    'wall_penalty': -500.0,
    'near_miss_penalty_scale': 5.0,
    'overtake_bonus': 50.0,
    'positioning_bonus': 2.0,
    'cooldown_violation_penalty': -100.0
}

# Curriculum Learning Configuration
CURRICULUM_CONFIG = {
    'enabled': True,
    'stage_episodes': [3000, 6000, 10000],  # Episode thresholds for each stage
    'reward_weights': {
        'stage_0': {'exploration': 1.0, 'safety': 2.0, 'efficiency': 0.5},
        'stage_1': {'exploration': 0.5, 'safety': 1.0, 'efficiency': 1.0},
        'stage_2': {'exploration': 0.1, 'safety': 1.0, 'efficiency': 1.5}
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_dir': 'logs',
    'model_dir': 'models/checkpoints',
    'tensorboard_dir': 'runs',
    'log_level': 'INFO',
    'save_replay_buffer': False,
    'log_gradients': False
}

# Evaluation Configuration
EVAL_CONFIG = {
    'num_eval_episodes': 10,
    'eval_epsilon': 0.0,  # No exploration during evaluation
    'eval_deterministic': True,
    'record_video': False,
    'eval_metrics': ['distance', 'survival_time', 'crashes', 'lane_changes']
}

# Complete configuration combining all components
FULL_CONFIG = {
    'dqn': DQN_CONFIG,
    'network': NETWORK_CONFIG,
    'training': TRAINING_CONFIG,
    'environment': ENV_CONFIG,
    'reward': REWARD_CONFIG,
    'curriculum': CURRICULUM_CONFIG,
    'logging': LOGGING_CONFIG,
    'evaluation': EVAL_CONFIG
}

# Quick access to commonly used configs
def get_default_config():
    """Get default configuration for standard training."""
    return FULL_CONFIG

def get_debug_config():
    """Get configuration optimized for debugging/testing."""
    debug_config = FULL_CONFIG.copy()
    debug_config['training']['max_episodes'] = 100
    debug_config['training']['max_episode_steps'] = 3600  # Keep full 60-second episodes
    debug_config['dqn']['buffer_size'] = 10000
    debug_config['environment']['verbose'] = True
    debug_config['logging']['log_frequency'] = 1
    return debug_config

def get_fast_config():
    """Get configuration for quick testing."""
    fast_config = FULL_CONFIG.copy()
    fast_config['training']['max_episodes'] = 1000
    fast_config['training']['max_episode_steps'] = 3600  # Keep full 60-second episodes
    fast_config['dqn']['buffer_size'] = 50000
    fast_config['dqn']['target_update_frequency'] = 500
    return fast_config