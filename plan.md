# DQN Lane Switching Agent Implementation Plan

## Overview
This plan outlines the implementation of a Deep Q-Network (DQN) agent for intelligent lane switching in the race car environment. The agent will learn to maximize distance traveled while avoiding crashes through strategic lane changes.

## 1. Environment Analysis

### Current State
- 5-lane racing environment with ego car spawning in middle lane
- 16 sensors providing 360° obstacle detection (1000px range)
- Existing cruise control system maintains safe following distance
- Lane change functions: `initiate_change_lane_left()` and `initiate_change_lane_right()`
- Game runs at 60 FPS for up to 60 seconds (3600 ticks)

### Key Constraints
- Safety cooldown: 3+ seconds between lane changes (≥180 ticks at 60 FPS)
- Wall collision prevention (leftmost/rightmost lane boundaries)
- Action space: {LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, DO_NOTHING}

## 2. DQN Architecture Design

### 2.1 State Representation
**Temporal Stacking with Geometric Backoff:**
- Store sensor readings at timesteps: [k, k-1, k-2, k-4, k-8, k-16, k-32]
- 16 sensors × 7 timesteps = 112 input features
- Additional state features:
  - Current lane position (normalized -2 to +2, where 0 is middle lane)
  - Time since last lane change (normalized 0-1, where 1 = full cooldown)
  - Current velocity components (vx normalized by 50, vy normalized by 5)
  - Distance traveled (normalized by episode progress)

**Total Input Size:** ~116 features

### 2.2 Network Architecture
```
Input Layer: 116 features
Hidden Layer 1: 256 neurons (ReLU)
Hidden Layer 2: 128 neurons (ReLU) 
Hidden Layer 3: 64 neurons (ReLU)
Output Layer: 3 neurons (Q-values for each action)
```

### 2.3 DQN Components
- **Main Network:** Current Q-function approximator
- **Target Network:** Stabilized target for training (updated every N steps)
- **Experience Replay:** Store and sample transitions
- **Epsilon-Greedy:** Exploration strategy with decay

## 3. Reward Function Design

### 3.1 Primary Rewards
- **Distance Reward:** +1.0 per unit distance traveled
- **Speed Bonus:** +0.1 × current_velocity (encourages faster driving)
- **Survival Bonus:** +0.5 per tick survived

### 3.2 Penalties
- **Crash Penalty:** -1000 (terminal state)
- **Wall Hit Attempt:** -500 (trying to change lanes at boundaries)
- **Inefficient Lane Change:** -10 (changing lanes without clear benefit)
- **Cooldown Violation:** -100 (attempting lane change during cooldown)

### 3.3 Strategic Rewards
- **Overtaking Bonus:** +50 (successfully passing slower car)
- **Gap Utilization:** +20 (moving into advantageous position)
- **Near Miss Penalty:** -5 × (1/distance_to_obstacle) when distance < 200px

## 4. Implementation Components

### 4.1 New Files to Create
```
src/agents/
├── dqn_agent.py          # Main DQN agent class
├── replay_buffer.py      # Experience replay implementation
├── neural_network.py     # DQN network architecture
└── training_env.py       # Training environment wrapper

src/utils/
├── state_processor.py    # State preprocessing and stacking
└── reward_calculator.py  # Reward function implementation

training/
├── train_dqn.py         # Main training script
├── hyperparameters.py   # Training configuration
└── evaluation.py        # Agent evaluation utilities

models/
└── checkpoints/         # Saved model weights
```

### 4.2 Core Classes

#### DQNAgent
```python
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.q_network = DQNNetwork(state_size, action_size)
        self.target_network = DQNNetwork(state_size, action_size)
        self.replay_buffer = ReplayBuffer(100000)
        self.epsilon = 1.0
        self.last_lane_change_tick = 0
        self.current_lane = 2  # Middle lane
    
    def act(self, state, tick):
        # Enforce cooldown period
        if tick - self.last_lane_change_tick < 180:
            return 2  # DO_NOTHING
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        
        q_values = self.q_network.predict(state)
        return np.argmax(q_values)
    
    def learn(self, batch_size=32):
        # Sample from replay buffer and update networks
```

#### StateProcessor
```python
class StateProcessor:
    def __init__(self):
        self.sensor_history = deque(maxlen=32)
        self.sensor_names = [
            'left_side', 'left_side_front', 'left_front', 'front_left_front',
            'front', 'front_right_front', 'right_front', 'right_side_front',
            'right_side', 'right_side_back', 'right_back', 'back_right_back',
            'back', 'back_left_back', 'left_back', 'left_side_back'
        ]
    
    def get_stacked_state(self, current_sensors, lane, velocity, time_since_lane_change, distance):
        # Create geometric backoff stacking
        indices = [0, 1, 2, 4, 8, 16, 32]
        stacked_features = []
        
        for idx in indices:
            if idx < len(self.sensor_history):
                sensors = self.sensor_history[-(idx+1)]
                stacked_features.extend([
                    sensors.get(name, 1000) / 1000.0  # Normalize to [0,1]
                    for name in self.sensor_names
                ])
        
        # Add normalized additional state information
        additional_features = [
            (lane - 2) / 2.0,  # Normalize lane to [-1,1] (lane 0-4 -> -2 to +2)
            min(time_since_lane_change / 180.0, 1.0),  # Normalize cooldown to [0,1]
            np.tanh(velocity.x / 50.0),  # Soft normalize vx (tanh keeps outliers bounded)
            np.tanh(velocity.y / 5.0),   # Soft normalize vy 
            min(distance / 36000.0, 1.0)  # Normalize by max possible distance in 60s
        ]
        
        stacked_features.extend(additional_features)
        return np.array(stacked_features)
```

### 4.3 Integration with Existing Code

#### Modify `src/game/core.py`:
- Add DQN agent option alongside rule-based agent
- Implement state tracking for temporal stacking
- Add reward calculation and experience collection
- Integrate cooldown enforcement

#### Modify `src/game/agent.py`:
- Add DQN agent import and initialization
- Implement action translation (DQN output → lane change functions)
- Add lane position tracking

## 5. Training Process

### 5.1 Training Hyperparameters
```python
HYPERPARAMETERS = {
    'learning_rate': 0.0001,
    'batch_size': 32,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_frequency': 1000,
    'replay_buffer_size': 100000,
    'training_episodes': 10000,
    'max_episode_steps': 3600
}
```

### 5.2 Training Loop
1. **Episode Initialization:** Reset environment, clear sensor history
2. **State Processing:** Collect sensor data, create stacked state
3. **Action Selection:** Use epsilon-greedy with cooldown enforcement
4. **Environment Step:** Execute action via lane change functions
5. **Reward Calculation:** Compute reward based on outcome
6. **Experience Storage:** Store (state, action, reward, next_state) in replay buffer
7. **Network Training:** Sample batch and update Q-networks
8. **Target Network Update:** Periodically sync target network

### 5.3 Training Stages
- **Stage 1 (Episodes 1-3000):** High exploration (ε = 1.0 → 0.1)
- **Stage 2 (Episodes 3001-7000):** Balanced exploration/exploitation (ε = 0.1 → 0.05)
- **Stage 3 (Episodes 7001-10000):** Fine-tuning (ε = 0.05 → 0.01)

## 6. Safety Mechanisms

### 6.1 Cooldown Enforcement
- Track `last_lane_change_tick` in agent state
- Force `DO_NOTHING` action if cooldown active
- Minimum 180 ticks (3 seconds) between lane changes

### 6.2 Boundary Protection
```python
def is_valid_action(action, current_lane):
    if action == LANE_CHANGE_LEFT and current_lane == 0:
        return False  # Can't go left from leftmost lane
    if action == LANE_CHANGE_RIGHT and current_lane == 4:
        return False  # Can't go right from rightmost lane
    return True
```

### 6.3 Emergency Override
- Monitor sensor readings for imminent collision
- Override DQN decision if crash probability > 90%

## 7. Evaluation Metrics

### 7.1 Training Metrics
- Average episode reward
- Distance traveled per episode
- Crash rate
- Lane change frequency
- Q-value convergence

### 7.2 Performance Metrics
- **Primary:** Average distance traveled
- **Secondary:** Time to crash, successful overtakes
- **Efficiency:** Lane changes per distance traveled

## 8. Implementation Timeline

### Phase 1: Core Infrastructure (Days 1-3)
- [ ] Implement state processing with temporal stacking
- [ ] Create DQN network architecture
- [ ] Build experience replay buffer
- [ ] Integrate reward calculation system

### Phase 2: Agent Development (Days 4-6)
- [ ] Implement DQN agent class
- [ ] Add cooldown and safety mechanisms
- [ ] Create training environment wrapper
- [ ] Build action translation layer

### Phase 3: Training Setup (Days 7-8)
- [ ] Implement training loop
- [ ] Add logging and monitoring
- [ ] Create evaluation scripts
- [ ] Set up model checkpointing

### Phase 4: Training & Optimization (Days 9-14)
- [ ] Initial training runs
- [ ] Hyperparameter tuning
- [ ] Performance analysis
- [ ] Final model selection

## 9. Expected Challenges & Solutions

### Challenge 1: Sparse Rewards
**Solution:** Implement reward shaping with intermediate rewards for good positioning

### Challenge 2: Sample Efficiency
**Solution:** Use prioritized experience replay and careful curriculum learning

### Challenge 3: Action Space Timing
**Solution:** Implement proper action buffering to handle lane change duration

### Challenge 4: Overfitting to Training Scenarios
**Solution:** Randomize traffic patterns and use domain randomization

## 10. Success Criteria

### Minimum Viable Performance
- Average distance > 80% of rule-based agent
- Crash rate < 20%
- Successful lane change rate > 70%

### Target Performance
- Average distance > 120% of rule-based agent
- Crash rate < 10%
- Intelligent lane selection based on traffic patterns
- Smooth, efficient driving behavior

This plan provides a comprehensive roadmap for implementing an intelligent DQN-based lane switching agent that balances performance, safety, and learning efficiency.