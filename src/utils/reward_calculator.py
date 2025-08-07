import numpy as np
from typing import Dict, Optional, Tuple
from ..mathematics.vector import Vector


class RewardCalculator:
    """
    Calculates rewards for DQN training based on game state and actions.
    Implements reward shaping to encourage efficient lane switching behavior.
    """
    
    def __init__(self):
        self.previous_distance = 0.0
        self.previous_velocity = 0.0
        self.lane_change_start_distance = {}  # Track distance when lane change started
        self.last_reward_components = {}  # For debugging
        
    def calculate_reward(self, 
                        current_distance: float,
                        velocity: Vector,
                        action: int,
                        crashed: bool,
                        sensors: Dict[str, Optional[float]],
                        lane: int,
                        time_since_lane_change: int,
                        is_performing_maneuver: bool,
                        attempted_invalid_action: bool = False) -> float:
        """
        Calculate reward based on current game state and action taken.
        
        Args:
            current_distance: Total distance traveled
            velocity: Current velocity vector
            action: Action taken (0=left, 1=right, 2=nothing)
            crashed: Whether the car crashed
            sensors: Current sensor readings
            lane: Current lane (0-4)
            time_since_lane_change: Ticks since last lane change
            is_performing_maneuver: Whether currently executing lane change
            attempted_invalid_action: Whether agent tried invalid action (e.g., wall hit)
            
        Returns:
            Reward value for this timestep
        """
        reward = 0.0
        
        # Terminal state penalties
        if crashed:
            reward += self._crash_penalty()
            self._reset_tracking()
            return reward
        
        if attempted_invalid_action:
            reward += self._wall_hit_penalty()
        
        # Primary rewards
        reward += self._distance_reward(current_distance)
        reward += self._speed_reward(velocity)
        reward += self._survival_reward()
        
        # Safety penalties
        reward += self._near_miss_penalty(sensors)
        
        # Lane change efficiency rewards/penalties
        if action in [0, 1]:  # Lane change actions
            reward += self._lane_change_initiation_reward(action, sensors, lane)
        
        # Strategic positioning rewards
        reward += self._positioning_reward(sensors, velocity)
        
        # Cooldown violation penalty
        if action in [0, 1] and time_since_lane_change < 180:  # 3 seconds at 60 FPS
            reward += self._cooldown_violation_penalty()
        
        # Track components for debugging
        self._update_tracking(current_distance, velocity.x)
        
        return reward
    
    def _crash_penalty(self) -> float:
        """Heavy penalty for crashing."""
        return -1000.0
    
    def _wall_hit_penalty(self) -> float:
        """Penalty for attempting to change lanes into walls."""
        return -500.0
    
    def _distance_reward(self, current_distance: float) -> float:
        """Reward based on distance progress."""
        distance_delta = current_distance - self.previous_distance
        return max(0, distance_delta * 1.0)  # 1.0 per unit distance
    
    def _speed_reward(self, velocity: Vector) -> float:
        """Bonus for maintaining high speed."""
        # Encourage forward speed, slight penalty for excessive lateral movement
        forward_bonus = velocity.x * 0.1
        lateral_penalty = abs(velocity.y) * 0.05
        return forward_bonus - lateral_penalty
    
    def _survival_reward(self) -> float:
        """Small reward for each tick survived."""
        return 0.5
    
    def _near_miss_penalty(self, sensors: Dict[str, Optional[float]]) -> float:
        """Penalty for getting too close to obstacles."""
        penalty = 0.0
        
        # Check critical sensors for close obstacles
        critical_sensors = ['front', 'left_side', 'right_side', 'back']
        
        for sensor_name in critical_sensors:
            distance = sensors.get(sensor_name)
            if distance is not None and distance < 200:  # Very close obstacle
                # Penalty inversely proportional to distance
                penalty -= 5.0 * (1.0 / max(distance, 50))  # Min distance to avoid division issues
        
        return penalty
    
    def _lane_change_initiation_reward(self, action: int, sensors: Dict[str, Optional[float]], lane: int) -> float:
        """
        Reward/penalty for initiating lane changes based on tactical advantage.
        """
        reward = 0.0
        
        # Get relevant sensor readings for the direction we're changing to
        if action == 0:  # Changing left
            front_sensor = sensors.get('left_front') or 1000
            side_sensor = sensors.get('left_side') or 1000
            back_sensor = sensors.get('left_back') or 1000
            target_lane = lane - 1
        else:  # Changing right
            front_sensor = sensors.get('right_front') or 1000
            side_sensor = sensors.get('right_side') or 1000
            back_sensor = sensors.get('right_back') or 1000
            target_lane = lane + 1
        
        # Check if lane change makes tactical sense
        current_front = sensors.get('front')
        if current_front is None:
            current_front = 1000
        
        # Positive scenarios for lane change
        if current_front < 500 and front_sensor > current_front * 1.5:
            # There's a car in front, and target lane has more space ahead
            reward += 20.0  # Gap utilization bonus
        
        if side_sensor < 300:
            # Target lane has immediate obstacle - bad timing
            reward -= 15.0  # Poor timing penalty
        
        if back_sensor < 200:
            # Fast car approaching from behind in target lane
            reward -= 10.0  # Dangerous move penalty
        
        return reward
    
    def _positioning_reward(self, sensors: Dict[str, Optional[float]], velocity: Vector) -> float:
        """
        Reward for good positioning relative to other cars.
        """
        reward = 0.0
        
        front_distance = sensors.get('front') or 1000
        back_distance = sensors.get('back') or 1000
        
        # Reward for maintaining good following distance
        if 300 < front_distance < 800:
            reward += 2.0  # Good following distance
        
        # Reward for having space behind (indicates passing slower cars)
        if back_distance > 600:
            reward += 1.0  # Successfully passed slower traffic
        
        # Penalty for being sandwiched between cars
        if front_distance < 300 and back_distance < 300:
            reward -= 3.0  # Trapped between cars
        
        return reward
    
    def _cooldown_violation_penalty(self) -> float:
        """Penalty for trying to change lanes during cooldown period."""
        return -100.0
    
    def _overtaking_bonus(self, sensors: Dict[str, Optional[float]]) -> float:
        """
        Bonus for successfully overtaking slower vehicles.
        This is called when a lane change completes successfully.
        """
        # Check if we have clear space ahead after lane change
        front_distance = sensors.get('front') or 1000
        if front_distance > 800:
            return 50.0  # Successful overtake
        return 0.0
    
    def _update_tracking(self, distance: float, velocity_x: float):
        """Update internal tracking variables."""
        self.previous_distance = distance
        self.previous_velocity = velocity_x
    
    def _reset_tracking(self):
        """Reset tracking variables for new episode."""
        self.previous_distance = 0.0
        self.previous_velocity = 0.0
        self.lane_change_start_distance.clear()
    
    def reset_episode(self):
        """Reset for new episode."""
        self._reset_tracking()
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get breakdown of last reward calculation for debugging."""
        return self.last_reward_components.copy()


class AdaptiveRewardCalculator(RewardCalculator):
    """
    Adaptive reward calculator that adjusts reward weights based on training progress.
    """
    
    def __init__(self, initial_exploration_weight: float = 1.0, 
                 final_exploration_weight: float = 0.1):
        super().__init__()
        self.initial_exploration_weight = initial_exploration_weight
        self.final_exploration_weight = final_exploration_weight
        self.episode_count = 0
        self.total_episodes = 10000  # Total training episodes
        
    def set_training_progress(self, episode: int, total_episodes: int):
        """Update training progress for adaptive rewards."""
        self.episode_count = episode
        self.total_episodes = total_episodes
    
    def _get_exploration_weight(self) -> float:
        """Get current exploration weight based on training progress."""
        progress = min(1.0, self.episode_count / self.total_episodes)
        return self.initial_exploration_weight + progress * (
            self.final_exploration_weight - self.initial_exploration_weight
        )
    
    def _lane_change_initiation_reward(self, action: int, sensors: Dict[str, Optional[float]], lane: int) -> float:
        """Override with adaptive exploration bonus."""
        base_reward = super()._lane_change_initiation_reward(action, sensors, lane)
        
        # Add exploration bonus that decreases over time
        exploration_weight = self._get_exploration_weight()
        exploration_bonus = 5.0 * exploration_weight
        
        return base_reward + exploration_bonus


class CurriculumRewardCalculator(RewardCalculator):
    """
    Curriculum-based reward calculator that starts with simpler objectives
    and gradually increases complexity.
    """
    
    def __init__(self):
        super().__init__()
        self.curriculum_stage = 0  # 0=basic, 1=intermediate, 2=advanced
        
    def set_curriculum_stage(self, stage: int):
        """Set current curriculum stage (0-2)."""
        self.curriculum_stage = max(0, min(2, stage))
    
    def calculate_reward(self, *args, **kwargs) -> float:
        """Calculate reward based on current curriculum stage."""
        base_reward = super().calculate_reward(*args, **kwargs)
        
        if self.curriculum_stage == 0:
            # Stage 0: Focus on survival and basic movement
            return base_reward
        elif self.curriculum_stage == 1:
            # Stage 1: Add lane change efficiency
            return base_reward  # Base implementation already includes this
        else:
            # Stage 2: Full strategic rewards
            return base_reward + self._advanced_strategic_rewards(*args, **kwargs)
    
    def _advanced_strategic_rewards(self, current_distance, velocity, action, crashed, 
                                   sensors, lane, time_since_lane_change, 
                                   is_performing_maneuver, attempted_invalid_action=False) -> float:
        """Advanced strategic rewards for final curriculum stage."""
        reward = 0.0
        
        # Reward for maintaining optimal lane position
        if lane == 2:  # Middle lane is often optimal
            reward += 1.0
        
        # Penalty for excessive lane changing
        if action in [0, 1] and time_since_lane_change < 600:  # Recent lane change
            reward -= 2.0  # Discourage frequent lane changes
        
        # Reward for speed consistency
        speed_change = abs(velocity.x - self.previous_velocity)
        if speed_change < 0.5:  # Smooth driving
            reward += 1.0
        
        return reward