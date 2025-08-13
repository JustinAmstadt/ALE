import gymnasium as gym
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper that adds additional rewards to the original environment reward.
    You can customize the reward function by modifying the calculate_custom_reward method.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.prev_score = 0
        self.prev_lives = None
        self.steps_since_score = 0
        self.max_y_position = 0  # Track highest position reached
        
    def reset(self, **kwargs):
        """Reset the environment and internal tracking variables."""
        obs, info = self.env.reset(**kwargs)
        self.prev_score = 0
        self.prev_lives = None
        self.steps_since_score = 0
        self.max_y_position = 0
        return obs, info
    
    def step(self, action):
        """Execute action and apply custom reward modifications."""
        #print(f"Action: {action}")
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        
        # Get the custom reward addition
        custom_reward = self.calculate_custom_reward(obs, original_reward, terminated, truncated, info)
        
        # Combine original and custom rewards
        total_reward = original_reward + custom_reward
        
        return obs, total_reward, terminated, truncated, info
    
    def calculate_custom_reward(self, obs, original_reward, terminated, truncated, info):
        """
        Calculate additional reward to add to the original reward.
        Override this method to implement your custom reward logic.
        
        Args:
            obs: Current observation
            original_reward: The reward from the original environment
            terminated: Whether episode is terminated
            truncated: Whether episode is truncated
            info: Info dict from environment
            
        Returns:
            float: Additional reward to add to original reward
        """
        custom_reward = 0.0
        
        # Example custom rewards - modify these based on your needs:
        
        # 1. Small penalty for each step to encourage efficiency
        custom_reward -= 0.01
        
        # 2. Bonus for score increases (if score info is available)
        if 'score' in info:
            score_increase = info['score'] - self.prev_score
            if score_increase > 0:
                custom_reward += score_increase * 0.1  # 10% bonus of score increase
                self.steps_since_score = 0
            else:
                self.steps_since_score += 1
            self.prev_score = info['score']
        
        # 3. Bonus for staying alive longer
        if 'lives' in info:
            if self.prev_lives is not None and info['lives'] < self.prev_lives:
                # Life lost penalty
                custom_reward -= 10.0
            self.prev_lives = info['lives']
        
        # 5. Penalty for being idle (no score progress)
        if self.steps_since_score > 100:  # No score for 100 steps
            custom_reward -= 0.1
        
        return custom_reward

