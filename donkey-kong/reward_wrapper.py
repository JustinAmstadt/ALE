import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt


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
        
        # 3. Bonus for staying alive longer
        if 'lives' in info:
            if self.prev_lives is not None and info['lives'] < self.prev_lives:
                # Life lost penalty
                custom_reward -= 10.0
            self.prev_lives = info['lives']

        # 4. Reward for moving up
        averaged_position = self._get_averaged_mario_pixel_position(obs)
        y_position = averaged_position[0]
        if y_position < self.max_y_position:
            custom_reward += 1.0
        self.max_y_position = max(self.max_y_position, y_position)

        
        # 5. Penalty for being idle (no score progress)
        if self.steps_since_score > 100:  # No score for 100 steps
            custom_reward -= 0.1
        
        return custom_reward

    def _get_averaged_mario_pixel_position(self, obs):
        # mario red color: [200,  72,  72]
        mario_red_color = torch.tensor([200,  72,  72])
        obs_t = torch.tensor(obs)
        mario_red_mask = (obs_t == mario_red_color).all(dim=-1)

        positions = torch.nonzero(mario_red_mask, as_tuple=False)

        def remove_peach_belt(positions):
            remove = torch.tensor([[26, 66], [26, 67]])
            # Create a mask to keep only rows NOT in remove
            mask = ~((positions[:, None] == remove).all(dim=2).any(dim=1))
            return positions[mask]

        positions = remove_peach_belt(positions)

        averaged_position = positions.float().mean(dim=0)
        return averaged_position



