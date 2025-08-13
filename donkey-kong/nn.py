import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class PPOAtariFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # Initialize with expected output feature size for the policy/value heads
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # usually 4 for Atari frame stacks

        # Same conv layers as your PPOAtariNet (without policy/value heads)
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the size of the output from conv layers (flattened)
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            conv_out = self.conv(sample_input / 255.0)  # normalize inside forward, here too for shape
            conv_out_size = conv_out.shape[1] * conv_out.shape[2] * conv_out.shape[3]

        # Fully connected layer to get features_dim size output
        self.fc = nn.Linear(conv_out_size, features_dim)

    def forward(self, observations):
        # Normalize inputs (important!)
        x = observations / 255.0
        x = self.conv(x)
        x = x.reshape(x.size(0), -1) # flatten
        x = F.relu(self.fc(x))
        return x
