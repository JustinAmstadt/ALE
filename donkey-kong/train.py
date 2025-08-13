import gymnasium as gym
from stable_baselines3 import PPO
from nn import PPOAtariFeatureExtractor
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/DonkeyKong-v5")

policy_kwargs = dict(
    features_extractor_class=PPOAtariFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=100000)

model.save("ppo_donkey_kong.pth")