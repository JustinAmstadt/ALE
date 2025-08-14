import gymnasium as gym
from stable_baselines3 import PPO
from nn import PPOAtariFeatureExtractor
from reward_wrapper import CustomRewardWrapper
import ale_py

gym.register_envs(ale_py)

base_env = gym.make("ALE/DonkeyKong-v5")

env = CustomRewardWrapper(base_env)

policy_kwargs = dict(
    features_extractor_class=PPOAtariFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=512),
)

# model = PPO.load("ppo_donkey_kong.pth", env=env, policy_kwargs=policy_kwargs, verbose=1)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

try:
    model.learn(total_timesteps=1_000_000)
except Exception as e:
    print(f"Error: {e}")
finally:
    model.save("ppo_donkey_kong.pth")
    print("Model saved successfully!")