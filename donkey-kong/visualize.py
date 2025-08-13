import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
import ale_py

gym.register_envs(ale_py)

def make_env():
    env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
    env = Monitor(env)
    return env

# Create vectorized environment matching the training setup
env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

model = PPO.load("ppo_donkey_kong.pth", env=env)

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    
    if dones[0]:
        obs = env.reset()
        break