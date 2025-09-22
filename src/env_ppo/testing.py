# ppo_test.ipynb
import pandas as pd
import gym
from stable_baselines3 import PPO
from env_discrete import TradingEnvDiscrete
from env_continuous import TradingEnvContinuous

# load sample data
data = pd.read_csv("AAPL.csv")  # must have 'Close' column

# Choose environment type
env = TradingEnvDiscrete(data)   # OR TradingEnvContinuous(data)

# Wrap in Gym
env = gym.wrappers.TimeLimit(env, max_episode_steps=500)

# Train PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Test agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break
