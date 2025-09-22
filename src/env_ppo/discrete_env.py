# env_discrete.py
import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnvDiscrete(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance=10000):
        super(TradingEnvDiscrete, self).__init__()

        # price data (assumed dataframe with 'Close' column)
        self.data = data.reset_index(drop=True)
        self.n_steps = len(data)

        # environment params
        self.initial_balance = initial_balance

        # spaces
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        return self._get_obs()

    def _get_obs(self):
        price = self.data.iloc[self.current_step]["Close"]
        return np.array([
            self.balance,
            self.shares_held,
            price,
            self.net_worth,
            self.max_net_worth
        ], dtype=np.float32)

    def step(self, action):
        price = self.data.iloc[self.current_step]["Close"]

        # execute action
        if action == 1:  # buy
            shares_bought = self.balance // price
            self.balance -= shares_bought * price
            self.shares_held += shares_bought
        elif action == 2:  # sell
            self.balance += self.shares_held * price
            self.shares_held = 0

        # update metrics
        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        reward = self.net_worth - self.initial_balance
        done = self.current_step >= self.n_steps - 1
        return self._get_obs(), reward, done, {}
