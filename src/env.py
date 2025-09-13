# src/env.py# src/env.py
import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    """
    Simple trading environment (Gym API).
    Actions: 0 = Hold, 1 = Buy, 2 = Sell
    Observation: current Close price (can be extended to window of features)
    Reward: change in net worth since previous step
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data_path, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = pd.read_csv(data_path, parse_dates=["Date"])
        self.df = self.df.sort_values("Date").reset_index(drop=True)

        # discrete actions: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)

        # simple observation: scalar price (float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.initial_balance = initial_balance
        self._reset_internal_state()

        # logs
        self.actions_log = []
        self.net_worth_log = []
        self.date_log = []
        self._seed = None

    def _reset_internal_state(self):
        self.current_step = 0
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.net_worth = float(self.initial_balance)
        self._prev_net_worth = float(self.initial_balance)

    def seed(self, seed=None):
        """
        Set random seed for reproducibility. Shimmy/SB3 may call env.seed(...).
        Returns list [seed].
        """
        self._seed = seed
        import random as _random
        _random.seed(seed)
        np.random.seed(seed if seed is not None else None)
        return [seed]

    def reset(self, seed=None):
        """
        Reset environment and optionally set seed (Gym compatibility).
        """
        if seed is not None:
            self.seed(seed)

        self._reset_internal_state()
        self.actions_log = []
        self.net_worth_log = [self.net_worth]
        self.date_log = [self.df.loc[self.current_step, "Date"]]
        obs = np.array([self.df.loc[self.current_step, "Close"]], dtype=np.float32)
        return obs

    def step(self, action):
        """Apply action and return (obs, reward, done, info)."""
        price = float(self.df.loc[self.current_step, "Close"])

        if action == 1:  # Buy one share
            if self.balance >= price:
                self.shares_held += 1.0
                self.balance -= price
        elif action == 2:  # Sell one share (if any)
            if self.shares_held >= 1.0:
                self.shares_held -= 1.0
                self.balance += price

        self.current_step += 1

        done = False
        if self.current_step >= len(self.df) - 1:
            done = True
            next_price = float(self.df.loc[len(self.df) - 1, "Close"])
        else:
            next_price = float(self.df.loc[self.current_step, "Close"])

        self.net_worth = self.balance + self.shares_held * next_price
        reward = self.net_worth - self._prev_net_worth
        self._prev_net_worth = self.net_worth

        self.actions_log.append(int(action))
        self.net_worth_log.append(self.net_worth)
        self.date_log.append(self.df.loc[self.current_step, "Date"])

        obs = np.array([next_price], dtype=np.float32)

        info = {
            "step": int(self.current_step),
            "balance": float(self.balance),
            "shares_held": float(self.shares_held),
            "net_worth": float(self.net_worth),
        }

        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Date: {self.df.loc[self.current_step, 'Date']}, Net Worth: {self.net_worth:.2f}")

    def close(self):
        pass
