# env_continuous.py
"""
Continuous allocation trading environment.
Action: target allocation vector (fractions summing to <= 1). Example: [0.3, 0.2, 0.0] for 3 assets.
Observation similar to discrete env: price windows + cash + holdings (in shares) or portfolio fractions.
On each step we rebalance portfolio towards the target allocation using current portfolio value.
Reward: change in portfolio value (PV_t - PV_{t-1}).
"""

from typing import Optional, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class ContinuousAllocationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        price_df: Union[pd.Series, pd.DataFrame],
        window_size: int = 10,
        initial_cash: float = 100000.0,
        transaction_cost_pct: float = 0.001,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self._rng = np.random.default_rng(seed)

        if isinstance(price_df, pd.Series):
            price_df = price_df.to_frame("PRICE")
        self.price_df = price_df.reset_index(drop=True)
        self.tickers = list(self.price_df.columns)
        self.n_assets = len(self.tickers)

        self.window_size = window_size
        self.initial_cash = initial_cash
        self.transaction_cost_pct = transaction_cost_pct

        # action: continuous allocation vector (n_assets,), each element in [0,1], interpretation: target fraction
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

        obs_low = -np.inf * np.ones(self.n_assets * self.window_size + 1 + self.n_assets)
        obs_high = np.inf * np.ones_like(obs_low)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed: Optional[int] = None, options: dict = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.current_step = self.window_size - 1
        self.cash = float(self.initial_cash)
        self.holdings = np.zeros(self.n_assets, dtype=float)  # shares (float to allow fractional mapping, but we round on buy)
        self.done = False
        self._update_portfolio_value()
        self.last_portfolio_value = self.portfolio_value
        return self._get_obs(), {}

    def _get_price(self, step_index: int):
        return self.price_df.iloc[step_index].to_numpy(dtype=float)

    def _get_obs(self):
        start = self.current_step - (self.window_size - 1)
        window = self.price_df.iloc[start : self.current_step + 1].to_numpy(dtype=float)
        norm = window / (window[0] + 1e-9)
        obs = np.concatenate([norm.flatten(), np.array([self.cash], dtype=float), self.holdings.astype(float)])
        return obs.astype(np.float32)

    def _update_portfolio_value(self):
        price = self._get_price(self.current_step)
        self.portfolio_value = self.cash + np.dot(self.holdings, price)

    def step(self, action):
        if self.done:
            raise RuntimeError("Step called on done environment; call reset().")
        action = np.clip(action.astype(float), 0.0, 1.0)
        # If the action does not sum to <=1 we normalize to sum <=1 (so remaining is cash)
        total = action.sum()
        if total > 1.0:
            action = action / total

        price = self._get_price(self.current_step)
        self._update_portfolio_value()
        pv = self.portfolio_value

        # compute target dollar allocation per asset
        target_dollars = action * pv
        current_dollars = self.holdings * price
        delta_dollars = target_dollars - current_dollars

        # For sells: free up cash, for buys: spend cash
        # Execute sells first to ensure we have cash (no margin allowed).
        # Sells
        for i in range(self.n_assets):
            if delta_dollars[i] < 0:
                # sell shares to reduce exposure
                shares_to_sell = int(np.floor((-delta_dollars[i]) / price[i]))
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price[i]
                    cost = proceeds * self.transaction_cost_pct
                    self.holdings[i] -= shares_to_sell
                    self.cash += proceeds - cost

        # Buys: allocate available cash proportionally to positive deltas
        buy_indices = np.where(delta_dollars > 0)[0]
        if buy_indices.size > 0:
            # compute money we want to spend, but limited by cash
            desired_spend = np.sum(delta_dollars[buy_indices])
            available_cash = self.cash
            spend_factor = min(1.0, available_cash / (desired_spend + 1e-9))
            for i in buy_indices:
                spend = delta_dollars[i] * spend_factor
                # convert to shares (integer shares)
                shares_to_buy = int(np.floor(spend / price[i]))
                if shares_to_buy <= 0:
                    continue
                cost = shares_to_buy * price[i]
                txn_cost = cost * self.transaction_cost_pct
                total_cost = cost + txn_cost
                if total_cost <= self.cash + 1e-8:
                    self.holdings[i] += shares_to_buy
                    self.cash -= total_cost

        prev_pv = pv
        self._update_portfolio_value()
        reward = self.portfolio_value - prev_pv

        # advance
        self.current_step += 1
        if self.current_step >= len(self.price_df):
            self.done = True

        info = {
            "step": self.current_step,
            "portfolio_value": float(self.portfolio_value),
            "cash": float(self.cash),
            "holdings": self.holdings.copy(),
        }

        obs = self._get_obs() if not self.done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), bool(self.done), False, info

    def render(self):
        price = self._get_price(min(self.current_step, len(self.price_df) - 1))
        print(
            f"Step: {self.current_step} | PV: {self.portfolio_value:.2f} | Cash: {self.cash:.2f} | Holdings: {self.holdings} | Prices: {price}"
        )

    def close(self):
        return
