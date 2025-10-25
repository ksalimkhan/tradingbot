import os
import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, csv_path=None, initial_balance=10000):
        super(TradingEnv, self).__init__()

        # Resolve CSV path dynamically
        if csv_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(base_dir, "data", "AAPL_DQN_features.csv")

        # Load dataset
        self.df = pd.read_csv(csv_path)

        # --- Clean non-numeric junk (like 'AAPL' strings or duplicated headers) ---
        # Remove any rows that are duplicated headers accidentally written
        self.df = self.df[self.df["Close"] != "Close"]

        # Force numeric conversion for Close column
        self.df["Close"] = pd.to_numeric(self.df["Close"], errors="coerce")

        # Drop any rows that still have NaN in Close
        self.df.dropna(subset=["Close"], inplace=True)

        # Convert all numeric columns
        self.df = self.df.apply(pd.to_numeric, errors="ignore")

        # Store Close prices for trading
        self.close_prices = self.df["Close"].astype(float).values

        # Keep only numeric features for state representation
        numeric_df = self.df.select_dtypes(include=[np.number])
        self.features = numeric_df.values.astype(np.float32)
        self.n_features = self.features.shape[1]

        print(f"✅ Cleaned dataset with {len(self.df)} rows and {self.n_features} features.")
        print(f"Close price range: {self.df['Close'].min():.2f} → {self.df['Close'].max():.2f}")

        # Define action/observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

        # Initialize
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.portfolio_value = self.balance
        return self.features[self.current_step]

    def step(self, action):
        prev_value = self.portfolio_value
        price = self.close_prices[self.current_step]
        next_price = self.close_prices[min(self.current_step + 1, len(self.close_prices) - 1)]
        # Debug: check for invalid or negative prices
        if price <= 0 or next_price <= 0 or np.isnan(price) or np.isnan(next_price):
            print(f"⚠️ Invalid price at step {self.current_step}: price={price}, next_price={next_price}")  

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = self.balance / price
            self.balance = 0
        elif action == 2 and self.position > 0:  # Sell
            self.balance = self.position * price
            self.position = 0

        # Update portfolio value
        self.portfolio_value = self.balance + self.position * next_price
        reward = self.portfolio_value - prev_value

        # Step forward
        self.current_step += 1
        done = self.current_step >= len(self.features) - 1
        next_state = self.features[self.current_step]

        return next_state, reward, done, {}

    def render(self, mode="human"):
        print(
            f"Step {self.current_step} | Balance: {self.balance:.2f} | "
            f"Position: {self.position:.4f} | Portfolio: {self.portfolio_value:.2f}"
        )
