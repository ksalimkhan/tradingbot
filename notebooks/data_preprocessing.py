import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os

# ===============================
# 1. DOWNLOAD AAPL DATA
# ===============================
print("Downloading AAPL data from Yahoo Finance...")
df = yf.download("AAPL", start="2014-01-01", end="2024-12-31")
df.reset_index(inplace=True)

# ===============================
# 2. FEATURE ENGINEERING
# ===============================

# Raw features
df["High_Low"] = df["High"] - df["Low"]
df["Close_Open"] = df["Close"] - df["Open"]

# Moving Averages
df["MA10"] = df["Close"].rolling(window=10).mean()
df["MA30"] = df["Close"].rolling(window=30).mean()

# RSI (Relative Strength Index)
df["RSI14"] = ta.momentum.RSIIndicator(df["Close"].squeeze(), window=14).rsi()

# Momentum (5-day)
df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1

# Volatility (20-day rolling std of returns)
df["Returns"] = df["Close"].pct_change()
df["Volatility20"] = df["Returns"].rolling(window=20).std()

# Bollinger Band Width
# Bollinger Band Width (fixed for 1D output)
bb = ta.volatility.BollingerBands(df["Close"].squeeze(), window=20, window_dev=2)
upper = bb.bollinger_hband().squeeze()
lower = bb.bollinger_lband().squeeze()
df["BB_Width"] = (upper - lower) / df["MA10"]


# Position Flag (placeholder, initially 0)
df["Position_Flag"] = 0

# Recent Return (t−1 to t)
df["Recent_Return"] = df["Returns"]

# Drop NaN values from rolling calculations
df.dropna(inplace=True)

# ===============================
# 3. NORMALIZE NUMERIC FEATURES
# ===============================
cols_to_normalize = [
    "Volume", "High_Low", "Close_Open",
    "MA10", "MA30", "RSI14", "Momentum_5",
    "Volatility20", "BB_Width", "Recent_Return"
]

df[cols_to_normalize] = (df[cols_to_normalize] - df[cols_to_normalize].mean()) / df[cols_to_normalize].std()

# ===============================
# 4. SAVE DATASET
# ===============================
import os

# Dynamically resolve path relative to project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(base_dir, "data", "AAPL_DQN_features.csv")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")
print(df.head())
