# 🧠 Deep Q-Learning Stock Trading Bot
### Reinforcement Learning Project — DQN Agent for Apple (AAPL)

This project implements a **Deep Q-Network (DQN)** agent that learns to trade stock data (AAPL) using reinforcement learning.  
The model uses a custom **Gym environment**, preprocessed historical data, and reward signals based on portfolio performance.

---

## 🚀 Project Overview
The system uses:
- **Data Source:** AAPL historical data (2014–2024) from Yahoo Finance  
- **Model Type:** Deep Q-Network (DQN)  
- **Frameworks:** PyTorch + Stable-Baselines3 + Gymnasium  
- **Goal:** Learn buy/sell/hold decisions to maximize portfolio value  

---

## 🧩 Project Structure
tradingbot/
├── data/ # Stores processed CSV data
│ └── AAPL_DQN_features.csv
├── env/
│ └── trading_env.py # Custom OpenAI Gym environment
├── models/ # Folder for trained DQN models
├── notebooks/
│ └── data_preprocessing.py # Cleans and prepares the dataset
├── main_train.py # Main training script
├── requirements.txt # Dependencies list
├── .gitignore
└── README.md # Project documentation

---

## 🖥️ Installation Guide

## 1️⃣ Clone the repository
git clone https://github.com/ksalimkhan/tradingbot.git
cd tradingbot

## 2️⃣ Install dependencies
pip install -r requirements.txt

If you see warnings about LF ↔ CRLF, they can be ignored.
All dependencies will install automatically (Stable-Baselines3, Gymnasium, Pandas, PyTorch, etc.).

## 3️⃣ Verify installation

Run this to confirm everything is installed properly:

python -c "import torch, gym, stable_baselines3; print('✅ Environment ready!')"

## 📊 Data Preparation

The dataset is automatically downloaded and cleaned through the preprocessing script.

Run preprocessing:
python notebooks/data_preprocessing.py

This will:

Download Apple (AAPL) stock data from 2014–2024

Compute technical indicators (RSI, MA10, MA30, Bollinger Bands, etc.)

Clean missing values

Save the final file to:

data/AAPL_DQN_features.csv

## 🤖 Model Training
Start the training:
python main_train.py

This script:

Loads AAPL_DQN_features.csv

Initializes the custom TradingEnv environment

Trains a DQN model for 500,000 timesteps

Saves results under the models/ directory

## 🧠 Evaluation Output Example

After training, you’ll see:

✅ Evaluation complete! Total reward: 137789.94 over 2737 steps.
Final portfolio value: 147789.94

If the portfolio is negative, check data scaling (Close price normalization).
If positive, the DQN is learning effectively.

## 🧾 Notes & Tips

The project uses a custom Gym environment (TradingEnv) to simulate trading actions:

Actions:

0: Hold

1: Buy

2: Sell

Reward: Change in portfolio value each step

Training time depends on your CPU/GPU performance.

To check if your GPU is being used:

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

You can adjust total training steps in main_train.py:

model.learn(total_timesteps=500000)
