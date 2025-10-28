from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import TradingEnv

# Initialize environment
env = DummyVecEnv([lambda: TradingEnv()])

# Initialize DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=50000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    target_update_interval=250,
)

# Train the agent
print("Training started...")
model.learn(total_timesteps=50000)
print("Training complete!")

# Save the model
model.save("models/dqn_trading_model.zip")
print("Model saved to /models/dqn_trading_model.zip")

# ==========================
#  EVALUATION PHASE
# ==========================
from stable_baselines3 import DQN
from env.trading_env import TradingEnv

print("\nEvaluating trained model...")

# Reload trained model and environment
env = TradingEnv()
model = DQN.load("models/dqn_trading_model.zip", env=env)

obs = env.reset()
total_reward = 0
steps = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    steps += 1
    env.render()
    if done:
        break

print(f"\nEvaluation complete! Total reward: {total_reward:.2f} over {steps} steps.")
print(f"Final portfolio value: {env.portfolio_value:.2f}")
