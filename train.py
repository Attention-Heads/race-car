from stable_baselines3 import PPO
from src.game.racecar_env import RaceCarEnv

env = RaceCarEnv(render_mode=False)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs")
model.learn(total_timesteps=200_000)
model.save("ppo_racecar")
