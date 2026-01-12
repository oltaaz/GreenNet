# resume_train.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from greennet.env import GreenNetEnv

def make_env():
    return Monitor(GreenNetEnv())

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = PPO.load("runs/ppo_greennet.zip", env=env)
    model.learn(total_timesteps=200_000, reset_num_timesteps=False)
    model.save("runs/ppo_greennet_continued")
    print("Saved to runs/ppo_greennet_continued.zip")