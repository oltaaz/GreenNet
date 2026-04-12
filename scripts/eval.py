from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from greennet.env import GreenNetEnv


def make_env():
    return Monitor(GreenNetEnv())


def main() -> None:
    env = DummyVecEnv([make_env])

    model = PPO.load("runs/ppo_greennet.zip", env=env)

    obs = env.reset()
    ep_reward = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward[0])
        if done[0]:
            break

    print("Eval episode reward:", ep_reward)


if __name__ == "__main__":
    main()
