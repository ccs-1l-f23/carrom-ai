import gymnasium as gym
import numpy as np
env = gym.make('Pendulum-v1', g=9.81, render_mode="human")
# env = gym.make('Pendulum-v1', g=9.81)

observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()