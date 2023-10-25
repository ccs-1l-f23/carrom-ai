import gymnasium
import carrom_ai

env = gymnasium.make('carrom_ai/CarromGym-v0', render_mode='human')

observation, info = env.reset()
while True:
    # if (observation["Player"] == 1):
    #     print("Player 1's turn")
    #     # take three values from stdin
    #     action = input().split()
    #     action = [float(i) for i in action]
    # else:
    #     action = env.action_space.sample()

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated:
        break