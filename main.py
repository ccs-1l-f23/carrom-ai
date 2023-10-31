from carrom_env.carrom_env import CarromEnv

env = CarromEnv(render_mode=None)
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    print(agent, termination, reward)
    if agent == "player_0":
        # print("Player 0's turn")
        # # take three values from stdin
        # action = input().split()
        # action = [float(i) for i in action]
        action = env.action_space(agent).sample()
    else:
        # print("Player 1's turn")
        # # take three values from stdin
        # action = input().split()
        # action = [float(i) for i in action]
        action = env.action_space(agent).sample()

    if termination or truncation:
        action = None
    # else:
    #     # this is where you would insert your policy
    #     action = env.action_space(agent).sample()

    env.step(action)