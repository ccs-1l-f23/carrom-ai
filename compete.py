import multiprocessing
import os
import random
import time
from carrom_env.carrom_env import CarromEnv
import agents.geometric

NUMBER_OF_GAMES = 10

env = CarromEnv(render_mode=None)

agent_functions = {
    "random": lambda _, agent: env.action_space(agent).sample(),
    "center-of-mass": lambda observation, agent: agents.geometric.com(observation),
    "random-coin": lambda observation, agent: agents.geometric.random_coin(observation),
    "queen": lambda observation, agent: agents.geometric.queen(observation),
}

def play_game(players):
    random.seed((os.getpid() * int(time.time())) % 123456789)

    env.reset()
    winner = -1
    moves = 0

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
            if (agent == "0" and reward == 1) or (agent == "1" and reward == 0):
                winner = 0
            else:
                winner = 1
        else:
            if agent == "0":
                action = agent_functions[players[0]](observation, agent)
            else:
                action = agent_functions[players[1]](observation, agent)

        moves += 1
        env.step(action)
    
    return (winner, moves)
            
def multiprocessed(players):
    p = multiprocessing.Pool()
    data = p.map(play_game, [players for _ in range(NUMBER_OF_GAMES)])
    p.close()
    p.join()

    white_wins = 0
    black_wins = 0
    moves = 0

    for i in data:
        if i[0] == 0:
            white_wins += 1
        else:
            black_wins += 1
        moves += i[1]

    return (white_wins, black_wins, moves / NUMBER_OF_GAMES)

if __name__ == '__main__':
    for i in agent_functions:
        for j in agent_functions:
            # write to results.txt
            with open("results.txt", "a") as f:
                f.write(i + " vs " + j + "\n")
                f.write(str(multiprocessed([i, j])) + "\n")

    # print(multiprocessed(["queen", "queen"]))

