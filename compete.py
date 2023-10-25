import multiprocessing
import os
import random
import time
import gymnasium
import carrom_ai
import agents.geometric

NUMBER_OF_GAMES = 10
PLAYERS = ["random-coin", "random"]

env = gymnasium.make('carrom_ai/CarromGym-v0', render_mode='ansi')

agent_functions = {
    "random": lambda _: env.action_space.sample(),
    "center-of-mass": lambda observation: agents.geometric.com(observation),
    "random-coin": lambda observation: agents.geometric.random_coin(observation),
    "queen": lambda observation: agents.geometric.queen(observation),
}

def play_game(players):
    random.seed((os.getpid() * int(time.time())) % 123456789)
    observation, info = env.reset()
    moves = 0
    
    while True:
        action = agent_functions[players[observation["Player"] - 1]](observation)
        observation, reward, terminated, truncated, info = env.step(action)

        moves += 1
        if terminated:
            if (observation["Player"] == 1 and reward == 1) or (observation["Player"] == 2 and reward == -1):
                return (1, moves)
            else:
                return (2, moves)
            
def multiprocessed(players):
    p = multiprocessing.Pool()
    data = p.map(play_game, [players for _ in range(NUMBER_OF_GAMES)])
    p.close()
    p.join()

    white_wins = 0
    black_wins = 0
    moves = 0

    for i in data:
        if i[0] == 1:
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

