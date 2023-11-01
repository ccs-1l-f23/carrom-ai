import math
import random
import numpy as np

def get_coins(observation):
    return [coin for coin in observation["observation"][0] if coin[0] != 0 and coin[1] != 0]

def get_queen(observation):
    queen = observation["observation"][2][0]
    if queen[0] == 0 and queen[1] == 0:
        return []
    else:
        return [queen]
    
# angle [-45, 225] striker to point
def angle(strikerX, pointX, pointY):
    angle = math.degrees(math.atan2(pointY - 140, pointX - strikerX))
    # bound angle between -45 and 225
    if angle < -135:
        return angle + 360
    elif angle < -45:
        return -45
    else:
        return angle

def com(observation):
    # random position [170, 630], high force between [0.25, 0.75]
    action = [random.randint(170, 630), 0, random.uniform(0.25, 0.75)]
    
    coins = get_coins(observation)

    # calculate the center of mass of the coins
    center_of_mass = [0, 0]
    for coin in coins:
        center_of_mass[0] += coin[0]
        center_of_mass[1] += coin[1]
    center_of_mass[0] /= len(coins)
    center_of_mass[1] /= len(coins)

    # angle striker to center of mass
    action[1] = angle(action[0], center_of_mass[0], center_of_mass[1])

    # remap action[0] to [0, 1]
    action[0] = (action[0] - 170) / (630 - 170)
    
    return action

def random_coin(observation):
    # random position [170, 630], high force between [0.25, 0.75]
    action = [random.randint(170, 630), 0, random.uniform(0.25, 0.75)]

    coins = get_coins(observation)

    # pick a random coin
    coin = coins[random.randint(0, len(coins) - 1)]

    # angle striker to coin
    action[1] = angle(action[0], coin[0], coin[1])

    # remap action[0] to [0, 1]
    action[0] = (action[0] - 170) / (630 - 170)

    return action

def queen(observation):
    # random position [170, 630], high force between [0.25, 0.75]
    action = [random.randint(170, 630), 0, random.uniform(0.25, 0.75)]

    queen = get_queen(observation)

    # if queen is pocketed, shoot randomly
    if len(queen) == 0:
        return random_coin(observation)

    # angle striker to queen
    action[1] = angle(action[0], queen[0][0], queen[0][1])

    # remap action[0] to [0, 1]
    action[0] = (action[0] - 170) / (630 - 170)

    return action


# def closest_to_pocket(observation):
#     # random position [170, 630], high force between [0.25, 0.75]
#     action = [random.randint(170, 630), 0, random.uniform(0.25, 0.75)]

#     coins = get_coins(observation)

#     # calculate the closest coin to a pocket
#     closest_coin = coins[0]
#     closest_distance = 1000
#     for coin in coins:
#         for pocket in [(44.1, 43.1), (756.5, 43), (756.5, 756.5), (44, 756.5)]:
#             distance = math.sqrt((coin[0] - pocket[0]) ** 2 + (coin[1] - pocket[1]) ** 2)
#             if distance < closest_distance:
#                 closest_distance = distance
#                 closest_coin = coin

#     # angle striker to closest coin
#     action[1] = angle(action[0], closest_coin[0], closest_coin[1])

#     # remap action[0] to [0, 1]
#     action[0] = (action[0] - 170) / (630 - 170)

#     return action