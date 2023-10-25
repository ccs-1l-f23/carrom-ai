import math
import random
from carrom_ai.envs.utils import transform_state

def get_coins(observation):
    if observation["Player"] == 1:
        return observation["White_Locations"]
    else:
        return transform_state(observation)["Black_Locations"]
    
def get_queen(observation):
    if observation["Player"] == 1:
        return observation["Red_Location"]
    else:
        return transform_state(observation)["Red_Location"]

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
    print(coin, action[1])

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


# closest to pocket coin

