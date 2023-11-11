# Adapted from https://github.com/samiranrl/Carrom_rl/blob/master/2_player_server/start_server.py under GNU GPL v3
# by Rohil Shah
##########

from carrom_env.utils import *
from _thread import *
from math import pi
import time
import sys
from functools import reduce

# Hardcode the defaults
render_rate = 10
random.seed(0)
noise = 0

if noise == 1:
    noise1 = 0.005
    noise2 = 0.01
    noise3 = 2
else:
    noise1 = 0
    noise2 = 0
    noise3 = 0

##########################################################################

# play one step of carrom
# Input: state, player, action
# Output: next_state, reward
def play(action, state, agent, render_mode):
    if render_mode == "human":
        pygame.init()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
        pygame.display.set_caption("Carrom RL Simulation")
        background = BACKGROUND('use_layout.png', [-30, -30])
        draw_options = pymunk.pygame_util.DrawOptions(screen)

    space = pymunk.Space(threaded=True)
    
    score = state["Score"][agent]
    prevscore = state["Score"][agent]

    # pass through object // Dummy Object for handling collisions
    passthrough = pymunk.Segment(space.static_body, (0, 0), (0, 0), 5)
    passthrough.collision_type = 2
    passthrough.filter = pymunk.ShapeFilter(categories=0b1000)

    init_space(space)
    init_walls(space)
    pockets = init_pockets(space)

    init_coins(space, state["Black_Locations"], state["White_Locations"], state["Red_Location"], passthrough)

    striker = init_striker(space, passthrough, action, agent)    

    ticks = 0
    foul = False
    pocketed = []
    queen_pocketed = False

    while True:
        if render_mode == "human":
            if ticks % render_rate == 0:
                local_vis = True
                for event in pygame.event.get():
                    if event.type == QUIT:
                        sys.exit(0)
                    elif event.type == KEYDOWN and event.key == K_ESCAPE:
                        sys.exit(0)
            else:
                local_vis = False

        ticks += 1

        if render_mode == "human":
            if local_vis == True:
                screen.blit(background.image, background.rect)
                space.debug_draw(draw_options)

        space.step(1 / TIME_STEP)

        for pocket in pockets:
            if dist(pocket.body.position, striker[0].position) < POCKET_RADIUS - STRIKER_RADIUS + (STRIKER_RADIUS * 0.75):
                foul = True
                for shape in space.shapes:
                    if shape.color == STRIKER_COLOR:
                        print("player " + str(agent) + ": Foul, Striker pocketed")
                        space.remove(shape, shape.body)
                        break

        for pocket in pockets:
            for coin in space.shapes:
                if dist(pocket.body.position, coin.body.position) < POCKET_RADIUS - COIN_RADIUS + (COIN_RADIUS * 0.75):
                    if coin.color == BLACK_COIN_COLOR:
                        score += 1
                        pocketed.append((coin, coin.body))
                        space.remove(coin, coin.body)
                        if agent == 0:
                            foul = True
                            print("Foul, player 0 pocketed black")
                    if coin.color == WHITE_COIN_COLOR:
                        score += 1
                        pocketed.append((coin, coin.body))
                        space.remove(coin, coin.body)
                        if agent == 1:
                            foul = True
                            print("Foul, player 1 pocketed white")
                    if coin.color == RED_COIN_COLOR:
                        pocketed.append((coin, coin.body))
                        space.remove(coin, coin.body)
                        queen_pocketed = True

        if render_mode == "human":
            if local_vis == True:
                font = pygame.font.Font(None, 25)

                text = font.render("Player 0 Score: " +
                                    str(state["Score"][0]), 1, (220, 220, 220))
                screen.blit(text, (BOARD_SIZE / 3 + 67, 2, 0, 0))
                text = font.render("Player 1 Score: " +
                                    str(state["Score"][1]), 1, (220, 220, 220))
                screen.blit(text, (BOARD_SIZE / 3 + 67, 780, 0, 0))

                # First tick, draw an arrow representing action

                if ticks == 1:
                    force = action[2]
                    angle = action[1]
                    position = action[0]
                    draw_arrow(screen, position, angle, force, agent)

                pygame.display.flip()
                if ticks == 1:
                    time.sleep(1)

                clock.tick()

        # Do post processing and return the next State
        if is_ended(space) or ticks > TICKS_LIMIT:
            state_new = {"Black_Locations": [],
                         "White_Locations": [], "Red_Location": [], "Score": state["Score"],
                         "Queen": 0}
            next_agent = 1 if agent == 0 else 0

            for coin in space.shapes:
                if coin.color == BLACK_COIN_COLOR:
                    state_new["Black_Locations"].append(coin.body.position)
                if coin.color == WHITE_COIN_COLOR:
                    state_new["White_Locations"].append(coin.body.position)
                if coin.color == RED_COIN_COLOR:
                    state_new["Red_Location"].append(coin.body.position)

            if foul == True:
                print("Foul.. striker pocketed")
                for coin in pocketed:
                    if coin[0].color == BLACK_COIN_COLOR:
                        state_new["Black_Locations"].append(ret_pos(state_new))
                        score -= 1
                    if coin[0].color == WHITE_COIN_COLOR:
                        state_new["White_Locations"].append(ret_pos(state_new))
                        score -= 1
                    if coin[0].color == RED_COIN_COLOR:
                        state_new["Red_Location"].append(ret_pos(state_new))

            if (queen_pocketed == True and foul == False):
                if len(state_new["Black_Locations"]) + len(state_new["White_Locations"]) == 18:
                    print("The queen cannot be the first to be pocketed: player ", agent)
                    state_new["Red_Location"].append(ret_pos(state_new))
                else:
                    if score - prevscore > 0:
                        score += 3
                        print("Queen pocketed and covered in one shot")
                    else:
                        state_new["Queen"] = 1

            print("player " + str(agent) + ": Turn ended in ", ticks, " Ticks")
            state_new["Score"][agent] = score
            print("Coins Remaining: ", len(state_new["Black_Locations"]), "B ", len(state_new["White_Locations"]), "W ", len(state_new["Red_Location"]), "R")
            return state_new, next_agent, score - prevscore

def validate(action, state, agent):
    # print "Action Received", action

    position = action[0]
    angle = action[1]
    force = action[2]

    if (angle < -45 or angle > 225) and agent == 0:
        print("Invalid angle, taking random angle", end=' ')
        angle = random.randrange(-45, 225)
        print("which is ", angle)

    if (angle > 45 and angle < 135) and agent == 1:
        print("Invalid angle, taking random angle", end=' ')
        angle = random.randrange(135, 405)
        if angle > 360:
            angle = angle - 360
        print("which is ", angle)

    if position < 0 or position > 1:
        print("Invalid position, taking random position")
        position = random.random()

    if force < 0 or force > 1:
        print("Invalid force, taking random position")
        force = random.random()

    angle = angle + (random.choice([-1, 1]) * gauss(0, noise3))

    if angle < 0:
        angle = 360 + angle
    angle = angle / 180.0 * pi
    position = 170 + \
        (float(max(min(position + gauss(0, noise1), 1), 0)) * (460))
    force = MIN_FORCE + \
        float(max(min(force + gauss(0, noise2), 1), 0)) * MAX_FORCE

    tmp_state = state["White_Locations"] + state["Black_Locations"] + state["Red_Location"]

    check = 0
    fuse = 10

    if agent == 0:
        check = 0
        fuse = 10
        while check == 0 and fuse > 0:
            fuse -= 1
            check = 1

            for coin in tmp_state:
                if dist((position, 140), coin) < STRIKER_RADIUS + COIN_RADIUS:
                    check = 0
                    print("Position ", (position, 140), " clashing with a coin, taking random")
                    position = 170 + \
                        (float(
                            max(min(float(random.random()) + gauss(0, noise1), 1), 0)) * (460))

    if agent == 1:
        check = 0
        fuse = 10
        while check == 0 and fuse > 0:
            fuse -= 1
            check = 1

            for coin in tmp_state:
                if dist((position, BOARD_SIZE - 140), coin) < STRIKER_RADIUS + COIN_RADIUS:
                    check = 0
                    print("Position ", (position, 140), " clashing with a coin, taking random")
                    position = 170 + \
                        (float(
                            max(min(float(random.random()) + gauss(0, noise1), 1), 0)) * (460))

    action = (position, angle, force)
    return action


def step(action, agent, state, render_mode):
    winner = 0
    rewards = [0, 0]

    action = tuplise(action) if agent == 0 else transform_action(tuplise(action))
    next_state, next_agent, reward = play(validate(action, state, agent), state, agent, render_mode)
    
    if state["Queen"] == 1:
        if reward > 0:
            next_state["Score"][agent] += 3
            print("Sucessfully covered the queen")
        else:
            print("Could not cover the queen")
            next_state["Red_Location"].append(ret_pos(next_state))

    if next_state["Queen"] or reward > 0 and (len(next_state["Black_Locations"]) != 0 and len(next_state["White_Locations"]) != 0):
        if next_state["Queen"] == 1:
            print("Pocketed Queen, pocket any coin in this turn to cover it")
        
        # keep same agent for second turn
        return next_state, agent, rewards

    if len(next_state["White_Locations"]) == 0:
        if len(next_state["Red_Location"]) > 0:
            winner = 1
            # rewards = [-1, 1]
            rewards = [0, 1]
        else:
            winner = 0
            # rewards = [1, -1]
            rewards = [1, 0]

        print("Winner is player " + str(winner))
    elif len(next_state["Black_Locations"]) == 0:
        if len(next_state["Red_Location"]) > 0:
            winner = 0
            # rewards = [1, -1]
            rewards = [1, 0]
        else:
            winner = 1
            # rewards = [-1, 1]
            rewards = [0, 1]
        print("Winner is player " + str(winner))

    return next_state, next_agent, rewards

