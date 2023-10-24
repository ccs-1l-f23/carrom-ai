import gymnasium as gym
from gymnasium import spaces
import numpy as np
from envs.utils import *
from envs.board import *

class carrom_env(gym.Env):
    # human: render the environment to the screen
    # ansi: print the observation to the terminal
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None):
        # Black/White/Red: [0, 800] ^ 9 or ^ 1
        # Score: [0, 12] ^ 2
        # Player: 1 or 2
        # Queen: 0 or 1
        self.observation_space = spaces.Dict({
            "Black_Locations": spaces.Box(low = 0, high = 800, shape = (9, 2), dtype=float),
            "White_Locations": spaces.Box(low = 0, high = 800, shape = (9, 2), dtype=float),
            "Red_Location": spaces.Box(low = 0, high = 800, shape = (1, 2), dtype=float),
            "Score": spaces.Box(low = 0, high = 12, shape = (2,), dtype=int),
            "Player": spaces.Discrete(1),
            "Queen": spaces.Discrete(1)
        })

        # Actions are 3-tuples: (position, angle, force)
        self.action_space = spaces.Box(
            low = np.array([0.0, -45.0, 0.0]),
            high = np.array([1.0, 225.0, 1.0]),
            shape = (3,),
            dtype = float
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return self._state
    
    def _get_info(self):
        return { "debug-info": "hello world" }

    def reset(self):
        self._state = INITIAL_STATE

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        new_state, winner = step(action, self._state, self.render_mode)
        self._state = new_state

        observation = self._get_obs()
        if winner == self._state["Player"]:
            reward = 1
        elif winner == 1 or winner == 2:
            reward = -1
        else:
            reward = 0
        terminated = winner != 0
        info = self._get_info()

        return observation, reward, terminated, False, info