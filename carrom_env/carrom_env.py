import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from carrom_env.utils import *
from carrom_env.board import *
from gymnasium import spaces

class CarromEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "carrom_env_v0",
        "is_parallelizable": True
    }

    def __init__(self, render_mode=None):
        # two agents
        self.possible_agents = ["player_" + str(i) for i in range(2)]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    # Black/White/Red: [0, 800] ^ 9 or ^ 1
    # Score: [0, 12] ^ 2
    # Queen: 0 or 1
    def observation_space(self, agent):
        return spaces.Dict({
            "Black_Locations": spaces.Box(low = 0, high = 800, shape = (9, 2), dtype=float),
            "White_Locations": spaces.Box(low = 0, high = 800, shape = (9, 2), dtype=float),
            "Red_Location": spaces.Box(low = 0, high = 800, shape = (1, 2), dtype=float),
            "Score": spaces.Box(low = 0, high = 12, shape = (2,), dtype=int),
            "Queen": spaces.Discrete(2)
        })

    # Actions are 3-tuples: (position, angle, force)
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(
            low = np.array([0.0, -45.0, 0.0]),
            high = np.array([1.0, 225.0, 1.0]),
            shape = (3,),
            dtype = float
        )
    
    def observe(self, agent):
        if agent == "player_0":
            return self._state
        else:
            return transform_state(self._state)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        # necessary for last() (in addition to observe())
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self._state = INITIAL_STATE

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()


    def step(self, action):
        # steps over terminated agent (accepts None as action)
        if (self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return

        agent = 0 if self.agent_selection == "player_0" else 1

        next_state, next_agent, rewards = step(action, agent, self._state, self.render_mode)
        self._state = next_state
        if agent != next_agent:
            self.agent_selection = self._agent_selector.next()

        if rewards != [0, 0]:
            self.rewards = {agent: reward for agent, reward in zip(self.agents, rewards)}
            self.terminations = {agent: True for agent in self.agents}
        
        self._accumulate_rewards()