import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from carrom_env.utils import *
from carrom_env.board import *
from gymnasium import spaces

class CarromEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "carrom_env_v0",
        "is_parallelizable": True
    }

    def __init__(self, render_mode=None):
        # two agents
        # self.possible_agents = ["player_" + str(i) for i in range(2)]
        self.possible_agents = ["0", "1"]
        
        # Black: [0, 800] ^ 9
        # White: [0, 800] ^ 9
        # Red:   [0, 800] ^ 1
        # Padded with (0, 0)
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low = 0.0, high = 800.0, shape = (3, 9, 2), dtype=np.float32)
            })
            for agent in self.possible_agents
        }

        # Actions are 3-tuples: (position, angle, force)
        self.action_spaces = {
            agent: spaces.Box(
                low = np.array([0.0, -45.0, 0.0], dtype=np.float32),
                high = np.array([1.0, 225.0, 1.0], dtype=np.float32),
                shape = (3,),
                dtype = np.float32
            )
            for agent in self.possible_agents
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        state = copy.deepcopy(self._state) if agent == "0" else transform_state(self._state)

        for i in ["Black_Locations", "White_Locations", "Red_Location"]:
            state[i] += [(0, 0) for _ in range(9 - len(state[i]))]
        
        return OrderedDict({
            "observation": np.array([
                state["White_Locations"] if agent == "0" else state["Black_Locations"],
                state["Black_Locations"] if agent == "0" else state["White_Locations"],
                state["Red_Location"],
            ]).astype(np.float32)
        })


    def reset(self, seed=None, options=None):
        random.seed(seed)
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

        agent = int(self.agent_selection)
        self._cumulative_rewards[self.agent_selection] = 0

        # next_state, next_agent, rewards = step(action, agent, self._state, self.render_mode)
        next_state, next_agent, rewards, terminated = step(action, agent, self._state, self.render_mode)
        self._state = next_state
        if agent != next_agent:
            self.agent_selection = self._agent_selector.next()
        
        self.rewards = {agent: reward for agent, reward in zip(self.agents, rewards)}
        self.terminations = {agent: terminated for agent in self.agents}

        # if rewards != [0, 0]:
        #     self.rewards = {agent: reward for agent, reward in zip(self.agents, rewards)}
        #     self.terminations = {agent: False for agent in self.agents}
        
        self._accumulate_rewards()