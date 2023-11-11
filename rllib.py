"""
Adapted from https://pettingzoo.farama.org/main/tutorials/rllib/holdem/

Author: Rohil Shah
"""

import os

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

from ray.rllib.env import PettingZooEnv
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from carrom_env.carrom_env import CarromEnv

torch, nn = try_import_torch()

if __name__ == "__main__":
    ray.init()

    alg_name = "PPO"

    def env_creator():
        env = CarromEnv()
        return env

    env_name = "carrom_env"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space["player_0"], act_space["player_0"], {}),
                "player_1": (None, obs_space["player_1"], act_space["player_1"], {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        # .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
        .framework(framework="torch")
    )

    tune.run(
        alg_name,
        name="PPO",
        stop={"timesteps_total": 10000000},
        checkpoint_freq=10,
        config=config.to_dict(),
    )