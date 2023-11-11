"""
Implement self-play using this as a guide: https://github.com/ray-project/ray/blob/master/rllib/examples/self_play_with_open_spiel.py

Author: Rohil Shah
"""

import os

import numpy as np

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.tune import CLIReporter, register_env

# The new RLModule / Learner API
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule

from carrom_env.carrom_env import CarromEnv

class SelfPlayCallback(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.
        main_rew = result["hist_stats"].pop("policy_main_reward")
        opponent_rew = list(result["hist_stats"].values())[0]
        assert len(main_rew) == len(opponent_rew)
        won = 0
        for r_main, r_opponent in zip(main_rew, opponent_rew):
            if r_main > r_opponent:
                won += 1
        win_rate = won / len(main_rew)
        result["win_rate"] = win_rate
        print(f"Iter={algorithm.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > 0.95:
            self.current_opponent += 1
            new_pol_id = f"main_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # agent_id = [0|1] -> policy depends on episode ID
                # This way, we make sure that both policies sometimes play
                # (start player) and sometimes agent1 (player to move 2nd).
                return (
                    "main"
                    if episode.episode_id % 2 == agent_id
                    else "main_v{}".format(
                        np.random.choice(list(range(1, self.current_opponent + 1)))
                    )
                )

            main_policy = algorithm.get_policy("main")
            if algorithm.config._enable_new_api_stack:
                new_policy = algorithm.add_policy(
                    policy_id=new_pol_id,
                    policy_cls=type(main_policy),
                    policy_mapping_fn=policy_mapping_fn,
                    module_spec=SingleAgentRLModuleSpec.from_module(main_policy.model),
                )
            else:
                new_policy = algorithm.add_policy(
                    policy_id=new_pol_id,
                    policy_cls=type(main_policy),
                    policy_mapping_fn=policy_mapping_fn,
                )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            main_state = main_policy.get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            algorithm.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        # +2 = main + random
        result["league_size"] = self.current_opponent + 2


if __name__ == "__main__":
    ray.init(include_dashboard=False)

    register_env("carrom_env", lambda _: PettingZooEnv(CarromEnv()))

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # agent_id = [0|1] -> policy depends on episode ID
        # This way, we make sure that both policies sometimes play agent0
        # (start player) and sometimes agent1 (player to move 2nd).
        return "main" if episode.episode_id % 2 == agent_id else "random"
    
    config = (
        PPOConfig()
        .environment("carrom_env")
        .framework("torch")
        .callbacks(SelfPlayCallback)
        .rollouts(num_envs_per_worker=5, num_rollout_workers=2)
        .training(num_sgd_iter=20, model={"fcnet_hiddens": [512, 512]})
        .multi_agent(
            # Initial policy map: Random and PPO. This will be expanded
            # to more policy snapshots taken from "main" against which "main"
            # will then play (instead of "random"). This is done in the
            # custom callback defined above (`SelfPlayCallback`).
            policies={
                # Our main policy, we'd like to optimize.
                "main": PolicySpec(),
                # An initial random opponent to play against.
                "random": PolicySpec(policy_class=RandomPolicy),
            },
            # Assign agent 0 and 1 randomly to the "main" policy or
            # to the opponent ("random" at first). Make sure (via episode_id)
            # that "main" always plays against "random" (and not against
            # another "main").
            policy_mapping_fn=policy_mapping_fn,
            # Always just train the "main" policy.
            policies_to_train=["main"],
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "main": SingleAgentRLModuleSpec(),
                    "random": SingleAgentRLModuleSpec(module_class=RandomRLModule),
                }
            ),
        )
    )
    
    stop = {
        "timesteps_total": 10000000,
        "training_iteration": 200,
    }

    # Train the "main" policy to play really well using self-play.
    results = None
    from_checkpoint = None
    if not from_checkpoint:
        create_checkpoints = not bool(os.environ.get("RLLIB_ENABLE_RL_MODULE", False))
        results = tune.Tuner(
            "PPO",
            param_space=config,
            run_config=air.RunConfig(
                stop=stop,
                verbose=2,
                failure_config=air.FailureConfig(fail_fast="raise"),
                progress_reporter=CLIReporter(
                    metric_columns={
                        "training_iteration": "iter",
                        "time_total_s": "time_total_s",
                        "timesteps_total": "ts",
                        "episodes_this_iter": "train_episodes",
                        "policy_reward_mean/main": "reward",
                        "win_rate": "win_rate",
                        "league_size": "league_size",
                    },
                    sort_by_metric=True,
                ),
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_at_end=create_checkpoints,
                    checkpoint_frequency=10 if create_checkpoints else 0,
                ),
            ),
        ).fit()
    
    ray.shutdown()