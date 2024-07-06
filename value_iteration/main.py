import envs.envs
import random
import numpy as np
import gymnasium as gym

from value_iteration.frozen_lake_agent import FrozenLakeAgent
from value_iteration.walk_agent import WalkAgent


if __name__ == '__main__':
    random.seed(133)
    np.random.seed(133)

    env = gym.make('SlipperyWalkFive-v0', render_mode='rgb_array')
    _, __ = env.reset()

    agent = WalkAgent(
        env=env,
        n_states=7,
        policy=lambda s: {
            0: WalkAgent.LEFT,
            1: WalkAgent.LEFT,
            2: WalkAgent.LEFT,
            3: WalkAgent.LEFT,
            4: WalkAgent.LEFT,
            5: WalkAgent.LEFT,
            6: WalkAgent.LEFT,
        }[s]
    )

    env2 = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='rgb_array')
    env2.metadata["render_fps"] = 60
    _, __ = env2.reset()

    agent2 = FrozenLakeAgent(
        env=env2,
        n_cols=4,
        n_rows=4,
        policy=lambda s: {
            0: FrozenLakeAgent.RIGHT, 1: FrozenLakeAgent.LEFT, 2: FrozenLakeAgent.DOWN, 3: FrozenLakeAgent.UP,
            4: FrozenLakeAgent.LEFT, 5: FrozenLakeAgent.LEFT, 6: FrozenLakeAgent.RIGHT, 7: FrozenLakeAgent.LEFT,
            8: FrozenLakeAgent.UP, 9: FrozenLakeAgent.DOWN, 10: FrozenLakeAgent.UP, 11: FrozenLakeAgent.LEFT,
            12: FrozenLakeAgent.LEFT, 13: FrozenLakeAgent.RIGHT, 14: FrozenLakeAgent.DOWN, 15: FrozenLakeAgent.LEFT
        }[s]
    )
    agent2.policy_iteration()
