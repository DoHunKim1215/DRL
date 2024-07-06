import envs.envs
import random
import numpy as np
import gymnasium as gym

from value_iteration.frozen_lake_agent import FrozenLakeAgent
from value_iteration.walk_agent import WalkAgent


if __name__ == '__main__':
    random.seed(133)
    np.random.seed(133)

    # Create Slippery Walk Five Environment and Agent
    # States: Hole - Frozen - Frozen - Start - Frozen - Frozen - Goal
    # Action: Left or Right
    # Transition: 50% (go to intended direction), 33.33% (stay in same state), 16.66% (go to reverse direction)
    SWF_env = gym.make('SlipperyWalkFive-v0', render_mode='rgb_array')
    _, __ = SWF_env.reset()

    SWF_agent = WalkAgent(
        env=SWF_env,
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

    # Get the info of agent which has dummy policy
    print("Current Agent Info : ")
    print(SWF_agent)
    print("###########################################################")

    # Policy Evaluation
    state_value = SWF_agent.evaluate_policy()
    print("###########################################################")

    # Let's improve the current policy
    SWF_agent.get_improved_policy(state_value=state_value)
    print("###########################################################")

    # Policy Iteration
    SWF_agent.policy_iteration()
    print("###########################################################")

    # Value Iteration
    SWF_agent.value_iteration()
    print("###########################################################")

    # Create Frozen Lake 4x4 Environment and Agent
    FL_env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], is_slippery=True, render_mode='rgb_array')
    _, __ = FL_env.reset()

    FL_agent = FrozenLakeAgent(
        env=FL_env,
        n_rows=4,
        n_cols=4,
        policy=lambda s: {
            0: FrozenLakeAgent.RIGHT, 1: FrozenLakeAgent.LEFT, 2: FrozenLakeAgent.DOWN, 3: FrozenLakeAgent.UP,
            4: FrozenLakeAgent.LEFT, 5: FrozenLakeAgent.LEFT, 6: FrozenLakeAgent.RIGHT, 7: FrozenLakeAgent.LEFT,
            8: FrozenLakeAgent.UP, 9: FrozenLakeAgent.DOWN, 10: FrozenLakeAgent.UP, 11: FrozenLakeAgent.LEFT,
            12: FrozenLakeAgent.LEFT, 13: FrozenLakeAgent.RIGHT, 14: FrozenLakeAgent.DOWN, 15: FrozenLakeAgent.LEFT
        }[s]
    )

    # Get the info of agent which has dummy policy
    print("Current Agent Info : ")
    print(FL_agent)
    print("###########################################################")

    # Policy Evaluation
    state_value = FL_agent.evaluate_policy()
    print("###########################################################")

    # Let's improve the current policy
    FL_agent.get_improved_policy(state_value=state_value)
    print("###########################################################")

    # Policy Iteration
    FL_agent.policy_iteration()
    print("###########################################################")

    # Value Iteration
    FL_agent.value_iteration()
    print("###########################################################")
