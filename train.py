import os
import random

import numpy as np

from model.model_factory import create_rl_model

from utils.argument_manager import get_args
from utils.plot_manager import plot_result
from utils.utils import get_make_env_fn


if __name__ == '__main__':
    # Get program arguments
    args = get_args()

    # Generate seed array from initial seed
    seeds = []
    random.seed(args.init_seed)
    for _ in range(args.n_case):
        seeds.append(random.randint(0, 2**16 - 1))

    results = []
    agents, best_agent_key, best_eval_score = {}, None, float('-inf')
    make_env_fn, make_env_kwargs = get_make_env_fn(env_name=args.env_name)

    for seed in seeds:
        agent = create_rl_model(args.model_name, args)
        result, final_eval_score, training_time, wallclock_time \
            = agent.train(make_env_fn,
                          make_env_kwargs,
                          seed,
                          args.gamma,
                          args.max_minutes,
                          args.max_episodes,
                          args.goal_mean_100_reward)
        results.append(result)
        agents[seed] = agent
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent_key = seed

    # save training progress data
    np.save(
        os.path.join(args.log_out_dir, 'env_{}_model_{}'.format(args.env_name, args.model_name)),
        np.array(results)
    )

    # Simulate training progression
    agents[best_agent_key].demo_progression()
    # Simulate last training model
    agents[best_agent_key].demo_last()

    plot_result(results, args.model_name, args.fig_out_dir, args.figure_name)
