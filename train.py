import os
import random

import numpy as np

from model.model_factory import create_rl_model

from utils.argument_manager import get_args
from utils.make_envs import make_envs_fn
from utils.plot_manager import plot_result
from utils.make_env import make_env_fn, create_environment_setting

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

    env_setting = create_environment_setting(args.env_name)

    for seed in seeds:
        agent = create_rl_model(args.model_name, args)
        result, final_eval_score, training_time, wallclock_time \
            = agent.train(make_env_fn,
                          {'env_name': env_setting['env_name']},
                          seed,
                          env_setting['gamma'],
                          env_setting['max_minutes'],
                          env_setting['max_episodes'],
                          env_setting['goal_mean_100_reward']) \
            if args.model_name != 'A2C' else agent.train(make_envs_fn,
                                                         make_env_fn,
                                                         {'env_name': env_setting['env_name']},
                                                         seed,
                                                         env_setting['gamma'],
                                                         env_setting['max_minutes'],
                                                         env_setting['max_episodes'],
                                                         env_setting['goal_mean_100_reward'])

        results.append(result)
        agents[seed] = agent
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent_key = seed

    # save training progress data
    np.save(
        os.path.join(args.log_out_dir, 'env_{}_model_{}'.format(env_setting['env_name'], args.model_name)),
        np.array(results)
    )

    # Simulate training progression
    agents[best_agent_key].demo_progression()
    # Simulate last training model
    agents[best_agent_key].demo_last()

    plot_result(results, args.model_name, env_setting['env_name'], args.fig_out_dir, args.figure_name)
