import argparse
import os


def make_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    model_name = 'VPG'

    # file and directory
    parser.add_argument('--model_out_dir', type=str, default=f'results\\{model_name}\\weights')
    make_dir(f'results\\{model_name}\\weights')
    parser.add_argument('--fig_out_dir', type=str, default=f'results\\{model_name}\\figures')
    make_dir(f'results\\{model_name}\\figures')
    parser.add_argument('--video_out_dir', type=str, default=f'results\\{model_name}\\videos')
    make_dir(f'results\\{model_name}\\videos')
    parser.add_argument('--log_out_dir', type=str, default=f'results\\{model_name}\\logs')
    make_dir(f'results\\{model_name}\\logs')
    parser.add_argument('--figure_name', type=str, default='plot')

    # environment
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--init_seed', type=int, default=13)
    parser.add_argument('--n_case', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=1.00)
    parser.add_argument('--max_minutes', type=int, default=20)
    parser.add_argument('--max_episodes', type=int, default=10000)
    parser.add_argument('--goal_mean_100_reward', type=int, default=475)

    # learning
    parser.add_argument('--strategy_name',
                        type=str,
                        choices=['greedy',
                                 'epsilon_greedy',
                                 'linearly_decaying_epsilon_greedy',
                                 'exponentially_decaying_epsilon_greedy',
                                 'softmax'],
                        default='exponentially_decaying_epsilon_greedy')
    parser.add_argument('--model_name',
                        type=str,
                        choices=['NFQ',
                                 'DQN',
                                 'DDQN',
                                 'DuelingDDQN',
                                 'DuelingDDQN+PER',
                                 'REINFORCE',
                                 'VPG'],
                        default=model_name)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--policy_lr', type=float, default=0.0005)
    parser.add_argument('--value_lr', type=float, default=0.0007)
    parser.add_argument('--n_warmup_batches', type=int, default=5)
    parser.add_argument('--update_target_every_steps', type=int, default=10)
    parser.add_argument('--max_buffer_size', type=int, default=50000)
    parser.add_argument('--max_gradient_norm', type=float, default=float('inf'))
    parser.add_argument('--policy_model_max_gradient_norm', type=float, default=1.0)
    parser.add_argument('--value_model_max_gradient_norm', type=float, default=float('inf'))

    # log
    parser.add_argument('--leave_print_every_n_secs', type=int, default=60)

    args = parser.parse_args()

    return args
