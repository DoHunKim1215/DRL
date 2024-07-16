import gymnasium


def create_environment_setting(env_name: str) -> dict:
    if env_name == 'CartPole-v1':
        return {
            'env_name': env_name,
            'gamma': 0.99,
            'max_minutes': 20,
            'max_episodes': 10000,
            'goal_mean_100_reward': 475
        }
    elif env_name == 'Pendulum-v1':
        return {
            'env_name': env_name,
            'gamma': 0.99,
            'max_minutes': 20,
            'max_episodes': 500,
            'goal_mean_100_reward': -150
        }
    elif env_name == 'Hopper-v5':
        return {
            'env_name': env_name,
            'gamma': 0.99,
            'max_minutes': 300,
            'max_episodes': 10000,
            'goal_mean_100_reward': 1500
        }
    else:
        assert False, 'No such environment name {}'.format(env_name)


def make_env_fn(env_name, unwrapped=False, inner_wrappers=None, outer_wrappers=None):
    env = gymnasium.make(env_name, render_mode='rgb_array')
    env = env.unwrapped if unwrapped else env

    if inner_wrappers:
        for wrapper in inner_wrappers:
            env = wrapper(env)

    if outer_wrappers:
        for wrapper in outer_wrappers:
            env = wrapper(env)

    return env
