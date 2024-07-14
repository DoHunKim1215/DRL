import gymnasium


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
